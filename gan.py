from __future__ import print_function
#%matplotlib inline
import itertools
import functools
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import AudioTransformSet
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ops import conv_norm_lrelu, get_norm_layer, init_network, conv_norm_relu, dconv_norm_relu, ResidualBlock
from torch.nn import functional as F
import functools
import utils


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Root directory for dataset
dataroot = "./data"

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# song size of training images.
song_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32




class cycleRandGAN(object):
	def __init__(self, args):
		##The Network
		self.g_AB = define_Gen(input_size=80, ngf=32, norm='batch', use_dropout=False, gpu_ids=[0])
		self.g_BA = define_Gen(input_size=80, ngf=32, norm='batch', use_dropout=False, gpu_ids=[0])
		self.Da = define_Dis(input_size=80, ndf=16, n_layers=3, norm='batch', gpu_ids=[0])
		self.Db = define_Dis(input_size=80, ndf=16, n_layers=3, norm='batch', gpu_ids=[0])


		#The Losses
		self.MSE = nn.MSELoss()
		self.L1 = nn.L1Loss()

		# Optimizers
		#####################################################
		self.g_optimizer = torch.optim.Adam(itertools.chain(self.g_AB.parameters(),self.g_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
		self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))
		

		self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
		self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

		
		#Load potential checkpoint
		if not os.path.isdir(args.checkpoint_dir):
			os.makedirs(args.checkpoint_dir)

		try:
			ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
			self.start_epoch = ckpt['epoch']
			self.Da.load_state_dict(ckpt['Da'])
			self.Db.load_state_dict(ckpt['Db'])
			self.g_AB.load_state_dict(ckpt['Gab'])
			self.g_BA.load_state_dict(ckpt['Gba'])
			self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
			self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
		except:
			print(' [*] No checkpoint!')
			self.start_epoch = 0
	def train(self,args):
		con_dataloader = DataLoader(
		AudioTransformSet(args.dataset_dir+"Joni_Mitchell/files.txt", 8192, 22050, augment=True)),
		style_dataloader = DataLoader(
		AudioTransformSet(args.dataset_dir+"Nancy_Sinatra/files.txt", 8192, 22050, augment=True)),

		a_fake_sample = utils.Sample_from_Pool()
		b_fake_sample = utils.Sample_from_Pool()
	
		for epoch in range(self.start_epoch, args.epochs):

			lr = self.g_optimizer.param_groups[0]['lr']
			print('learning rate = %.7f' % lr)

			for i, (a_real, b_real) in enumerate(zip(con_dataloader, style_dataloader)):
				# step
				step = epoch * min(len(con_dataloader), len(style_dataloader)) + i + 1

				# Generator Computations
				##################################################

				set_grad([self.Da, self.Db], False)
				self.g_optimizer.zero_grad()

				print (a_real[0])


				# Forward pass through generators
				##################################################
				a_fake = self.g_AB(b_real)
				b_fake = self.g_BA(a_real)

				a_recon = self.g_AB(b_fake)
				b_recon = self.g_BA(a_fake)

				a_idt = self.g_AB(a_real)
				b_idt = self.g_BA(b_real)

				# Identity losses
				###################################################
				a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
				b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

				# Adversarial losses
				###################################################
				a_fake_dis = self.Da(a_fake)
				b_fake_dis = self.Db(b_fake)

				real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

				a_gen_loss = self.MSE(a_fake_dis, real_label)
				b_gen_loss = self.MSE(b_fake_dis, real_label)

				# Cycle consistency losses
				###################################################
				a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
				b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

				# Total generators losses
				###################################################
				gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

				# Update generators
				###################################################
				gen_loss.backward()
				self.g_optimizer.step()


				# Discriminator Computations
				#################################################

				set_grad([self.Da, self.Db], True)
				self.d_optimizer.zero_grad()

				# Sample from history of generated images
				#################################################
				a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
				b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
				a_fake, b_fake = utils.cuda([a_fake, b_fake])

				# Forward pass through discriminators
				################################################# 
				a_real_dis = self.Da(a_real)
				a_fake_dis = self.Da(a_fake)
				b_real_dis = self.Db(b_real)
				b_fake_dis = self.Db(b_fake)
				real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
				fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

				# Discriminator losses
				##################################################
				a_dis_real_loss = self.MSE(a_real_dis, real_label)
				a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
				b_dis_real_loss = self.MSE(b_real_dis, real_label)
				b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

				# Total discriminators losses
				a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
				b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

				# Update discriminators
				##################################################
				a_dis_loss.backward()
				b_dis_loss.backward()
				self.d_optimizer.step()

				print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
											(epoch, i + 1, min(len(a_loader), len(b_loader)),
															gen_loss,a_dis_loss+b_dis_loss))

			# Override the latest checkpoint
			#######################################################
			utils.save_checkpoint({'epoch': epoch + 1,
								   'Da': self.Da.state_dict(),
								   'Db': self.Db.state_dict(),
								   'Gab': self.g_AB.state_dict(),
								   'Gba': self.g_BA.state_dict(),
								   'd_optimizer': self.d_optimizer.state_dict(),
								   'g_optimizer': self.g_optimizer.state_dict()},
								  '%s/latest.ckpt' % (args.checkpoint_dir))

			# Update learning rates
			########################
			self.g_lr_scheduler.step()
			self.d_lr_scheduler.step()
		
		


def WNConv1d(*args, **kwargs):
	return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
	return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
	def __init__(self, dim, dilation=1):
		super().__init__()
		self.block = nn.Sequential(
			nn.LeakyReLU(0.2),
			nn.ReflectionPad1d(dilation),
			WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
			nn.LeakyReLU(0.2),
			WNConv1d(dim, dim, kernel_size=1),
		)
		self.shortcut = WNConv1d(dim, dim, kernel_size=1)

	def forward(self, x):
		return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
	def __init__(self, input_size, ngf, n_residual_layers):
		super().__init__()
		ratios = [8, 8, 2, 2]
		self.hop_length = np.prod(ratios)
		mult = int(2 ** len(ratios))

		model = [
			nn.ReflectionPad1d(3),
			WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
		]

		# Upsample to raw audio scale
		for i, r in enumerate(ratios):
			model += [
				nn.LeakyReLU(0.2),
				WNConvTranspose1d(
					mult * ngf,
					mult * ngf // 2,
					kernel_size=r * 2,
					stride=r,
					padding=r // 2 + r % 2,
					output_padding=r % 2,
				),
			]

			for j in range(n_residual_layers):
				model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

			mult //= 2

		model += [
			nn.LeakyReLU(0.2),
			nn.ReflectionPad1d(3),
			WNConv1d(ngf, 1, kernel_size=7, padding=0),
			nn.Tanh(),
		]

		self.model = nn.Sequential(*model)
		self.apply(weights_init)

	def forward(self, x):
		return self.model(x)


class NLayerDiscriminator(nn.Module):
	def __init__(self, ndf, n_layers, downsampling_factor):
		super().__init__()
		model = nn.ModuleDict()

		model["layer_0"] = nn.Sequential(
			nn.ReflectionPad1d(7),
			WNConv1d(1, ndf, kernel_size=15),
			nn.LeakyReLU(0.2, True),
		)

		nf = ndf
		stride = downsampling_factor
		for n in range(1, n_layers + 1):
			nf_prev = nf
			nf = min(nf * stride, 1024)

			model["layer_%d" % n] = nn.Sequential(
				WNConv1d(
					nf_prev,
					nf,
					kernel_size=stride * 10 + 1,
					stride=stride,
					padding=stride * 5,
					groups=nf_prev // 4,
				),
				nn.LeakyReLU(0.2, True),
			)

		nf = min(nf * 2, 1024)
		model["layer_%d" % (n_layers + 1)] = nn.Sequential(
			WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.2, True),
		)

		model["layer_%d" % (n_layers + 2)] = WNConv1d(
			nf, 1, kernel_size=3, stride=1, padding=1
		)

		self.model = model

	def forward(self, x):
		results = []
		for key, layer in self.model.items():
			x = layer(x)
			results.append(x)
		return results


class Discriminator(nn.Module):
	def __init__(self, num_D, ndf, n_layers, downsampling_factor):
		super().__init__()
		self.model = nn.ModuleDict()
		for i in range(num_D):
			self.model[f"disc_{i}"] = NLayerDiscriminator(
				ndf, n_layers, downsampling_factor
			)

		self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
		self.apply(weights_init)

	def forward(self, x):
		results = []
		for key, disc in self.model.items():
			results.append(disc(x))
			x = self.downsample(x)
		return results


def define_Dis(input_size, ndf, n_layers=3, norm='batch', gpu_ids=[0]):
	dis_net = None
	norm_layer = get_norm_layer(norm_type=norm)
	if type(norm_layer) == functools.partial:
		use_bias = norm_layer.func == nn.InstanceNorm2d
	else:
		use_bias = norm_layer == nn.InstanceNorm2d


	dis_net = Discriminator(input_size, ndf, n_layers, downsampling_factor=4)
	

	#raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

	return init_network(dis_net, gpu_ids)




def define_Gen(input_size, ngf, norm='batch', use_dropout=False, gpu_ids=[0]):
	gen_net = None
	norm_layer = get_norm_layer(norm_type=norm)

	
	gen_net = Generator(input_size, ngf, n_residual_layers=3)
	
	#else:
	#	raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

	return init_network(gen_net, gpu_ids)


def set_grad(nets, requires_grad=False):
	for net in nets:
		for param in net.parameters():
			param.requires_grad = requires_grad

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

