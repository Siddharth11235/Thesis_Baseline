from __future__ import print_function
#%matplotlib inline
import itertools
import functools
import argparse
import os
import random
import torchaudio
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import weight_norm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.filters import constant_q as librosa_cqt_fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn import functional as F
import functools
import utils
from dataset import AudioTransformSet
from torch.utils.tensorboard import SummaryWriter


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
OUT_CHANNELS = 32


writer = SummaryWriter("./logdir/experiment_1")

class cycleRandGAN(object):
	def __init__(self, args):
		##The Network
		self.g_AB =  Generator(conv_dim=args.ngf, n_res_blocks=2).cuda() #initialise generator with n mel channels
		
		self.g_BA =  Generator(conv_dim=args.ngf, n_res_blocks=2).cuda() #initialise generator with n mel channels
		self.Da = Discriminator(conv_dim=args.ndf).cuda() #initialize discriminator
		self.Db =  Discriminator(conv_dim=args.ndf).cuda() #initialize discriminator

		
		
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
			ckpt = utils.load_checkpoint('%slatest_rcnn.ckpt' % (args.checkpoint_dir))
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
		train_set=AudioTransformSet(args.dataset_dir+"Joni_Mitchell/files.txt", args.dataset_dir+"Nancy_Sinatra/files.txt", args.seq_len, 
		sampling_rate=22050, augment=True)
		dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)   
		
		
		a_fake_sample = utils.Sample_from_Pool()
		b_fake_sample = utils.Sample_from_Pool()
	
		for epoch in range(self.start_epoch, args.epochs):

			lr = self.g_optimizer.param_groups[0]['lr']
			print('learning rate = %.7f' % lr)

			for i, data in enumerate(dataloader):
				# step

				step = epoch *len(dataloader) + i + 1
				print(step)
			
				a_real = data[0]
				b_real = data[1]
				
				
				a_r_spec = torchaudio.transforms.Spectrogram()(a_real)
				b_r_spec = torchaudio.transforms.Spectrogram()(b_real)




				print("Shape of a-spectrogram: {}".format(a_r_spec.size()))
				print("Shape of b-spectrogram: {}".format(b_r_spec.size()))
				# Generator Computations
				##################################################

				set_grad([self.Da, self.Db], False)
				self.g_optimizer.zero_grad()


				# Forward pass through generators
				##################################################
				a_fake = self.g_AB(b_r_spec.cuda())
				b_fake = self.g_BA(a_r_spec.cuda())


				print("Shape of a-fake spectrogram: {}".format(a_fake.size()))
				print("Shape of b-fake spectrogram: {}".format(b_fake.size()))

				


				a_recon = self.g_AB(b_fake)
				b_recon = self.g_BA(a_fake)


				a_idt = self.g_AB(a_r_spec.cuda())
				b_idt = self.g_BA(b_r_spec.cuda())

				a_idt = self.fft(a_idt).detach()
				b_idt = self.fft(b_idt).detach()

				print("Shape of a_recon spectrogram: {}".format(a_recon.size()))
				print("Shape of b_recon spectrogram: {}".format(b_recon.size()))

				# Identity losses
				###################################################
				a_idt_loss = self.L1(a_idt, a_r_spec) * args.lamda * args.idt_coef
				b_idt_loss = self.L1(b_idt, b_r_spec) * args.lamda * args.idt_coef

				# Adversarial losses
				###################################################
				a_fake_dis = self.Da(a_fake)
				b_fake_dis = self.Db(b_fake)

				print(a_fake_dis.size())
				real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))


				a_gen_loss = self.MSE(a_fake_dis, real_label)
				b_gen_loss = self.MSE(b_fake_dis, real_label)

				# Cycle consistency losses
				###################################################
				a_cycle_loss = self.L1(a_recon, a_r_spec) * args.lamda
				b_cycle_loss = self.L1(b_recon, b_r_spec) * args.lamda

				# Total generators losses
				###################################################
				gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

				# Update generators
				###################################################
				gen_loss.backward(retain_graph=True)
				self.g_optimizer.step()


				# Discriminator Computations
				#################################################


				set_grad([self.Da, self.Db], True)
				self.d_optimizer.zero_grad()

				# Sample from history of generated images
				#################################################
				a_f_spec = Variable(torch.Tensor(a_fake_sample([a_f_spec.cpu().data.numpy()])))
				b_f_spec = Variable(torch.Tensor(b_fake_sample([b_f_spec.cpu().data.numpy()])))
				a_f_spec, b_f_spec = utils.cuda([a_f_spec, b_f_spec])


				print("Shape of a-fake spectrogram: {}".format(a_f_spec.size()))
				print("Shape of b-fake spectrogram: {}".format(b_f_spec.size()))

				print("Shape of a-spectrogram: {}".format(a_r_spec.size()))
				print("Shape of b-spectrogram: {}".format(b_r_spec.size()))


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
				a_dis_loss.backward(retain_graph=True)
				b_dis_loss.backward(retain_graph=True)
				self.d_optimizer.step()

					# every 1000 mini-batches...

				# ...log the running loss
				writer.add_scalar('DisA loss',  a_dis_loss / 1000,
						epoch * len(dataloader) + i)
				writer.add_scalar('DisB loss',  b_dis_loss / 1000,
						epoch * len(dataloader) + i)
				
				writer.add_scalar('Generator loss',  gen_loss / 1000,
						epoch * len(dataloader) + i)


				print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %(epoch, i + 1, len(dataloader), gen_loss,a_dis_loss+b_dis_loss))

			# Override the latest checkpoint
			#######################################################
			utils.save_checkpoint({'epoch': epoch + 1,
								   'Da': self.Da.state_dict(),
								   'Db': self.Db.state_dict(),
								   'Gab': self.g_AB.state_dict(),
								   'Gba': self.g_BA.state_dict(),
								   'd_optimizer': self.d_optimizer.state_dict(),
								   'g_optimizer': self.g_optimizer.state_dict()},
								  '%s/slatest_rcnn.ckpt' % (args.checkpoint_dir))

			# Update learning rates
			########################
			self.g_lr_scheduler.step()
			self.d_lr_scheduler.step()

	def test(self,args):
		test_set=AudioTransformSet(args.test_dir+"Joni_Mitchell/files.txt", args.test_dir+"Nancy_Sinatra/files.txt", args.seq_len, 
		sampling_rate=22050, augment=True)
		dataloader = DataLoader(test_set, batch_size=1, num_workers=4) 
		self.g_BA.eval()
		for i, data in enumerate(dataloader):
			x = data[1]
			x = x.cuda()
			x_spec = self.fft(x).detach()
			y = self.g_BA(x_spec)
			print (y)



		
		


def WNConv1d(*args, **kwargs):
	return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
	return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def conv(in_channels=32,out_channels= 32, kernel_size=(3, 1), stride=2, padding=1, batch_norm=True):
	"""Creates a convolutional layer, with optional batch normalization.
	"""
	layers = []
	conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
						   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
	
	layers.append(conv_layer)

	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	return nn.Sequential(*layers)




def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
	"""Creates a transpose convolutional layer, with optional batch normalization.
	"""
	layers = []
	# append transpose conv layer
	layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
	# optional batch norm layer
	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
	"""Defines a residual block.
	   This adds an input x to a convolutional layer (applied to x) with the same size input and output.
	   These blocks allow a model to learn an effective transformation from one domain to another.
	"""
	def __init__(self, conv_dim):
		super(ResidualBlock, self).__init__()
		# conv_dim = number of inputs
		
		# define two convolutional layers + batch normalization that will act as our residual function, F(x)
		# layers should have the same shape input as output; I suggest a kernel_size of 3
		
		self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
								kernel_size=3, stride=1, padding=1, batch_norm=True)
		
		self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
							   kernel_size=3, stride=1, padding=1, batch_norm=True)
		
	def forward(self, x):
		# apply a ReLu activation the outputs of the first layer
		# return a summed output, x + resnet_block(x)
		out_1 = F.relu(self.conv_layer1(x))
		out_2 = x + self.conv_layer2(out_1)
		return out_2




class Generator(nn.Module):
	def __init__(self, conv_dim=16, n_res_blocks=2):
		super(Generator, self).__init__()

		# 2-D CNN
		self.conv1 = nn.Conv2d(1, conv_dim, kernel_size=(3, 1)	, stride=1, padding=0)
		self.LeakyReLU = nn.LeakyReLU(0.2)

		self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=(3, 1)	, stride=1, padding=0)

		# Set the random parameters to be constant.
		weight = torch.randn(self.conv1.weight.data.shape)
		self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
		bias = torch.zeros(self.conv1.bias.data.shape)
		self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

		res_layers = []
		for layer in range(n_res_blocks):
			res_layers.append(ResidualBlock(conv_dim*2))
		# use sequential to create these layers
		self.res_blocks = nn.Sequential(*res_layers)

		# 3. Define the decoder part of the generator
		# two transpose convolutional layers and a third that looks a lot like the initial conv layer
		self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
		# no batch norm on last layer
		self.deconv2 = deconv(conv_dim, 1, 4, batch_norm=False)

	def forward(self, x_delta):
		out = self.LeakyReLU(self.conv1(x_delta))
		out = self.LeakyReLU(self.conv2(out))
		out = self.res_blocks(out)
		out = F.relu(self.deconv1(out))
		out = F.tanh(self.deconv2(out))
		return out




class Discriminator(nn.Module):
	
	def __init__(self, conv_dim=16):
		super(Discriminator, self).__init__()

		# Define all convolutional layers
		# Should accept an RGB image as input and output a single value

		# Convolutional layers, increasing in depth
		# first layer has *no* batchnorm
		self.conv1 = conv(1, conv_dim, 4, batch_norm=False) # x, y = 64, depth 64
		self.conv2 = conv(conv_dim, conv_dim*2, 4) # (32, 32, 64)
		
		
		# Classification layer
		self.conv5 = conv(conv_dim*2, 1, 4, stride=1, batch_norm=False)

	def forward(self, x):
		# relu applied to all conv layers but last
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		# last, classification layer
		out = self.conv5(out)
		print (out.size())
		return out



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


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def init_network(net, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)
	weights_init(net)
	return net



def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
								 norm_layer = nn.BatchNorm2d, bias = False):
	return nn.Sequential(
		nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
		norm_layer(out_dim), nn.LeakyReLU(0.2,True))


	

