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

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



class cycleRandGAN(object):
	def __init__(self, args):
		##The Network
		self.g_AB =  Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda() #initialise generator with n mel channels
		
		self.g_BA = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda() #initialise generator with n mel channels
		self.Da = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor).cuda() #initialize discriminator
		self.Db = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor).cuda() #initialize discriminator

		self.fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda()
		
		
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
			self.g_AB.load_state_;dict(ckpt['Gab'])
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
			
				a_real = data[0]
				b_real = data[1]
				
				
				a_real = a_real.cuda()
				b_real = b_real.cuda()


				a_r_spec = self.fft(a_real).detach()
				b_r_spec = self.fft(a_real).detach()



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

				a_f_spec = self.fft(a_fake).detach()
				b_f_spec = self.fft(b_fake).detach()

				print("Shape of a-fake spectrogram: {}".format(a_f_spec.size()))
				print("Shape of b-fake spectrogram: {}".format(b_f_spec.size()))

				


				a_recon = self.g_AB(b_f_spec)
				b_recon = self.g_BA(a_f_spec)

				a_recon_spec = self.fft(a_recon).detach()
				b_recon_spec = self.fft(b_recon).detach()

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

				print(a_fake_dis[2][6].size())
				real_label = utils.cuda(Variable(torch.ones(a_fake_dis[2][6].size())))


				a_gen_loss = self.MSE(a_fake_dis[2][6], real_label)
				b_gen_loss = self.MSE(b_fake_dis[2][6], real_label)

				# Cycle consistency losses
				###################################################
				a_cycle_loss = self.L1(a_recon_spec, a_r_spec) * args.lamda
				b_cycle_loss = self.L1(b_recon_spec, b_r_spec) * args.lamda

				# Total generators losses
				###################################################
				gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

				# Update generators
				###################################################
				gen_loss.backward(retain_graph=True)
				self.g_optimizer.step()


				# Discriminator Computations
				#################################################

				print("We're in the discriminator")

				set_grad([self.Da, self.Db], True)
				self.d_optimizer.zero_grad()
				print("We're past the grad of the discriminator")

				# Sample from history of generated images
				#################################################
				a_f_spec = Variable(torch.Tensor(a_fake_sample([a_f_spec.cpu().data.numpy()])[0]))
				b_f_spec = Variable(torch.Tensor(b_fake_sample([b_f_spec.cpu().data.numpy()])[0]))
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
				real_label = utils.cuda(Variable(torch.ones(a_real_dis[2][6].size())))
				fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis[2][6].size())))

				# Discriminator losses
				##################################################
				a_dis_real_loss = self.MSE(a_real_dis[2][6], real_label)
				a_dis_fake_loss = self.MSE(a_fake_dis[2][6], fake_label)
				b_dis_real_loss = self.MSE(b_real_dis[2][6], real_label)
				b_dis_fake_loss = self.MSE(b_fake_dis[2][6], fake_label)

				# Total discriminators losses
				a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
				b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

				# Update discriminators
				##################################################
				a_dis_loss.backward(retain_graph=True)
				b_dis_loss.backward(retain_graph=True)
				self.d_optimizer.step()

				print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
											(epoch, i + 1, len(dataloader),
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


class Audio2Mel(nn.Module):
	def __init__(
		self,
		n_fft=1024,
		hop_length=256,
		win_length=1024,
		sampling_rate=22050,
		n_mel_channels=80,
		mel_fmin=0.0,
		mel_fmax=None,
	):
		super().__init__()
		##############################################
		# FFT Parameters							  #
		##############################################
		window = torch.hann_window(win_length).float()
		mel_basis = librosa_mel_fn(
			sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
		)
		mel_basis = torch.from_numpy(mel_basis).float()
		self.register_buffer("mel_basis", mel_basis)
		self.register_buffer("window", window)
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.win_length = win_length
		self.sampling_rate = sampling_rate
		self.n_mel_channels = n_mel_channels

	def forward(self, audio):
		p = (self.n_fft - self.hop_length) // 2
		audio = F.pad(audio, (p, p), "reflect").squeeze(1)
		fft = torch.stft(
			audio,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			win_length=self.win_length,
			window=self.window,
			center=False,
		)
		real_part, imag_part = fft.unbind(-1)
		magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
		mel_output = torch.matmul(self.mel_basis, magnitude)
		log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
		return log_mel_spec

