from __future__ import print_function
#%matplotlib inline
import itertools
import functools
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
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
import functools
import utils
from torch.utils.tensorboard import SummaryWriter
import pickle
from trainset import trainingDataset
import dataset


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


writer = SummaryWriter("./logdir/experiment_2_wgan")

class cycleRandGAN(object):
	def __init__(self, args):
		##The Network
		self.g_AB =  Generator(conv_dim=args.ngf, n_res_blocks=2).cuda() #initialise generator with n mel channels
		
		self.g_BA =  Generator(conv_dim=args.ngf, n_res_blocks=2).cuda() #initialise generator with n mel channels
		self.Da = Discriminator(conv_dim=args.ndf).cuda() #initialize discriminator
		self.Db =  Discriminator(conv_dim=args.ndf).cuda() #initialize discriminator

		#Loss type: (Wasserstein or Basic)
		self.loss_type = args.loss
		
		#The Losses
		self.MSE = nn.MSELoss()
		self.L1 = nn.L1Loss()

		# Optimizers
		#####################################################
		self.g_optimizer = torch.optim.Adam(itertools.chain(self.g_AB.parameters(),self.g_BA.parameters()), lr=args.lr, betas=(0.5, 0.99))
		self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.99))
		

		self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
		self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

		
		#Load potential checkpoint
		if not os.path.isdir(args.checkpoint_dir):
			os.makedirs(args.checkpoint_dir)

		try:
			ckpt = utils.load_checkpoint('./checkpoints/w_gan_3.ckpt')
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



	#Criterion for WGAN
		self.criterionGAN = WassersteinGANLoss()
		self.wgan_n_critic = 10
		self.wgan_clamp_lower = 0.01
		self.wgan_clamp_upper = 0.01

	#Load audio data
		logf0s_normalization = np.load("./cache_check/logf0s_normalization.npz")
		self.log_f0s_mean_A = logf0s_normalization['mean_A']
		self.log_f0s_std_A = logf0s_normalization['std_A']
		self.log_f0s_mean_B = logf0s_normalization['mean_B']
		self.log_f0s_std_B = logf0s_normalization['std_B']

		mcep_normalization = np.load("./cache_check/mcep_normalization.npz")
		self.coded_sps_A_mean = mcep_normalization['mean_A']
		self.coded_sps_A_std = mcep_normalization['std_A']
		self.coded_sps_B_mean = mcep_normalization['mean_B']
		self.coded_sps_B_std = mcep_normalization['std_B']

	#Definition for a WGAN Dis loss

	def backward_D_wasserstein(self, netD, real, fake):

		pred_real = netD(real.cuda())
		pred_fake = netD(fake.cuda())
		loss_D = self.criterionGAN(pred_fake, pred_real, generator_loss=False)
		return loss_D

	

	def train(self,args):

		self.dataset_A = loadPickleFile("cache_check/coded_sps_A_norm.pickle")
		self.dataset_B = loadPickleFile("cache_check/coded_sps_B_norm.pickle")
		n_samples = len(self.dataset_A)

		dataset = trainingDataset(datasetA=self.dataset_A,
									  datasetB=self.dataset_B,
									  n_frames=128)
		train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,						   drop_last=False, num_workers=4)
		
		a_fake_sample = utils.ImagePool(50)
		b_fake_sample = utils.ImagePool(50)
		
	
		for epoch in range(self.start_epoch, args.epochs):

			lr = self.g_optimizer.param_groups[0]['lr']
			print('learning rate = %.7f' % lr)

			for i, (a_real, b_real) in enumerate(train_loader):
				# step
				a_real = a_real.float()
				b_real = b_real.float()

		

				step = epoch *len(train_loader) + i + 1
				print(step)
			
				# Generator Computations
				##################################################

				set_grad([self.Da, self.Db], False)
				self.g_optimizer.zero_grad()


				# Forward pass through generators
				##################################################
				a_fake = self.g_AB(a_real.cuda())
				b_fake = self.g_BA(b_real.cuda())

				a_recon = self.g_AB(b_fake)
				b_recon = self.g_BA(a_fake)


				a_idt = self.g_AB(a_real.cuda())
				b_idt = self.g_BA(b_real.cuda())


				# Identity losses
				###################################################
				a_idt_loss = self.L1(a_idt, a_real.cuda()) * args.lamda * args.idt_coef
				b_idt_loss = self.L1(b_idt, b_real.cuda()) * args.lamda * args.idt_coef

				if self.loss_type=='lsgan':

					# Adversarial losses
					###################################################
					a_fake_dis = self.Da(a_fake)
					b_fake_dis = self.Db(b_fake)

					real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))


					a_gen_loss = self.MSE(a_fake_dis, real_label)
					b_gen_loss = self.MSE(b_fake_dis, real_label)
				elif self.loss_type=='wgan':

					# Wasserstein-GAN loss
					# G_A(A)
					a_gen_loss = self.criterionGAN(b_fake, generator_loss=True)

					# G_B(B)
					b_gen_loss = self.criterionGAN(a_fake, generator_loss=True)

				# Cycle consistency losses
				###################################################
				a_cycle_loss = self.L1(a_recon, a_real.cuda()) * args.lamda
				b_cycle_loss = self.L1(b_recon, b_real.cuda()) * args.lamda

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
				a_fake = a_fake_sample.query(a_fake)
				b_fake = b_fake_sample.query(b_fake)
				a_fake, b_fake = utils.cuda([a_fake, b_fake])

				print ("A_R_Size",a_fake.size())
				print ("B_R_Size",b_fake.size())  


				if self.loss_type=='lsgan':

					# Forward pass through discriminators
					################################################# 
					a_real_dis = self.Da(a_real.cuda())
					a_fake_dis = self.Da(a_fake)
					b_real_dis = self.Db(b_real.cuda())
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

				elif self.loss_type=='wgan':
					for i_critic in range(self.wgan_n_critic):
					# Clip the parameters for k-Lipschitz continuity
						for p in self.Da.parameters():
							p.data.clamp_(self.wgan_clamp_lower, self.wgan_clamp_upper)
						for p in self.Db.parameters():
							p.data.clamp_(self.wgan_clamp_lower, self.wgan_clamp_upper)
					#D_A
						a_dis_loss = self.backward_D_wasserstein(self.Da, a_real.cuda(), a_fake)
						# D_B
						b_dis_loss = self.backward_D_wasserstein(self.Db, b_real.cuda(), b_fake)

				# Update discriminators
				##################################################
				a_dis_loss.backward(retain_graph=True)
				b_dis_loss.backward(retain_graph=True)
				self.d_optimizer.step()
				
				writer.add_scalar('DisA loss',  a_dis_loss,
						epoch * len(train_loader) + i)
				writer.add_scalar('DisB loss',  b_dis_loss,
						epoch * len(train_loader) + i)
				
				writer.add_scalar('Generator loss',  gen_loss / 1000,
						epoch * len(train_loader) + i)



				print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %(epoch, i + 1, len(train_loader), gen_loss,a_dis_loss+b_dis_loss))
				
			# Override the latest checkpoint
			#######################################################
			utils.save_checkpoint({'epoch': epoch + 1,
								   'Da': self.Da.state_dict(),
								   'Db': self.Db.state_dict(),
								   'Gab': self.g_AB.state_dict(),
								   'Gba': self.g_BA.state_dict(),
								   'd_optimizer': self.d_optimizer.state_dict(),
								   'g_optimizer': self.g_optimizer.state_dict()},
								  '%s/w_gan_2.ckpt' % (args.checkpoint_dir))

			# Update learning rates
			########################
			self.g_lr_scheduler.step()
			self.d_lr_scheduler.step()

	#Convert song from singer A to singer B
	def validation_for_A_dir(self):
		num_mcep = 24
		sampling_rate = 16000
		frame_period = 5.0
		n_frames = 128
		validation_A_dir = './Baseline_Data/Test/Joni_Mitchell/'
		output_A_dir = './Baseline_Data/Test/Don_Mclean/'


		try:
			ckpt = utils.load_checkpoint('./checkpoints/w_gan_3.ckpt')
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

		print("Generating Validation Data B from A...")
		for file in os.listdir(validation_A_dir):
			filePath = os.path.join(validation_A_dir, file)
			wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
			wav = dataset.wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period,multiple=4)



			f0, timeaxis, sp, ap = dataset.world_decompose(
				wav=wav, fs=sampling_rate, frame_period=frame_period)
			f0_converted = dataset.pitch_conversion(f0=f0, mean_log_src=self.log_f0s_mean_A, std_log_src=self.log_f0s_std_A, mean_log_target=self.log_f0s_mean_B, std_log_target=self.log_f0s_std_B)

			coded_sp = dataset.world_encode_spectral_envelop(
				sp=sp, fs=sampling_rate, dim=num_mcep)
			coded_sp_transposed = coded_sp.T
			coded_sp_norm = (coded_sp_transposed -
							 self.coded_sps_A_mean) / self.coded_sps_A_std
			coded_sp_norm = np.array([coded_sp_norm])

			coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
			

			coded_sp_converted_norm = self.g_AB(coded_sp_norm)
			coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
			coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)

			coded_sp_converted = coded_sp_converted_norm * (self.coded_sps_B_std + self.coded_sps_B_mean)

			coded_sp_converted = coded_sp_converted.T
			coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

			decoded_sp_converted = dataset.world_decode_spectral_envelop(
				coded_sp=coded_sp_converted, fs=sampling_rate)
			wav_transformed = dataset.world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate, frame_period=frame_period)
			librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
									 y=wav_transformed,
									 sr=sampling_rate)




def savePickle(self, variable, fileName):
	with open(fileName, 'wb') as f:
		pickle.dump(variable, f)

def loadPickleFile(fileName):
	with open(fileName, 'rb') as f:
		return pickle.load(f)		
		

#Wasserstein loss
class WassersteinGANLoss(nn.Module):
	"""WassersteinGANLoss
	ref: Wasserstein GAN (https://arxiv.org/abs/1701.07875)
	"""
	def __init__(self):
		super(WassersteinGANLoss, self).__init__()

	def __call__(self, fake, real=None, generator_loss=True):
		if generator_loss:
			wloss = fake.mean()
		else:
			wloss = torch.abs(real.mean() - fake.mean())
		return wloss


def conv(in_channels=32,out_channels= 32, kernel_size=2, stride=1, padding=1, batch_norm=True):
	"""Creates a convolutional layer, with optional batch normalization.
	"""
	layers = []
	conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
	
	layers.append(conv_layer)

	if batch_norm:
		layers.append(nn.BatchNorm1d(out_channels))
	return nn.Sequential(*layers)




def deconv(in_channels, out_channels, kernel_size=2, stride=1, padding=1, batch_norm=True):
	"""Creates a transpose convolutional layer, with optional batch normalization.
	"""
	layers = []
	# append transpose conv layer
	layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
	# optional batch norm layer
	if batch_norm:
		layers.append(nn.BatchNorm1d(out_channels))
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
		
		self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)
		
		self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
							   kernel_size=3, stride=1, padding=1, batch_norm=True)
		
	def forward(self, x):
		# apply a ReLu activation the outputs of the first layer
		# return a summed output, x + resnet_block(x)
		out_1 = F.relu(self.conv_layer1(x))
		out_2 = x + self.conv_layer2(out_1)
		return out_2




class Generator(nn.Module):
	def __init__(self, conv_dim=16, n_res_blocks=4):
		super(Generator, self).__init__()

		# 1. 1-D CNN
		self.conv1 = nn.Conv1d(24, conv_dim, 2	, stride=1, padding=0)
		self.LeakyReLU = nn.LeakyReLU(0.2)

		self.conv2 = nn.Conv1d(conv_dim, conv_dim*2, kernel_size=2	, stride=1, padding=0)

		# Set the random parameters to be constant.
		weight = torch.randn(self.conv1.weight.data.shape)
		self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
		bias = torch.zeros(self.conv1.bias.data.shape)
		self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

		# 2. Res Block
		res_layers = []
		for layer in range(n_res_blocks):
			res_layers.append(ResidualBlock(conv_dim*2))
		# use sequential to create these layers
		self.res_blocks = nn.Sequential(*res_layers)

		# 3. Define the decoder part of the generator
		self.deconv1 = deconv(conv_dim*2, conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)
		# no batch norm on last layer
		self.deconv2 = deconv(conv_dim, 24, kernel_size=3, stride=1, padding=0, batch_norm=False)

	def forward(self, x_delta):
		out = self.LeakyReLU(self.conv1(x_delta))
		out = self.LeakyReLU(self.conv2(out))
		out = self.res_blocks(out)
		out = F.relu(self.deconv1(out))
		out = torch.tanh(self.deconv2(out))
		return out



class Discriminator(nn.Module):
	
	def __init__(self, conv_dim=32):
		super(Discriminator, self).__init__()

		# Define all convolutional layers

		# Convolutional layers, increasing in depth
		self.conv1 = conv(24, conv_dim, 2, batch_norm=False) # x, y = 64, depth 64
		nn.BatchNorm2d(conv_dim),
		nn.LeakyReLU(negative_slope=0.2, inplace=True),
		self.conv2 = conv(conv_dim, conv_dim*2, 2) # (32, 32, 64)
		nn.BatchNorm2d(conv_dim*2),
		nn.LeakyReLU(negative_slope=0.2, inplace=True),
		
		
		# Classification layer
		self.conv3 = conv(conv_dim*2, 1, 2, stride=1, batch_norm=False)

	def forward(self, x):
		# relu applied to all conv layers but last
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		# last, classification layer
		out = self.conv3(out)
		return out

##Helper functions

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
