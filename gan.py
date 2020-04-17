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
from opt import Generator, Discriminator, set_grad, Audio2Mel


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


writer = SummaryWriter("./logdir/experiment_1")

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
			ckpt = utils.load_checkpoint('%slatest.ckpt' % (args.checkpoint_dir))
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


				set_grad([self.Da, self.Db], True)
				self.d_optimizer.zero_grad()

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
								  '%s/latest.ckpt' % (args.checkpoint_dir))

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