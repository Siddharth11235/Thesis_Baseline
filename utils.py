import numpy as np
import random
import matplotlib as plt
import librosa
from librosa.core import load
import os
import torch
import torchaudio
import copy
from torch.autograd import Variable


def mkdir(paths):
	for path in paths:
		if not os.path.isdir(path):
			os.makedirs(path)

# To make cuda tensor
def cuda(xs):
	if torch.cuda.is_available():
		if not isinstance(xs, (list, tuple)):
			return xs.cuda()
		else:
			return [x.cuda() for x in xs]



def create_link(dataset_dir):
	dirs = {}
	dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
	dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
	dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
	dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
	mkdir(dirs.values())

	for key in dirs:
		try:
			os.remove(os.path.join(dirs[key], 'Link'))
		except:
			pass
		os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
				   os.path.join(dirs[key], 'Link'))

	return dirs

def mp3_spec_file(filename):
	x, sr = load(filename)
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)

	S = np.log1p(np.abs(S))
	return S, sr

def mp3_spec(x):
	S =librosa.cqt(x.numpy(), sr=22050)
	S = np.abs(S)
	S = np.log1p(S)
	return torch.from_numpy(S).float().cuda()


def normalize(tensor):
 # Subtract the mean, and scale to the interval [-1,1]
	tensor_minusmean = tensor - tensor.mean()
	return tensor_minusmean / tensor_minusmean.abs().max()



def spec_wav(specgram, filename):
	# Return the all-zero vector with the same shape of `a_content`
    a = np.exp(specgram.cpu().detach().numpy()) - 1
    p = 2 * np.pi * np.random.random_sample(specgram.shape) - np.pi
    
    x = librosa.griffinlim(a)
    p = np.angle(librosa.stft(x, 400))
    librosa.output.write_wav(filename, x, sr=22050)

def compute_content_loss(a_C, a_G):
	"""
	Compute the content cost
	Arguments:
	a_C -- tensor of dimension (1, n_C, n_H, n_W)
	a_G -- tensor of dimension (1, n_C, n_H, n_W)
	Returns:
	J_content -- scalar that you compute using equation 1 above
	"""
	m, n_C, n_H, n_W = a_G.shape

	# Reshape a_C and a_G to the (m * n_C, n_H * n_W)
	a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
	a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

	# Compute the cost
	J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

	return J_content


def gram(A):
	"""
	Argument:
	A -- matrix of shape (n_C, n_L)
	Returns:
	GA -- Gram matrix of shape (n_C, n_C)
	"""
	GA = torch.matmul(A, A.t())

	return GA


def gram_over_time_axis(A):
	"""
	Argument:
	A -- matrix of shape (1, n_C, n_H, n_W)
	Returns:
	GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
	"""
	m, n_C, n_H, n_W = A.shape

	# Reshape the matrix to the shape of (n_C, n_L)
	# Reshape a_C and a_G to the (m * n_C, n_H * n_W)
	A_unrolled = A.view(m * n_C * n_H, n_W)
	GA = torch.matmul(A_unrolled, A_unrolled.t())

	return GA


# To store 50 generated files in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images



class LambdaLR():
	def __init__(self, epochs, offset, decay_epoch):
		self.epochs = epochs
		self.offset = offset
		self.decay_epoch = decay_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


def save_checkpoint(state, save_path):
	torch.save(state, save_path)

def load_checkpoint(ckpt_path, map_location=None):
	ckpt = torch.load(ckpt_path, map_location=map_location)
	print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
	return ckpt