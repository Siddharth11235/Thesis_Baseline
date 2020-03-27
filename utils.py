import numpy as np
import matplotlib as plt
import librosa
import os
import torch
import torchaudio

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

def mp3_spec(filename):
	x, sr = librosa.load(filename + ".mp3")
	S = librosa.stft(x, 512)
	p = np.angle(S)

	S = np.log1p(np.abs(S))
	return S, sr


def normalize(tensor):
 # Subtract the mean, and scale to the interval [-1,1]
	tensor_minusmean = tensor - tensor.mean()
	return tensor_minusmean / tensor_minusmean.abs().max()



def spec_wav(specgram, filename):
	# Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    librosa.output.write_wav(outfile, x, sr)

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
class Sample_from_Pool(object):
	def __init__(self, max_elements=50):
		self.max_elements = max_elements
		self.cur_elements = 0
		self.items = []

	def __call__(self, in_items):
		return_items = []
		for in_item in in_items:
			if self.cur_elements < self.max_elements:
				self.items.append(in_item)
				self.cur_elements = self.cur_elements + 1
				return_items.append(in_item)
			else:
				if np.random.ranf() > 0.5:
					idx = np.random.randint(0, self.max_elements)
					tmp = copy.copy(self.items[idx])
					self.items[idx] = in_item
					return_items.append(tmp)
				else:
					return_items.append(in_item)
		return return_items


class LambdaLR():
	def __init__(self, epochs, offset, decay_epoch):
		self.epochs = epochs
		self.offset = offset
		self.decay_epoch = decay_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

