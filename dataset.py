import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import utils
import torchaudio



def files_to_list(filename):
	"""
	Takes a text file of filenames and makes a list of filenames
	"""
	with open(filename, encoding="utf-8") as f:
		files = f.readlines()

	files = [f.rstrip() for f in files]
	return files

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

song_frame = pd.read_csv('data/content.csv')
csv_file = 'data/content.csv'


class AudioTransformSet(Dataset):
	"""Audio transform dataset.

	This is the main class that calculates the spectrogram and returns the
	spectrogram, audio pair.
	"""

	def __init__(self, content_files, style_files, segment_length, sampling_rate, augment=True):
		self.sampling_rate = sampling_rate
		self.segment_length = segment_length
		self.content_files = files_to_list(content_files)
		self.content_files = [Path(content_files).parent / x for x in self.content_files]
		
		self.style_files = files_to_list(style_files)
		self.style_files = [Path(style_files).parent / x for x in self.style_files]			

		random.seed(1234)
		random.shuffle(self.content_files)
		self.augment = augment

	def __getitem__(self, index):
		# Read content
		con_filename = self.content_files[index]
		content, sampling_rate = self.load_wav_to_torch(con_filename)
		# Take content segment
		if content.size(0) >= self.segment_length:
			max_content_start = content.size(0) - self.segment_length
			content_start = random.randint(0, max_content_start)
			content = content[content_start : content_start + self.segment_length]
		else:
			content = F.pad(
				content, (0, self.segment_length - content.size(0)), "constant").data

		# audio = audio / 32768.0

		# Read style
		style_filename = self.style_files[index]
		style, sampling_rate = self.load_wav_to_torch(style_filename)
		# Take content segment
		if style.size(0) >= self.segment_length:
			max_style_start = style.size(0) - self.segment_length
			style_start = random.randint(0, max_style_start)
			style = style[style_start : style_start + self.segment_length]
		else:
			style = F.pad(
				style, (0, self.segment_length - style.size(0)), "constant"
			).data

		# audio = audio / 32768.0


		return content.unsqueeze(0), style.unsqueeze(0)

	def __len__(self):
		return len(self.content_files)

	def load_wav_to_torch(self, con_path):
		"""
		Loads wavdata into torch array
		"""
		con_data, sampling_rate = load(con_path, sr=self.sampling_rate)
		con_data = 0.95 * normalize(con_data)

		if self.augment:
			amplitude = np.random.uniform(low=0.3, high=1.0)
			con_data = con_data * amplitude

		return torch.from_numpy(con_data).float(), sampling_rate



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	

	dataloader = DataLoader(
		AudioTransformSet("./Baseline_Data/Content/Joni_Mitchell/files.txt", "./Baseline_Data/Content/Nancy_Sinatra/files.txt", 8192, 22050, augment=True))

	for index, data in enumerate(dataloader):
		
		specgram = torchaudio.transforms.Spectrogram()(data[1])

		print("Shape of spectrogram: {}".format(specgram.size()))
		print(specgram.size())

		plt.figure()
		plt.imshow(specgram.log2()[:,:,:].numpy())

		plt.show()

		if index == 0:
			break


	
