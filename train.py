import os
from argparse import ArgumentParser
import test as tst
import gan as md
from utils import create_link

# To get arguments from commandline
def get_args():
	parser = ArgumentParser()
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--decay_epoch', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--lamda', type=int, default=10)
	parser.add_argument('--lr', type=float, default=.02)
	parser.add_argument('--gpu_ids', type=str, default='0')
	parser.add_argument("--seq_len", type=int, default=8192)
	parser.add_argument('--idt_coef', type=float, default=0.5)
	parser.add_argument('--training', type=bool, default=False)
	parser.add_argument('--testing', type=bool, default=False)
	parser.add_argument('--results_dir', type=str, default='./results')
	parser.add_argument('--dataset_dir', type=str, default='./Baseline_Data/Content/')
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
	parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
	parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
	parser.add_argument("--ngf", type=int, default=32)
	parser.add_argument("--n_residual_layers", type=int, default=3)
	parser.add_argument("--sr", type=int, default=22050)
    
	parser.add_argument("--n_mel_channels", type=int, default=80)

	parser.add_argument("--ndf", type=int, default=16)
	parser.add_argument("--num_D", type=int, default=3)
	parser.add_argument("--n_layers_D", type=int, default=4)
	parser.add_argument("--downsamp_factor", type=int, default=4)
	parser.add_argument('--n_bins', type=int, default=84)

	args = parser.parse_args()
	return args


def main():
	args = get_args()

	create_link(args.dataset_dir)

	str_ids = args.gpu_ids.split(',')
	args.gpu_ids = []
	for str_id in str_ids:
		id = int(str_id)
	if id >= 0:
		args.gpu_ids.append(id)
	print(not args.no_dropout)
	if args.training:
		print("Training")
		model = md.cycleRandGAN(args)
		model.train(args)
	if args.testing:
		print("Testing")
		tst.test(args)

if __name__ == '__main__':
	main()

