import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import warnings

def has_cuda():
	return torch.cuda.is_available()

def which_device():
	return torch.device("cuda" if has_cuda() else "cpu")

def init_seed(args):
	torch.manual_seed(args.seed)

	if has_cuda():
		print("CUDA Available")
		torch.cuda.manual_seed(args.seed)

def show_model_summary(model, input_size):
	print(summary(model, input_size=input_size))

def imshow(img, title):
	img = denormalize(img)
	npimg = img.numpy()
	fig = plt.figure(figsize=(15,7))
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(title)


def normalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	if len(mean) == 1:
		return normalize_gray(tensor, mean, std)
	single_img = False
	if tensor.ndimension() == 3:
		if tensor.shape[0]!=3:
			return normalize_gray(tensor, mean, std)
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
		raise TypeError('tensor should be 4D')

	num_ch = 3
	if tensor.shape[1] == 1:
		num_ch = 1

	mean = torch.FloatTensor(mean).view(1, num_ch, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, num_ch, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.sub(mean).div(std)
	return ret[0] if single_img else ret

def normalize_gray(tensor, mean=[0.4914], std=[0.2023]):
	if len(mean)!=1:
		raise TypeError('mean should be 1D')
	
	single_img = False
	if tensor.ndimension() == 2:
		single_img = True
		tensor = tensor[None,:,:]

	if not tensor.ndimension() == 3:
		raise TypeError('tensor should be 3D')

	mean = torch.FloatTensor(mean).view(1, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.sub(mean).div(std)
	return ret[0] if single_img else ret

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	return tensor
	if len(mean) == 1:
		return denormalize_gray(tensor, mean, std)

	single_img = False
	if tensor.ndimension() == 3:
		if tensor.shape[0]!=3:
			return denormalize_gray(tensor, mean, std)
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
		raise TypeError('tensor should be 4D')

	num_ch = 3
	if tensor.shape[1] == 1:
		num_ch = 1
	mean = torch.FloatTensor(mean).view(1, num_ch, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, num_ch, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret

def denormalize_gray(tensor, mean=[0.4914], std=[0.2023]):
	if len(mean)!=1:
		raise TypeError('mean should be 1D')
	
	single_img = False
	if tensor.ndimension() == 2:
		single_img = True
		tensor = tensor[None,:,:]

	if not tensor.ndimension() == 3:
		raise TypeError('tensor should be 3D')

	mean = torch.FloatTensor(mean).view(1, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret

def write_json(fname, data):
	with open(fname, 'w') as f:
		json.dump(data, f, indent=2)

def read_json(fname, data):
	with open(fname, 'r') as f:
		json.load(f)

def data_stats():
	stats = {
		"bg": {
			"mean": np.array([0.5039, 0.5001, 0.4849]),
			"std": np.array([0.2465, 0.2463, 0.2582])
		},
		"fg_bg": {
			"mean": np.array([0.5056, 0.4969, 0.4817]),
			"std": np.array([0.2486, 0.2490, 0.2604]),
		},
		"fg_bg_mask": {
			"mean": np.array([0.0454]),
			"std": np.array([0.2038]),
		},
		"fg_bg_depth": {
			"mean": np.array([0.4334]),
			"std": np.array([0.2715]),
		},
	}
	return stats

def show_images(images, cols=1, titles=[], figsize=(5,5), tmax=16,
	fname="viz_tmp.jpg", show=False, denorm=""):
	warnings.filterwarnings("ignore")
	stats = data_stats()
	n_images = min(len(images), tmax)
	fig = plt.figure(figsize=figsize)
	for n in range(n_images):
		image = images[n]
		if image.shape[0] == 1:
			image = image[0]
			if denorm in ("fg_bg_mask", "fg_bg_depth"):
				image = denormalize(image, mean=stats[denorm]["mean"],
									std=stats[denorm]["std"])
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			# image = np.transpose(image.numpy(), (1,0))
			plt.gray()
		else:
			if denorm=="fg_bg":
				image = denormalize(image, mean=stats[denorm]["mean"],
									std=stats[denorm]["std"])
			image = np.transpose(image.numpy(), (1, 2, 0))
		plt.axis("off")
		plt.imshow(image)
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.tight_layout()
		if n < len(titles):
			a.set_title(titles[n], fontdict={'fontsize': 'x-large',
				'fontweight': 'roman'})
	plt.subplots_adjust(wspace=0, hspace=0)
	fig.tight_layout()
	plt.savefig(fname)
	if show:
		plt.show()
	plt.close()
