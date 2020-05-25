import torch
import torchvision
import glob
from random import shuffle
from math import floor

from .data_transforms import albumentations_transforms, torch_transforms
from utils import has_cuda, imshow, show_images, normalize, denormalize, data_stats
from .data_set import PersonFgBgDataset

class DataEngine(object):

	def __init__(self, args):
		super(DataEngine, self).__init__()
		self.batch_size_cuda = args.batch_size_cuda
		self.batch_size_cpu = args.batch_size_cpu
		self.num_workers = args.num_workers
		self.data_path = args.data_path
		self.im_size = args.im_size
		self.stats = data_stats()
		# self.test_data_path = args.test_data_path
		self.split_data()
		self.load()

	# def _transforms(self):
	# 	# Data Transformations
	# 	# train_transforms = albumentations_transforms(p=1.0, is_train=True, stats=self.stats)
	# 	# test_transforms = albumentations_transforms(p=1.0, is_train=False, stats=self.stats)
	# 	train_transforms = torch_transforms(is_train=True, stats=self.stats)
	# 	test_transforms = torch_transforms(is_train=False, stats=self.stats)
	# 	return train_transforms, test_transforms

	def _dataset(self):
		# # Get data transforms
		# train_transforms, test_transforms = self._transforms()

		# Dataset and Creating Train/Test Split
		train_set = PersonFgBgDataset(self.data_path,
			files=self.labels_data["train"], data_stats=self.stats,
			do_img_aug=True, im_size=self.im_size)
		test_set = PersonFgBgDataset(self.data_path,
			files=self.labels_data["test"], data_stats=self.stats,
			do_img_aug=False, im_size=self.im_size)
		return train_set, test_set

	def split_data(self):
		train_labels = []
		test_labels = []
		data_root = self.data_path
		if data_root[-1] == "/":
			data_root = data_root[:-1]
		train_split = 0.8
		for bg_idx in range(100):
			for fg_idx in range(100):
				files = glob.glob(f"{data_root}/fg_bg/bg_{bg_idx:02d}/bg_{bg_idx:02d}_fg_{fg_idx:02d}*.jpg")
				files = list(map(lambda fp: fp.split("/")[-1], files))
				shuffle(files)
				split_n = floor(len(files)*train_split)
				train_labels.extend(files[:split_n])
				test_labels.extend(files[split_n:])
		
		self.labels_data = {
			"train": train_labels,
			"test": test_labels
		}

	def load(self):
		# Get Train and Test Data
		train_set, test_set = self._dataset()

		# Dataloader Arguments & Test/Train Dataloaders
		dataloader_args = dict(
			shuffle= True,
			batch_size= self.batch_size_cpu)
		if has_cuda():
			dataloader_args.update(
				batch_size= self.batch_size_cuda,
				num_workers= self.num_workers,
				pin_memory= True)

		self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
		self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

	def show_samples(self):
		# get some random training images
		samples = next(iter(self.train_loader))
		print("Dimensions:")
		print("bg         :", list(samples["bg"].shape)[1:])
		print("fg_bg      :", list(samples["fg_bg"].shape)[1:])
		print("fg_bg_mask :", list(samples["fg_bg_mask"].shape)[1:])
		print("fg_bg_depth:", list(samples["fg_bg_depth"].shape)[1:],"\n")
		num_img = 4
		images = []
		keys = ("bg","fg_bg","fg_bg_mask","fg_bg_depth")
		for i in range(num_img):
			for k in keys:
				# if k in ("bg", "fg_bg"):
				images.append(denormalize(samples[k][i],
					mean=self.stats[k]["mean"], std=self.stats[k]["std"]))
				# else:
				# 	images.append(samples[k][i])
		show_images(images, cols=num_img, figsize=(6,6), show=True, titles=keys)

