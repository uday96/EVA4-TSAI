import torch
import torchvision
import torchvision.transforms as transforms

import pprint

from utils import has_cuda

class DataEngine(object):

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
			'frog', 'horse', 'ship', 'truck')

	def __init__(self, args):
		super(DataEngine, self).__init__()
		self.batch_size_cuda = args.batch_size_cuda
		self.batch_size_cpu = args.batch_size_cpu
		self.num_workers = args.num_workers
		self.load()

	def _transforms(self):
		# Mean and standard deviation of train dataset
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2023, 0.1994, 0.2010)
		# mean = (0.5, 0.5, 0.5)
		# std = (0.5, 0.5, 0.5)

		# Data Transformations
		train_transform =  transforms.Compose(
		    [transforms.RandomCrop(32, padding=4),
		     transforms.RandomHorizontalFlip(),
		     transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		     transforms.RandomErasing(0.25)])
		
		test_transform =  transforms.Compose(
		    [transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		return train_transform, test_transform

	def _dataset(self):
		# Get data transforms
		train_transform, test_transform = self._transforms()

		# Dataset and Creating Train/Test Split
		train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=train_transform)
		test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=test_transform)
		return train_set, test_set

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

