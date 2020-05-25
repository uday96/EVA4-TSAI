import albumentations as A
import albumentations.pytorch as AP
import torch
import PIL
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

def albumentations_transforms(data_map, data_stats, do_img_aug=False,
							im_size=64):
	transforms_list_map = {}
	transforms_map = {}
	for dname in data_map.keys():
		transforms_list_map[dname] = []

	# Resize
	size = (im_size, im_size)
	for k, v in transforms_list_map.items():
		v.append(
			A.Resize(height=size[0], width=size[0],
				interpolation=cv2.INTER_LANCZOS4, always_apply=True)
		)
	# Use data aug only for train data
	if do_img_aug:
		# RGB shift and Hue Saturation value
		for k, v in transforms_list_map.items():
			if k not in ("bg", "fg_bg"):
				continue
			v.append(
				A.OneOf([
					A.RGBShift(r_shift_limit=20, g_shift_limit=20,
						b_shift_limit=20, p=0.5),
					A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
						val_shift_limit=20, p=0.5)
				], p=0.5)
			)
		# GaussNoise
		for k, v in transforms_list_map.items():
			if k not in ("bg", "fg_bg"):
				continue
			v.append(
				A.GaussNoise(p=0.5)
			)
		# Random horizontal flipping
		if random.random() > 0.5:
			for k, v in transforms_list_map.items():
				v.append(
					A.HorizontalFlip(always_apply=True)
				)
		# Random vertical flipping
		if random.random() > 0.7:
			for k, v in transforms_list_map.items():
				v.append(
					A.VerticalFlip(always_apply=True)
				)
		# Random rotate
		if random.random() > 0.3:
			angle = random.uniform(-15, 15)
			for k, v in transforms_list_map.items():
				v.append(
					A.Rotate(limit=(angle, angle), interpolation=cv2.INTER_LANCZOS4,
						always_apply=True)
				)
		# Coarse dropout
		for k, v in transforms_list_map.items():
			if k not in ("bg","fg_bg"):
				continue
			v.append(
				A.CoarseDropout(max_holes=2, fill_value=0, max_height=size[0]//3,
					max_width=size[1]//4, p=0.5)
			)
			# fill_value=data_stats[k]["mean"]*255.0

	# for k, v in transforms_list_map.items():
	# 	v.append(
	# 		A.Normalize(
	# 			mean=data_stats[k]["mean"],
	# 			std=data_stats[k]["std"],
	# 			max_pixel_value=255.0,
	# 			always_apply=True
	# 		)
	# 	)
	for k, v in transforms_list_map.items():
		v.append(
			AP.ToTensor()
		)

	for k, v in transforms_list_map.items(): 
		np_img = np.array(data_map[k])
		if np_img.ndim == 2:
			np_img = np_img[:,:,np.newaxis]
		transforms_map[k] = A.Compose(v, p=1.0)(image=np_img)["image"]

	# transforms_map["fg_bg_mask"] = torch.gt(transforms_map["fg_bg_mask"], 0.8).float()
	return transforms_map

def torch_transforms(data_map, data_stats, do_img_aug=False, im_size=64):

	# Resize
	size = (im_size, im_size)
	resize = transforms.Resize(size)
	for k, v in data_map.items():
		data_map[k] = resize(v)

	if do_img_aug:
		# Color jitter
		for k, v in data_map.items():
			if k in ("fg_bg_mask", "fg_bg_depth"):
				continue
			data_map[k] = transforms.ColorJitter(brightness=0.1, contrast=0.1,
				saturation=0.1, hue=0.1)(v)

		# Random horizontal flipping
		if random.random() > 0.5:
			for k, v in data_map.items():
				data_map[k] = TF.hflip(v)

		# Random vertical flipping
		if random.random() > 0.7:
			for k, v in data_map.items():
				data_map[k] = TF.vflip(v)

	for k, v in data_map.items():
		data_map[k] = transforms.ToTensor()(v)

	# for k, v in data_map.items():
	# 	if k in ("fg_bg_mask", "fg_bg_depth"):
	# 		continue
	# 	data_map[k] = transforms.Normalize(data_stats[k]["mean"],
	# 		data_stats[k]["std"])(v)

	if do_img_aug:
		# Random erasing
		for k in ("bg",):
			data_map[k] = transforms.RandomErasing(0.25)(data_map[k])

	# data_map["fg_bg_mask"] = torch.gt(data_map["fg_bg_mask"], 0.8).float()

	return data_map
