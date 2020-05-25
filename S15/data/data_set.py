import torch
from torch.utils.data import Dataset
from PIL import Image
from .data_transforms import albumentations_transforms, torch_transforms

class PersonFgBgDataset(Dataset):
  def __init__(self, data_root, files, data_stats={}, do_img_aug=False,
              im_size=64):
    self.data_root = data_root
    if self.data_root[-1] == "/":
      self.data_root = self.data_root[:-1]
    self.fg_bg_files = files
    self.do_img_aug = do_img_aug
    self.data_stats = data_stats
    self.im_size = im_size

  def __len__(self):
    return len(self.fg_bg_files)

  def __getitem__(self, idx):
    fg_bg_file = self.fg_bg_files[idx]
    bg_id = fg_bg_file[:5]
    bg_img = Image.open(f"{self.data_root}/bg/{bg_id}.jpg")
    fg_bg_img = Image.open(f"{self.data_root}/fg_bg/{bg_id}/{fg_bg_file}")
    fg_bg_mask_img = Image.open(f"{self.data_root}/fg_bg_masks/{bg_id}/{fg_bg_file}")
    fg_bg_depth_img = Image.open(f"{self.data_root}/fg_bg_depth/{bg_id}/{fg_bg_file}")
    
    return albumentations_transforms(
      data_map={
        "bg": bg_img,
        "fg_bg": fg_bg_img,
        "fg_bg_mask": fg_bg_mask_img,
        "fg_bg_depth": fg_bg_depth_img
      },
      data_stats=self.data_stats,
      do_img_aug=self.do_img_aug,
      im_size=self.im_size
    )
