import torch
import torch.nn as nn
from kornia.losses import SSIM
from kornia.filters import SpatialGradient


def inverse_huber_loss(target, output):
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))

def edge_loss(target, output):
	gt_pred = SpatialGradient()(output)
	assert(gt_pred.ndim == 5)
	assert(gt_pred.shape[2] == 2)
	dy_pred = gt_pred[:,:,0,:,:]
	dx_pred = gt_pred[:,:,1,:,:]
	gt_true = SpatialGradient()(target)
	dy_true = gt_true[:,:,0,:,:]
	dx_true = gt_true[:,:,1,:,:]
	l_edge = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
	return l_edge

def loss_mask_function(target, output, w_depth=1.0, w_ssim=1.0, w_edge=1.0):
	# Point-wise depth	
	l_depth = nn.L1Loss()(output, target)

	# Structural similarity (SSIM) index
	# 1 - ssim_index is computed internally, within SSIM()
	l_ssim = SSIM(3, reduction="mean")(output, target)

	# Edges
	l_edge = edge_loss(target, output)
	s = f"(d{l_depth.item():0.3f},s{l_ssim.item():0.3f},e{l_edge.item():0.3f})"
	loss = (w_ssim * l_ssim) + (w_depth * l_depth) + (w_edge * l_edge)
	return loss, s

def loss_depth_function(target, output, w_ssim=1.0, w_edge=1.0, w_depth=1.0):
	# Structural similarity (SSIM) index
	# 1 - ssim_index is computed internally, within SSIM()
	l_ssim = SSIM(3, reduction="mean")(output, target)

	# Edges
	l_edge = edge_loss(output, target)

	# Point-wise depth
	l_depth = nn.L1Loss()(output, target)
	# l_huber = inverse_huber_loss(target, output)
	# l_mse = nn.MSELoss()(output*10, target*10)
	# l_bce = nn.BCEWithLogitsLoss()(output, target)
	s = f"(d{l_depth.item():0.3f},s{l_ssim.item():0.3f},e{l_edge.item():0.3f})"
	loss = (w_ssim * l_ssim) + (w_depth * l_depth) + (w_edge * l_edge)
	return loss, s
