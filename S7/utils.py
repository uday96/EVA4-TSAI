import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

def has_cuda():
	return torch.cuda.is_available()

def which_device():
	return torch.device("cuda" if has_cuda() else "cpu")

def init_seed(args):
	torch.manual_seed(args.seed)

	if has_cuda():
	    print("CUDA Available")
	    torch.cuda.manual_seed(args.seed)

def show_model_summary(model, device, input_size):
    print(summary(model, input_size=input_size))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))