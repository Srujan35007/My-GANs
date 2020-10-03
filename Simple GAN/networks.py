import time 
b = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from math import log2, exp
a = time.time()
print('Imports complete in %.3f seconds'%(a-b))

class Generator(nn.Module):
	def __init__(self,latent_dims=128,max_resolution=64,max_channels=256):
		super(Generator, self).__init__()
		num_blocks = int(log2(max_resolution))
		layers = []

		self.layers = nn.Sequential(*layers)

	def make_gen_block(self,in_channels,out_channels,kernel_size_=3,stride_=1,padding_=1):
		return nn.Sequential([
			nn.ConvTranspose2d(in_channels,out_channels,
			kernel_size_,stride_,padding=padding_,bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(negative_slope=0.2)
		])
		
	def forward(self, x):
		return self.layers(x)

class Discriminator(nn.Module):
	pass
