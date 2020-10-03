import time
b = time.time()
import torch.nn.functional as F
from math import log2, exp
from torchsummary import summary
import torch.nn as nn
import torch
a = time.time()
print('Imports complete in %.3f seconds' % (a-b))


class Generator(nn.Module):
    def __init__(self, latent_dims=128, max_resolution=64, max_channels=256):
        super(Generator, self).__init__()
        self.n_z = latent_dims
        assert latent_dims > 16
        num_blocks = int(log2(max_resolution))-2
        layers_ = []
        layers_.append(nn.ConvTranspose2d(self.n_z//16, max_channels, 3, 2, 1))

        # The upsampling pyramid
        for i in range(num_blocks):
            in_channels_ = max(16, max_channels//((2**i)))
            out_channels_ = max(16, in_channels_//2)
            layers_.append(self.make_gen_block(in_channels_,out_channels_,4,2,1))

        self.layers = nn.Sequential(*layers_)

    def make_gen_block(self,in_channels,out_channels,kernel_size_=3,stride_=1,padding_=1):
        return nn.Sequential([
            nn.ConvTranspose2d(in_channels,out_channels,
            kernel_size_,stride_,padding=padding_,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        ])
        
    def forward(self, latent):
        latent = latent.view(-1,self.n_z//16,4,4)
        return self.layers(latent)

class Discriminator(nn.Module):
    pass


gen = Generator()
a = torch.empty(128).normal_(0,0.35)
