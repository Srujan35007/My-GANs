import time
b = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from math import log2, exp
a = time.time()
print('Imports complete in %.3f seconds' % (a-b))


class Generator(nn.Module):
    def __init__(self,latent_dims=128,n_channels=3,max_resolution=64,max_channels=256,min_channels=16):
        super(Generator, self).__init__()
        self.n_z = latent_dims
        assert latent_dims > 16
        num_blocks = int(log2(max_resolution))-2
        layers_ = []
        layers_.append(nn.ConvTranspose2d(self.n_z//16,max_channels,3,1,1))
        output_channels = -1
        # The upsampling pyramid
        for i in range(num_blocks):
            in_channels_ = max(min_channels, max_channels//((2**i)))
            out_channels_ = max(min_channels, in_channels_//2)
            layers_.append(self.make_gen_block(in_channels_,out_channels_,4,2,1))
            output_channels = out_channels_
        
        layers_.append(nn.Conv2d(output_channels,n_channels,3,1,1))
        layers_.append(nn.Tanh())
        self.layers = nn.Sequential(*layers_)

    def make_gen_block(self,in_channels,out_channels,kernel_size_=3,stride_=1,padding_=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,
            kernel_size_,stride_,padding=padding_,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        
    def forward(self, latent):
        latent = latent.view(-1,self.n_z//16,4,4)
        return self.layers(latent)

class Discriminator(nn.Module):
    pass


gen = Generator(max_resolution=512,max_channels=512,min_channels=8)
a = torch.empty(128).normal_(0,0.35).view(-1,128)
print(summary(gen, (1,128)))