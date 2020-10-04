import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import random
from a import Generator, Discriminator
import matplotlib.pyplot as plt 
print('Imports complete')


def normalize(array, original_range, target_range):
    min_array,max_array = original_range
    target_min,target_max = target_range
    normalized = target_min+((target_max-target_min)*(array-min_array))/(max_array-min_array)
    return normalized


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
netG = Generator(max_resolution=32,latent_dims=128,max_channels=256,min_channels=16).to(device)
netD = Discriminator(max_resolution=32,max_channels=256,min_channels=16).to(device)
optimG = optim.Adam(netG.parameters(),lr=0.00015,betas=(0.5,0.999))
optimD = optim.Adam(netD.parameters(),lr=0.00015,betas=(0.5,0.999))
loss_fn = nn.BCELoss()

BATCH_SIZE = 10
epoch_count = 0
n_epochs = 100

while epoch_count < n_epochs:
    for data in tqdm(train_loader):    
        real_label = normalize(torch.rand(BATCH_SIZE),(0,1),(0.9,1)).float()
        fake_label = normalize(torch.rand(BATCH_SIZE),(0,1),(0,0.1)).float()
        temp = torch.empty(BATCH_SIZE,128).normal_(0,0.5)
        noise = normalize(temp,(torch.min(temp),torch.max(temp)),(-1,1)).float()

    epoch_count += 1