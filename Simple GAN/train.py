import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import random
from networks import Generator, Discriminator
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

# Hyper-params
BATCH_SIZE = 10
n_epochs = 100

# Fetch Training data
train_data = []
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

epoch_count = 0
while epoch_count < n_epochs:
    for data in tqdm(train_loader):    
        real_label = normalize(torch.rand(BATCH_SIZE),(0,1),(0.9,1)).float()
        fake_label = normalize(torch.rand(BATCH_SIZE),(0,1),(0,0.1)).float()
        temp = torch.empty(BATCH_SIZE,128).normal_(0,0.5)
        noise = normalize(temp,(torch.min(temp),torch.max(temp)),(-1,1)).float()

        fake_image_batch = netG(noise).detach()
        real_image_batch = normalize(data, (0,1),(-1,1))
        # Optimize Discriminator
        D_fake_batch = D(fake_image_batch)
        D_real_batch = D(real_image_batch)
        D_real_loss = loss_fn(D_real_batch, real_label)
        D_fake_loss = loss_fn(D_fake_batch, fake_label)
        D_loss = (D_real_loss + D_fake_loss)
        D_loss.backward()
        optimD.step()
        #TODO: Optimize Generator

    epoch_count += 1
