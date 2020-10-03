import time 
b = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
a = time.time()
print('Imports complete in %.3f seconds'%(a-b))

class Generator(nn.Module):
  pass
  
class Discriminator(nn.Module):
  pass
