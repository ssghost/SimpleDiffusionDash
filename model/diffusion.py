import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import numpy as np
import math

class Diffusion:
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.img_size = 64
        self.batch = 128
        self.device = "cpu"

    def load_dataset(self):
        data_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  
            transforms.Lambda(lambda t: (t * 2) - 1) ]
        data_transform = transforms.Compose(data_transforms)

        train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

        test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
        self.dataset = torch.utils.data.ConcatDataset([train, test])
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch, shuffle=True, drop_last=True)
    
    def forward_sample(self):
        betas = torch.linspace(0.0001, 0.02, 300)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        sqrt_recip_alphas = torch.sqrt(1. / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        
