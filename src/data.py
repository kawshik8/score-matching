import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from args import *
import random 

transform = torchvision.transforms.ToTensor()
random.seed(42)
np.random.seed(42)

class cifar10(Dataset):
    def __init__(self, args, split):
        
        assert split in {'train','val','test'}

        if split == 'train':
            train_data = datasets.CIFAR10(root=args.data_dir, train=True,  download=True)
            self.images = train_data

        else:
            cifar10_test  = datasets.CIFAR10(root=args.data_dir, train=False, download=True)
            full_size = len(cifar10_test)
            val_size = int(full_size * 0.5)
            test_size = int(full_size * 0.5) + val_size
            full_idx = np.random.permutation(full_size).tolist()
            splits = {"val": full_idx[:val_size], "test": full_idx[val_size:]}

            data = [cifar10_test[idx] for idx in splits[split]]

            self.images = data

        self.transforms = transforms.Compose([transforms.ToTensor()])        

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):

        image = self.images[index][0]
        label = self.images[index][1]
        
        image = self.transforms(image)

        return image
           
if __name__ == '__main__':
    args = process_args()
    images = cifar10(args,'train')
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for i,batch in enumerate(loader):
        print(i,batch.shape)
        print(torch.min(batch),torch.max(batch),torch.mean(batch))
  

        
