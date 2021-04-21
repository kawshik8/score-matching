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

class Data(Dataset):
    def __init__(self, args, split):
        
        assert split in {'train','val','test'}

        self.get_data(args.dataset) 

        if split == 'train':
            if args.dataset != 'celeba':
                train_data = self.dataset(root=args.data_dir, train=True,  download=True)
            else:
                train_data = self.dataset(root=args.data_dir, split=split,  download=True)

            self.images = train_data

        else:
            if args.dataset != 'celeba':
                val_test  = self.dataset(root=args.data_dir, train=False, download=True)
            else:
                val_test = self.dataset(root=args.data_dir, split=split, download=True)

            full_size = len(val_test)
            test_size = val_size = int(full_size * 0.5)
            full_idx = np.random.permutation(full_size).tolist()
            splits = {"val": full_idx[:val_size], "test": full_idx[val_size:]}

            data = [val_test[idx] for idx in splits[split]]

            self.images = data

        self.transforms = transforms.Compose([transforms.ToTensor()])        

    def get_data(self, dataset_name):
        if dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10
        elif dataset_name == 'celeba':
            self.dataset = datasets.CelebA
        elif dataset_name == 'mnist':
            self.dataset = datasets.MNIST

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):

        image = self.images[index][0]
        label = self.images[index][1]
        
        image = self.transforms(image)

        return image
           
if __name__ == '__main__':
    args = process_args()
    images = Data(args,'train')
    loader = DataLoader(images, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for i,batch in enumerate(loader):
        print(i,batch.shape)
        print(torch.min(batch),torch.max(batch),torch.mean(batch))
  

        
