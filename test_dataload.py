# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:59:08 2021

@author: anass
"""
from PIL import Image
import numpy as np

# img_filename = "C:/Users/anass/OneDrive/Documents/thesis/gen_perf_results/test_dataset/test_dataset/ref_0.png"


# img = Image.open(img_filename)
# pixels = img.load()

# data = np.array(img)


import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SpecklesDataset(Dataset): 
   

    def __init__(self, csv_file, root_dir, transform=None):
 
        self.Speckles_frame = pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Ref_name   = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name   = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
       
        # Ref   = np.genfromtxt(Ref_name, delimiter=',')
        # Def   = np.genfromtxt(Def_name, delimiter=',')
        # Read Ref & Def Images in ".png" format
        Ref   = Image.open(Ref_name)
        Ref   = np.array(Ref, dtype=np.float64)
        Def   = Image.open(Def_name)
        Def   = np.array(Def, dtype=np.float64)
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        Ref = Ref
        Def = Def
        Dispx = Dispx
        Dispy = Dispy

        Ref   = Ref[np.newaxis, ...]       
        Def   = Def[np.newaxis, ...]
        Dispx   = Dispx[np.newaxis, ...]
        Dispy   = Dispy[np.newaxis, ...]

        
        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalization(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        Ref, Def, Dispx, Dispy = sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']

        self.mean = 0.0
        self.std  = 255.0        
        self.mean1 = -1.0
        self.std1  = 2.0
        
        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float() ,
                'Def': torch.from_numpy((Def - self.mean) / self.std).float() ,
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1 ).float() ,
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float() }

#######
# Data loading code
transform = transforms.Compose([Normalization()])
    

train_set = SpecklesDataset(csv_file='~/Train_annotations.csv', root_dir='~/Train_Data/', transform = transform)
test_set = SpecklesDataset(csv_file='~/Test_annotations.csv', root_dir='~/Test_Data/', transform = transform)

    
print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                       len(train_set),
                                                                       len(test_set)))



