import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_cub_transform

import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, fold_nr, transform=None, target_transform=None, train=False, loader=pil_loader):
    
        path_csv = os.path.join(root, 'label.csv')
        path_images = root + '/train_images/'
    
        df = pd.read_csv(path_csv)
        df['path'] = path_images + df['image_id']
        
        if train:
            self.df = df[df['kfold']!=fold_nr]
        else:
            self.df = df[df['kfold']==fold_nr]
        
        self.df = self.df.reset_index(drop=True)

        if len(self.df) < 1:
            raise(RuntimeError("no csv file"))
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        print('num of data:{}'.format(len(self.df)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.df.iloc[index]
        file_path = item['path']
        target = item['label']
        img = self.loader(file_path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.df)


def get_dataset(conf):

    datadir = 'data/cassava'

    if conf and 'datadir' in conf:
        datadir = conf.datadir

    conf['num_class'] = 5

    transform_train,transform_test = get_cub_transform(conf)

    ds_train = ImageLoader(datadir, conf['fold_nr'], train=True, transform=transform_train)
    ds_test  = ImageLoader(datadir, conf['fold_nr'], train=False, transform=transform_test)


    return ds_train,ds_test
