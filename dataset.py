import os
import random

import numpy as np
from skimage.util import random_noise
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def addGaussNoise(data, sigma):
    sigma2 = sigma**2 / (255 ** 2)
    noise = random_noise(data, mode='gaussian', var=sigma2, clip=True)
    return noise


class MyDataset(Dataset):
    def __init__(self, path, transform, sigma=30, ex=1):
        self.transform = transform
        self.sigma = sigma

        for _, _, files in os.walk(path):
            self.imgs = [path + file for file in files if Image.open(path + file).size >= (96,96)] * ex

        np.random.shuffle(self.imgs)

    def __getitem__(self, index):
        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg).convert('RGB')
        Img = np.array(self.transform(tempImg))/255
        nImg = addGaussNoise(Img, self.sigma)
        Img = torch.tensor(Img.transpose(2,0,1))
        nImg = torch.tensor(nImg.transpose(2,0,1))
        return Img, nImg

    def __len__(self):
        return len(self.imgs)


def get_data(batch_size, train_path, val_path, transform, sigma, ex=1, num_workers=0):
    train_dataset = MyDataset(train_path, transform, sigma, ex)
    val_dataset = MyDataset(val_path, transform, sigma, ex)
    train_iter = DataLoader(train_dataset, batch_size, drop_last=True, num_workers=num_workers)
    val_iter = DataLoader(val_dataset, batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, val_iter