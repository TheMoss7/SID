import os
from PIL import Image
from torch.utils import data
import pandas as pd
from torchvision import transforms as T
import torch
import numpy as np


class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        # img_obj = self.csv.loc[index+550]
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        # ind_data = Image.open(img_path)
        # ind_data2 = list(ind_data.getdata())
        pil_img = Image.open(img_path).convert('RGB')
        # ind_2 = list(pil_img.getdata())
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel,

    def __len__(self):
        return len(self.csv)


class DoubleImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        # ind_data = Image.open(img_path)
        # ind_data2 = list(ind_data.getdata())
        pil_img = Image.open(img_path).convert('RGB')
        # ind_2 = list(pil_img.getdata())
        if self.transforms:
            data = self.transforms(pil_img)

        img_obj2 = self.csv.loc[(index + 1)%1000]
        ImageID2 = img_obj2['ImageId'] + '.png'
        Truelabel2 = img_obj2['TrueLabel'] - 1
        TargetClass2 = img_obj2['TargetClass'] - 1
        img_path2 = os.path.join(self.dir, ImageID2)
        # ind_data = Image.open(img_path)
        # ind_data2 = list(ind_data.getdata())
        pil_img2 = Image.open(img_path2).convert('RGB')
        # ind_2 = list(pil_img.getdata())
        if self.transforms:
            data2 = self.transforms(pil_img2)
        return data, ImageID, Truelabel,data2, ImageID2, Truelabel2,

    def __len__(self):
        return len(self.csv)


class DoublePaletteImageNet(data.Dataset):
    def __init__(self, dir, cmap_dir, csv_path, transforms = None):
        self.dir = dir
        self.cmap_dir = cmap_dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        # ind_data = Image.open(img_path)
        # ind_data2 = list(ind_data.getdata())
        pil_img = Image.open(img_path).convert('RGB')
        # ind_2 = list(pil_img.getdata())
        if self.transforms:
            data = self.transforms(pil_img)

        img_obj2 = self.csv.loc[(index + 1)%1000]
        ImageID2 = img_obj2['ImageId'] + '.png'
        Truelabel2 = img_obj2['TrueLabel'] - 1
        TargetClass2 = img_obj2['TargetClass'] - 1
        img_path2 = os.path.join(self.dir, ImageID2)
        # ind_data = Image.open(img_path)
        # ind_data2 = list(ind_data.getdata())
        pil_img2 = Image.open(img_path2).convert('RGB')
        # ind_2 = list(pil_img.getdata())
        if self.transforms:
            data2 = self.transforms(pil_img2)

        cmap_img_path = os.path.join(self.cmap_dir, ImageID)
        ind_data = Image.open(cmap_img_path)
        palette = ind_data.getpalette()
        # Determine the total number of colours
        # num_colours = int(len(palette) / 3)
        # Determine maximum value of the image data type
        cmap = np.array(palette).reshape(-1, 3)
        cmap = torch.tensor(cmap)


        return data, ImageID, Truelabel,data2, ImageID2, Truelabel2, cmap

    def __len__(self):
        return len(self.csv)


class MyDataload(data.Dataset):
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
        self.totensor = T.ToTensor()

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        ind_data = Image.open(img_path)
        indexed = np.array(ind_data)
        palette = ind_data.getpalette()
        # Determine the total number of colours
        # num_colours = int(len(palette) / 3)
        # Determine maximum value of the image data type
        map = np.array(palette).reshape(-1, 3)
        map = torch.tensor(map)
        indexed = torch.tensor(indexed)

        pil_img = ind_data.convert('RGB')
        data = self.transforms(pil_img)

        return (data[0], data[1], data[2]), data, ImageID, Truelabel, indexed, map

    def __len__(self):
        return len(self.csv)







