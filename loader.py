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


