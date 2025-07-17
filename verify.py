"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
# from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import numpy as np
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
import argparse
from torchvision import models

batch_size = 10

input_csv = './dataset/images.csv'

adv_dir_list = ['./outputs/incv3',]

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
opt = parser.parse_args()

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model

def verify(model_name, path):

    model = get_model(model_name, path)
    for adv_dir in adv_dir_list:

        X = ImageNet(adv_dir, input_csv, T.Compose([T.Resize(299), T.ToTensor()]))
        length = len(X)
        data_loader = DataLoader(X, batch_size=batch_size, shuffle=False)
        sum = 0
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()
            with torch.no_grad():
                sum += (model(images)[0].argmax(1) != (gt+1)).detach().sum().cpu()

        print(model_name + '  acu = {:.2%}'.format(sum / length))

def main():

    model_names = ['tf_inception_v3', 'tf_inception_v4', 'tf_resnet_v2_50', 'tf_resnet_v2_101', 'tf_resnet_v2_152', 'tf_inc_res_v2',
                   'tf_adv_inception_v3', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']

    models_path = './models/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")

if __name__ == '__main__':
    main()
