import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur, ColorJitter
import torchvision.transforms.functional as TF
from torch import nn
import random
from dct import *

"""Translation-Invariant https://arxiv.org/abs/1904.02884   TIM对梯度进行变换"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978   DIM对图片进行变换"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret




def get_length(length, num_block):
    length = int(length)
    rand = np.random.uniform(size=num_block, low=0.1, high=0.9)
    rand_norm = np.round(rand*length/rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

def random_flip(x):
    ret = x.clone()
    if torch.rand(1) < 0.5:
        ret = torch.flip(ret, dims=(3,))
    return ret

def frequency_fusion(patch, x):
    org_x = x.clone()
    _, _, patch_w, patch_h = patch.shape
    rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)
    rescale_flip_x = random_flip(rescale_x)
    dctx = dct_2d(rescale_flip_x)  # torch.fft.fft2(x, dim=(-2, -1))
    dctp = dct_2d(patch)
    _, _, w, h = dctx.shape
    low_ratio = 0.4
    low_w = int(w * low_ratio)
    low_h = int(h * low_ratio)
    # patch_low = dctp[:, :, 0:low_w, 0:low_h]
    dctx[:, :, 0:low_w, 0:low_h] = dctp[:, :, 0:low_w, 0:low_h]
    idctx = idct_2d(dctx)
    return idctx

def linear_fusion(patch, x, weight=0.5):
    org_x = x.clone()
    _, _, patch_w, patch_h = patch.shape
    rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)

    rescale_flip_x = random_flip(rescale_x)
    ret = rescale_flip_x * weight + patch * (1 - weight)
    return ret

def block_fusion(patch, x):
    if torch.rand(1) < 0.5:
        return patch
    else:
        if torch.rand(1) < 0.5:
            return frequency_fusion(patch, x)
        else:
            return linear_fusion(patch, x)

def local_fusion(x, num_block=2):
    batch_size, _, w, h = x.shape
    width_length, height_length = get_length(w, num_block), get_length(h, num_block)
    x_split_w = torch.split(x, width_length, dim=2)
    x_split_h_l = [torch.split(x_split_w[i], height_length, dim=3) for i in range(num_block)]

    ret_list = []
    for strip in x_split_h_l:
        temp_list = []
        for i in range(num_block):
            x_enh = block_fusion(strip[i], x)
            x_enh_flip= random_flip(x_enh)

            temp_list.append(x_enh_flip)
        temp = torch.cat(temp_list, dim=3)
        ret_list.append(temp)
    x_h_perm = torch.cat(ret_list, dim=2)

    return x_h_perm


def multi_scale(x, resize_ratio):
    img_size = x.shape[-1]
    if resize_ratio == 1:
        ret =  x
    else:
        img_resize = int(img_size * resize_ratio)
        rescaled = F.interpolate(x, size=[img_resize, img_resize], mode='bilinear', align_corners=False)
        h_rem = img_size - img_resize
        w_rem = img_size - img_resize
        pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        ret = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = random_flip(ret)
    return ret