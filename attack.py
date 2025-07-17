"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from methods import multi_scale, local_fusion
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from torchvision.transforms import GaussianBlur
import torchvision.transforms.functional as TF
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/incv3/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")



opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)




def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)



def ours(images, gt, model, min, max, block_num=2):
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = 0
    N = opt.N
    for i in range(num_iter):
        noise = 0
        for n in range(N):
            x = V(x, requires_grad=True)
            x_emb = local_fusion(x, block_num)
            resize_ratio = 1 - n * 0.005
            x_enh= multi_scale(x_emb, resize_ratio)
            output_v3 = model(x_enh)
            loss = F.cross_entropy(output_v3, gt)
            loss.backward()
            gradient_total = x.grad.data
            noise += gradient_total
        noise = noise / N

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()






def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False)
    for images, images_ID,  gt_cpu in tqdm(data_loader):

        gt = gt_cpu.cuda()
        images = images.cuda()

        images_min = clip_by_tensor(images - (opt.max_epsilon / 255.0), 0.0, 1.0)
        images_max = clip_by_tensor(images + (opt.max_epsilon / 255.0), 0.0, 1.0)


        direction = ours(images, gt, model, images_min, images_max)
        adv_img_np = direction.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)




if __name__ == '__main__':
    main()