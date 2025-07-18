# Leveraging Spatial Invariance to Boost Adversarial Transferability
This repository is the official Pytorch code implementation for our paper __Leveraging Spatial Invariance to Boost Adversarial Transferability__.

# Requirements
-python 3.6  
-torch 1.8.1  
-pretrainedmodels 0.7.4  
-numpy 1.19.5  
-Pillow 8.4.0  
-pandas 1.2  

# Implementation

## Models
Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model). Then put these models into `./models/`.

## Datase
Please refer to [SSA](https://github.com/yuyang-long/SSA). Then put dataset into `./dataset/`.

## Generate adversarial examples
Running `attack.py` to generate adversarial examples.

## Evaluations
Running `verify.py` to evaluate the attack success rate.

