# Leveraging-Spatial-Invariance-to-Boost-Adversarial-Transferability
This repository is the official Pytorch code implementation for our paper Leveraging Spatial Invariance to Boost Adversarial Transferability.

# Requirements
-python 3.9  
-torch  
-pretrainedmodels  
-numpy  
-pandas  

# Implementation

## Models
Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model). Then put these models into `./models/`.

## Generate adversarial examples
Running `attack.py` to generate adversarial examples.

## Evaluations
Running `verify.py` to evaluate the attack success rate.

