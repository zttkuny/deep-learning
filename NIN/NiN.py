import torch.nn as nn
import  torch

def nin_block(in_channels,out_channels,kernel_size,padding=0,strides=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,stride=strides),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU()
    )