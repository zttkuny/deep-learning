from d2l import torch as d2l
import torch
from torch import nn
import  torch.nn.functional as F

# 加载《时间机器》数据集
print(F.one_hot(torch.tensor([0,2,3]),10))
