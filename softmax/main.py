import torch
import torchvision
from torch import  nn
import matplotlib.pyplot as plt
from d2l import torch as d2l

def sequence_mask(X,valid_len,value=0):
    maxlen=X.shape[1] # 固定序列长度

    mask=torch.arange(maxlen,dtype=torch.int).unsqueeze(dim=0)<valid_len.unsqueeze(-1)

    X[~mask]=value

    return X
X=torch.arange(1,11,dtype=torch.int).reshape((2,5))
valid_len=torch.tensor([4,2],dtype=torch.int)

print(sequence_mask(X,valid_len))

