import torch
import torch.nn as nn

class PostionalEncoding(nn.Module):
    def __init__(self,num_hiddens,max_len=1000):
        super().__init__()
        self.P=torch.zeros(size=(1,max_len,num_hiddens))

        X=torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(1000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)
    def forward(self,X):
        len=X.shape[1]
        return X+self.P[:,:len,:].to(X.device)

PE=PostionalEncoding(64)
X=torch.zeros(size=(5,10,64),dtype=torch.float32)
out=PE(X)
print(out)