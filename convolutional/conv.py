import torch
import  torch.nn as nn


def Conv5x5(X,kernel):
    kh,kw=kernel.shape
    outh,outw=X.shape[0]-kh+1,X.shape[1]-kw+1
    OutM=torch.zeros((outh,outw))

    for i in range(outh):
        for j in range(outw):
            OutM[i][j]=(X[i:i+kh,j:j+kw]*kernel).sum()
    return  OutM

class Conv2d(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(torch.randn((kernel_size,kernel_size)))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,X):
        return Conv5x5(X,self.weight)+self.bias

def main():
    net=Conv2d(5)
    X=torch.randn((5,5),dtype=torch.float64)
    print(net(X))

if __name__=='__main__':
    main()