import torch
import torch.nn as nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Hidden=nn.Linear(25,6)
        self.OutPut=nn.Linear(6,10)
    def forward(self,input):
        return self.OutPut(F.relu(self.Hidden(input)))

def init(model):
    nn.init.normal_(model.OutPut.weight)
    nn.init.zeros_(model.OutPut.bias)
    nn.init.constant_(model.Hidden.weight,0)
    nn.init.zeros_(model.Hidden.bias)

def main():
    M1,M2=torch.rand(size=(2,3,3)),torch.rand(size=(2,3,3))
    print(list(zip(M1,M2)))


if __name__=='__main__':
    main()