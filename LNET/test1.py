import torch.nn as nn
import torch

class my_NN(nn.Module):
    def __init__(self):
        super(my_NN,self).__init__()
        self.hidden1=nn.Linear(1,10)
        self.Sig=nn.Sigmoid()
        self.output=nn.Linear(10,3)
    def forward(self,X):
        return self.output(self.Sig(self.hidden1))

def main():
    NET=nn.Sequential(nn.Flatten(),nn.Linear(1,10),nn.ReLU(),nn.Linear(10,5))
    n=my_NN()
    for name,param in NET.parameters():
        print(name,param)
if __name__=='__main__':
    main()