from RNN.corpus import *
import torch
import torch.nn as nn
from d2l import torch as d2l


_,vocab=load_corups_time_machine()

vocab_size=len(vocab)

class GRU(nn.Module):
    def __init__(self,num_inputs,num_hiddens,batch_first=False):
        super().__init__()
        self.gru=nn.GRU(num_inputs,num_hiddens,batch_first=batch_first)
        self.output=nn.Linear(num_hiddens,num_inputs)

    def forward(self,X,state=None):
        if state==None:
            all_H,Hn=self.gru(X)
        else:
            all_H, Hn = self.gru(X, state)
        return self.output(Hn)

def init_param(m):
    if isinstance(m,(nn.GRU,nn.Linear)):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

def main():
    net=GRU(vocab_size,512)
    net.apply(init_param)


    d2l.train_ch8
main()