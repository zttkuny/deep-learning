import torch.nn as nn
import torch
from d2l import torch as d2l


class B_RNN(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens):
        super().__init__()
        self.hidden_size=num_hiddens
        self.embed_size=embed_size
        self.vocab_size=vocab_size
        self.Embedding=nn.Embedding(vocab_size,embed_size)

        # 正向参数
        self.W_hx=nn.Parameter(torch.zeros(size=(embed_size,num_hiddens),dtype=torch.float32))
        self.W_hh=nn.Parameter(torch.zeros(size=(num_hiddens,num_hiddens),dtype=torch.float32))
        self.b_h=nn.Parameter(torch.zeros(size=(num_hiddens,),dtype=torch.float32))

        # 反向参数
        self.W_hx_ = nn.Parameter(torch.zeros(size=(embed_size, num_hiddens), dtype=torch.float32))
        self.W_hh_ = nn.Parameter(torch.zeros(size=(num_hiddens, num_hiddens), dtype=torch.float32))
        self.b_h_ = nn.Parameter(torch.zeros(size=(num_hiddens,), dtype=torch.float32))

    def forward(self,X):

        X=self.Embedding(X) # 将词索引转化为嵌入向量

        B,T,D=X.shape
        H = torch.zeros(size=(B, self.hidden_size))

        X=X.permute([1,0,2]) # 把时间步维度移动到前面


        outputs=[]

        for i in range(len(X)):
            H=X[i]@self.W_hx+H@self.W_hh+self.b_h
            outputs.append(H)

        outputs=torch.stack(outputs,dim=0)


        # 反向传播
        outputs_=[]
        H = torch.zeros(size=(B, self.hidden_size))
        for i in range(len(X)-1,-1,-1):
            H = X[i] @ self.W_hx_ + H @ self.W_hh_ + self.b_h_
            outputs_.append(H)

        outputs_=torch.stack(outputs_,dim=0)

        return torch.cat([outputs,outputs_],dim=2).permute([1,0,2])



net=B_RNN(vocab_size=5000,embed_size=13,num_hiddens=512)
nn.init.zeros_(net.Embedding.weight)

X=torch.arange(10).reshape((2,5))

output=net(X)

print(output.shape)