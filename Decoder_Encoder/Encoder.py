import torch
import torch.nn as nn
from d2l import torch as d2l

class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self,vocab_size,emb_size,num_hiddens,num_layers,drop_out=0,**kwargs):
             super().__init__(**kwargs)
             self.Embedding=nn.Embedding(vocab_size,emb_size) # 嵌入层
             self.GRU=nn.GRU(emb_size,num_hiddens,num_layers=num_layers,dropout=drop_out)
             self.Out=nn.Linear(num_hiddens,vocab_size)
    def forward(self, X, *args):
           # 先输入嵌入层，转化为嵌入向量(batch_size,num_steps,emb_size)
           X=self.Embedding(X)
           X=X.permute(1,0,2) # 交换批次维度和时间步数维度

           if len(args)==0:
               output,Hn=self.GRU(X)
           else:
               state=args[0]
               output, Hn = self.GRU(X,state)

           return output,Hn


class Seq2SeqDecoder(d2l.Decoder):
     """用于序列到序列学习的循环神经网络解码器"""
     def __init__(self,vocab_size,embedding_size,num_hiddens,num_layers,drop_out=0,**kwargs):
         super().__init__(**kwargs)
         self.Embedding=nn.Embedding(vocab_size,embedding_size)
         self.rnn=nn.GRU(embedding_size+num_hiddens,num_hiddens,num_layers,dropout=drop_out)
         self.out=nn.Linear(num_hiddens,vocab_size)
     def forward(self, X, state):
         X=self.Embedding(X)
         X=X.permute(1,0,2)

         context=state[-1].repeat(X.shape[0],1,1)

         input=torch.cat((X,context),dim=2)

         output,state=self.rnn(input,state)
         output=self.out(output)

         return output.permute(1,0,2),state

class EncoderDecoder(nn.Module):
    def __init__(self,src_vocab_size,src_embed_size,tgt_vocab_size,tgt_embed_size,num_hiddens,num_layers,drop_out=0):
        super().__init__()
        self.Encoder=Seq2SeqEncoder(src_vocab_size,src_embed_size,num_hiddens,drop_out=drop_out,num_layers=num_layers)
        self.Decoder=Seq2SeqDecoder(tgt_vocab_size,tgt_embed_size,num_hiddens,num_layers,drop_out=drop_out)

    def forward(self,X,Y,*args):
        if len(args)==0:
            _,Hn=self.Encoder(X)

        else:
            _,Hn=self.Encoder(X,args[0])
        output, Hn = self.Decoder(Y, Hn)
        return output,Hn

def sequence_mask(X,valid_len):
    max_len=X.shape[1] # 序列长度

    mask=torch.tensor([i for i in range(max_len)]).unsqueeze(0).to(X.device)
    valid=valid_len.unsqueeze(-1)

    mask=mask<valid

    X[~mask]=0
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
     """带遮蔽的softmax交叉熵损失函数"""
     # pred的形状：(batch_size,num_steps,vocab_size)
     # label的形状：(batch_size,num_steps)
     # valid_len的形状：(batch_size,)
     def forward(self, pred, label, valid_len):
         weights = torch.ones_like(label)
         weights = sequence_mask(weights, valid_len)
         self.reduction='none'
         unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
         pred.permute(0, 2, 1), label)
         weighted_loss = (unweighted_loss * weights).mean(dim=1)
         return weighted_loss


def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

