import math

import torch

from transformer.multihead_attention import MultiHeadAttention
from transformer.PositionalWiseFFN import *
from d2l import torch as d2l

class EncoderBlock(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_heads,num_hiddens,
                 norm_size,num_FFN_inputs,num_FFN_hiddens,num_FFN_output,dropout):
        super().__init__()

        # 多头自注意力子层
        self.attention=MultiHeadAttention(query_size,key_size,value_size,num_hiddens,num_heads,dropout)
        # 层归一化和残差连接层
        self.add_norm=AddNorm(norm_size,dropout)
        # 前馈网络层
        self.FFN=PositionalWiseFFN(num_FFN_inputs,num_FFN_hiddens,num_FFN_output)

    def forward(self,X,valid_lens=None):
        Y=self.add_norm(X,self.attention(X,X,X,valid_lens))
        return self.add_norm(Y,self.FFN(Y))

class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,query_size,key_size,value_size,
                 num_heads,num_hiddens,
                 norm_size,num_FFN_inputs,num_FFN_hiddens,num_FFN_output,dropout,num_layers):
        super().__init__()

        self.num_hiddens=num_hiddens
        # 嵌入层
        self.embedding=nn.Embedding(vocab_size,num_hiddens)

        # 编码器块
        encoderblock=EncoderBlock(query_size,key_size,value_size,
                 num_heads,num_hiddens,
                 norm_size,num_FFN_inputs,num_FFN_hiddens,num_FFN_output,dropout)

        self.num_hiddens = num_hiddens
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)

        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_heads,num_hiddens,
                                              norm_size, num_FFN_inputs,num_FFN_hiddens,num_FFN_output
                                              , dropout)
                                 )

    def forward(self,X,valid_lens):
        X=self.embedding(X)*math.sqrt(self.num_hiddens)
        X=self.pos_encoding(X)

        self.attention_weights=[None]*len(self.blks)

        for i,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X


