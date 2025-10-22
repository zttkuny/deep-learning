import torch

from multihead_attention import *
from PositionalWiseFFN import *

class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self,query_size,key_size,value_size,num_hiddens,num_heads,
                 ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs,norm_shape,
                 dropout,i):
        super().__init__()
        self.i=i

        # 带掩码的自注意力
        self.attention1=MultiHeadAttention(query_size=query_size,key_size=key_size,value_size=value_size,
                                           num_hiddens=num_hiddens,num_heads=num_heads,dropout=dropout)

        self.addnorm1=AddNorm(normalized_shape=norm_shape,dropout=dropout)

        # 非自注意力，键值来自编码器
        self.attention2 = MultiHeadAttention(query_size=query_size, key_size=key_size, value_size=value_size,
                                             num_hiddens=num_hiddens, num_heads=num_heads, dropout=dropout)

        self.addnorm2=AddNorm(normalized_shape=norm_shape,dropout=dropout)

        # 基于位置的前馈网络层
        self.ffn=PositionalWiseFFN(ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs)
        self.addnorm3=AddNorm(normalized_shape=norm_shape,dropout=dropout)

    def forward(self,X,state):
        enc_outputs,enc_valid_lens=state[0],state[1]

        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示

        if state[2][self.i] is None:
            key_values=X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)

        state[2][self.i] = key_values # 记录当前时间步的输出表示

        # 生成掩码，确保序列中每个词元只能看到它之前的词元
        if self.training:
            batch_size,num_steps,_=X.shape
            dec_valid_lens=torch.arange(1,num_steps+1,1,device=X.device).unsqueeze(0).repeat(batch_size,1)

        # 自注意力
        X2=self.attention1(X,X,X,dec_valid_lens)
        Y=self.addnorm1(X,X2)

        # 跨编码器-解码器的注意力（不是自注意力）
        X3=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Y2=self.addnorm2(Y,X3)

        # 前馈网络
        Y3=self.ffn(Y2)
        Z=self.addnorm3(Y2,Y3)
        return Z,state

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, key_size, query_size, value_size,
             num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens,
             num_heads, num_layers, dropout, **kwargs):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers

        self.blks=nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(query_size,key_size,value_size,num_hiddens,num_heads,
                 ffn_num_inputs,ffn_num_hiddens,ffn_num_inputs,norm_shape,
                 dropout,i))

        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens):
        return (enc_outputs,enc_valid_lens,[None]*self.num_layers)

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights