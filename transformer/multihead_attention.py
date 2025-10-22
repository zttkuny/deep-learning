import torch.nn as nn
from d2l import torch as d2l
import torch
import math



class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.Dropout=nn.Dropout(p=dropout)

    def forward(self,queries,keys,values,valid_lens):
        d=queries.shape[-1]

        # 注意力分数，(batch_size,num_queries,num_keys)
        attention_score=torch.bmm(queries,keys.permute([0,2,1]))/math.sqrt(d)
        # 注意力权重
        self.attention_weights=d2l.masked_softmax(attention_score,valid_lens)

        return torch.bmm(self.Dropout(self.attention_weights),values)


class MultiHeadAttention(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hiddens,num_heads,dropout):
        self.num_heads=num_heads
        super().__init__()
        self.attention=DotProductAttention(dropout)

        self.W_q=nn.Linear(query_size,num_hiddens)
        self.W_k=nn.Linear(key_size,num_hiddens)
        self.W_v=nn.Linear(value_size,num_hiddens)
        self.W_o=nn.Linear(num_hiddens,num_hiddens)

    def forward(self,queries,keys,values,valid_lens):

        # queries、keys、values分别变成(batch_size*num_heads,num_queries,num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
            valid_lens, repeats=self.num_heads, dim=0)

        # (bacth_size*num_heads,num_queries,num_hiddens/num_heads)
        out=self.attention(queries,keys,values,valid_lens)

        out=transpose_output(out,self.num_heads)

        return self.W_o(out)


# @save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""

    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

# @save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
