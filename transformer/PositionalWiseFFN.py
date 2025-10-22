import torch.nn as nn
import torch
# 基于位置的前馈网络
class PositionalWiseFFN(nn.Module):
    def __init__(self,num_FFN_inputs,num_FFN_hiddens,num_FFN_outputs):
        super().__init__()
        self.dense1=nn.Linear(num_FFN_inputs,num_FFN_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(num_FFN_hiddens,num_FFN_outputs)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))



class AddNorm(nn.Module):
     """残差连接后进行层规范化"""
     def __init__(self, normalized_shape, dropout, **kwargs):
         super(AddNorm, self).__init__(**kwargs)
         self.dropout = nn.Dropout(dropout)
         self.ln = nn.LayerNorm(normalized_shape)

     def forward(self, X, Y):
         return self.ln(self.dropout(Y) + X)


