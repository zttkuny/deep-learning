import time

import torch.nn as nn
from d2l import torch as d2l
import torch
from transformer.Encoder import EncoderBlock
from multiprocessing import Pool

def get_tokens_and_segments(tokens_a, tokens_b=None):
     """获取输入序列的词元及其片段索引"""
     tokens = ['<cls>'] + tokens_a + ['<sep>']
     # 0和1分别标记片段A和B
     segments = [0] * (len(tokens_a) + 2)
     if tokens_b is not None:
         tokens += tokens_b + ['<sep>']
         segments += [1] * (len(tokens_b) + 1)
     return tokens, segments

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
         ffn_num_hiddens, num_heads, num_layers, dropout,
         max_len=1000, key_size=768, query_size=768, value_size=768,
         **kwargs):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        # 片段嵌入层
        self.segment_embedding=nn.Embedding(2,num_hiddens)
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size=key_size, query_size=query_size, value_size=value_size,num_hiddens= num_hiddens, norm_size=norm_shape,
            num_FFN_inputs=ffn_num_input, num_FFN_hiddens=ffn_num_hiddens, num_FFN_output=ffn_num_input,num_heads=num_heads, dropout=dropout))

    def forward(self,tokens, segments, valid_lens):
        # X的形状，（batch_size,num_steps,num_hiddens）
        X=self.token_embedding(tokens)+self.segment_embedding(segments)
        X=X+self.pos_embedding[:,:X.shape[1],:]
        for blk in self.blks:
            X=blk(X,valid_lens)
        return X

# 带掩码的语言模型任务
class MaskLM(nn.Module):
    def __init__(self,num_hiddens,vocab_size,num_inputs=768):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(num_inputs,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,vocab_size)
        )

    def forward(self,X,pred_position):
        #  X的形状是(bacth_size,seq_len,num_inputs),pred_position是(batch_size,num_pred_positon)
        num_pred_position=pred_position.shape[1]
        pred_position=pred_position.reshape(-1)
        batch_idx=torch.arange(X.shape[0])
        batch_size=X.shape[0]

        batch_idx=batch_idx.repeat_interleave(repeats=num_pred_position)
        # 取出需要预测的向量
        masked_X=X[batch_idx,pred_position]
        # out形状是(batch_size*num_pred_position,vocab_size)
        out=self.mlp(masked_X)
        return out.reshape(batch_size,num_pred_position,-1)

# 下个句子预测
class NXP(nn.Module):
    def __init__(self,num_inputs):
        super().__init__()
        self.mlp=nn.Linear(num_inputs,2)

    # X的形状：(batchsize,num_hiddens)
    def forward(self,X):
        return self.mlp(X)

class BERT(nn.Module):
   def __init__(self,vocab_size, num_hiddens, norm_shape, ffn_num_input,
                        ffn_num_hiddens, num_heads, num_layers, dropout,
                        max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
       super().__init__()

       # 编码器部分
       self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                                  ffn_num_hiddens, num_heads, num_layers, dropout,
                                  max_len=1000, key_size=768, query_size=768, value_size=768)

       self.mlm=MaskLM(num_hiddens,vocab_size,num_inputs=num_hiddens)
       self.nsp=NXP(num_hiddens)

   def forward(self,X,segments, valid_lens=None,
                pred_position=None):
            enc_output=self.encoder(X,segments,valid_lens)

            if pred_position !=None:
                pred=self.mlm(enc_output,pred_position)
            else:
                pred=None

            nxp=self.nsp(enc_output[:,0,:])
            # 返回编码器结果，两个预训练任务结果
            return enc_output,pred,nxp
