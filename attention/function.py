import torch.nn as nn
import torch.nn.functional as F
import torch
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
     return nn.functional.softmax(X, dim=-1)
    else:
         shape = X.shape
         if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
         else:
            valid_lens = valid_lens.reshape(-1)
         # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
         X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
            value=-1e6)
         return nn.functional.softmax(X.reshape(shape), dim=-1)

class AddictiveAttention(nn.Module):
    def __init__(self,key_size,query_size,hidden_size,drop_out=0.5,**args):
        super().__init__(**args)
        self.W_kh=nn.Linear(key_size,hidden_size,bias=False)
        self.W_qh=nn.Linear(query_size,hidden_size,bias=False)
        self.dense=nn.Linear(hidden_size,1,bias=False)
        self.Dropout=nn.Dropout(p=drop_out)

    def forward(self,queries,keys,values,valid_lens):
        queries=self.W_qh(queries)
        keys=self.W_kh(keys)

        queries=queries.unsqueeze(dim=-2)
        keys=keys.unsqueeze(dim=1)

        features=queries+keys # 广播加法
        features=F.tanh(features)


        # 去掉最后一个维度
        scores=self.dense(features).squeeze(-1)

        # 计算出注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.Dropout(self.attention_weights),values)



class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self,vocab_size, embed_size, num_hiddens, num_layers,
         dropout=0,**kwargs):
        super().__init__()
        self.attention=AddictiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(num_hiddens+embed_size,num_hiddens,num_layers,dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self,X,state):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        en_output,hidden_state,enc_valid_lens=state

        X=self.embedding(X).permute([1,0,2])
        outputs, self._attention_weights = [], []
        for x in X:
            query=hidden_state[-1].unsqueeze(dim=1) # 添加一个维度作为query数量

            # 返回维度为 (batch_size,1,num_hiddens)
            attention=self.attention(query,en_output,en_output,enc_valid_lens)

            # 上下文
            context=torch.cat([attention,x.unsqueeze(dim=1)],dim=-1)

            out,hidden_state=self.rnn(context.permute([1,0,2]),hidden_state)

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        outputs=self.dense(torch.cat(outputs,dim=0))
        return outputs.permute([1,0,2]),[en_output, hidden_state,
        enc_valid_lens]

    def attention_weights(self):
        return  self._attention_weights

decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
 num_layers=2)
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
 num_layers=2)
X = torch.zeros((4, 7), dtype=torch.long) # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape