import random
import re

from d2l import torch as d2l
import torch
from matplotlib import pyplot as plt

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
     '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine(): #@save
        """将时间机器数据集加载到文本行的列表中"""
        with open(d2l.download('time_machine'), 'r') as f:
            lines = f.readlines()

        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corups_time_machine(max_tokens=-1):
    lines=read_time_machine()
    tokens=d2l.tokenize(lines) #分词
    vocab=d2l.Vocab(tokens) #生成词典

    corups=[vocab[word] for line in tokens for word in line ]

    if max_tokens>0:
        corups=corups[:max_tokens]

    return corups,vocab


def seq_data_iter_sequential(corpus, batch_size, num_steps): #@save
     """使用顺序分区生成一个小批量子序列"""
     # 从随机偏移量开始划分序列
     offset = random.randint(0, num_steps)
     num_tokens = ((len(corpus)- offset- 1) // batch_size) * batch_size
     Xs = torch.tensor(corpus[offset: offset + num_tokens])
     Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
     Xs, Ys = Xs.reshape(batch_size,-1), Ys.reshape(batch_size,-1)
     num_batches = Xs.shape[1] // num_steps
     for i in range(0, num_steps * num_batches, num_steps):
         X = Xs[:, i: i + num_steps]
         Y = Ys[:, i: i+num_steps]
         yield X, Y

def main():
    _,vocab=load_corups_time_machine()
    print(len(vocab))

if __name__=='__main__':
    main()