import os

import torch
from d2l import torch as d2l

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
 '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
    encoding='utf-8') as f:
        return f.read()

#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    lines=text.split('\n')
    source,target=[],[]
    for i,line in enumerate(lines):
        if num_examples and i>=num_examples:
            break

        parts=line.split('\t')
        if len(parts)==2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    source_test, target_test = [], []
    if num_examples:
        for i in range(num_examples,len(lines)):
            parts = lines[i].split('\t')
            if len(parts) == 2:
                source_test.append(parts[0].split(' '))
                target_test.append(parts[1].split(' '))
    return source,target,source_test,target_test


raw_text=read_data_nmt()
text = preprocess_nmt(raw_text)

#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

#@save
def build_array_nmt(lines, vocab, num_steps):
     """将机器翻译的文本序列转换成小批量"""
     lines = [vocab[l] for l in lines]
     lines = [l + [vocab['<eos>']] for l in lines]
     array = torch.tensor([truncate_pad(
     l, num_steps, vocab['<pad>']) for l in lines])
     valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
     return array, valid_len

#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
     """返回翻译数据集的迭代器和词表"""
     text = preprocess_nmt(read_data_nmt())
     source, target,source_test,target_test = tokenize_nmt(text, num_examples)
     src_vocab = d2l.Vocab(source, min_freq=2,
     reserved_tokens=['<pad>', '<bos>', '<eos>'])
     tgt_vocab = d2l.Vocab(target, min_freq=2,
     reserved_tokens=['<pad>', '<bos>', '<eos>'])
     src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
     tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

     train_iter=[]
     test_iter=[]
     for i in range(0,len(src_array),batch_size):
         train_iter.append((src_array[i:i+batch_size],src_valid_len[i:i+batch_size],
                            tgt_array[i:i+batch_size],tgt_valid_len[i:i+batch_size]))

     return train_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=64, num_steps=10, num_examples=100000)
