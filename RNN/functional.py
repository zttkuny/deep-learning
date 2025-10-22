import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from corpus import *

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    lst = []
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state = state.detach()
            else:
                state=state.detach()

        Y = Y.reshape((-1))
        X, Y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, Y).mean()
        lst.append((l.item()))
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
    return lst


# 预测给的字符串的后面多个字符
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 将输入字符变化为批次大小*时间步数形状
    get_inputs = lambda: torch.tensor(outputs[-1], device=device).reshape((1, 1))

    # 预热，更新隐状态
    for i in range(1, len(prefix)):
        X = F.one_hot(torch.tensor(vocab[prefix[i]], device=device)).reshape((1, -1))
        _, state = net(get_inputs(), state)
        outputs.append(vocab[prefix[i]])

    for i in range(num_preds):
        out, state = net(get_inputs(), state)
        outputs.append(int(out.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


class RNNModelScratch:  # @save
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_hh, W_xh, b_h, W_hq, b_q = params

    H1= state

    outputs = []

    for X in inputs:
        H1 = torch.matmul(H1, W_hh) + torch.matmul(X, W_xh) + b_h
        O = torch.matmul(H1, W_hq) + b_q
        outputs.append(O)
    return torch.cat(outputs), H1


def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


#  初始隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)


# 初始化权重参数
def getparam(vacab_size, num_hiddens, device):
    num_inputs = num_outputs = vacab_size  # 输入和输出维度都和词表维度相同

    # 隐藏层参数
    Wxh = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    Whh = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    bh = torch.randn((num_hiddens,), device=device)

    # 输出层参数
    Whp = torch.randn((num_hiddens, num_outputs), device=device) * 0.01
    bp = torch.randn((num_outputs,), device=device)

    params = [Whh, Wxh, bh, Whp, bp]

    for param in params:
        param.requires_grad_(True)  # 开启梯度计算

    return params


def grad_clipping(net, theta):  # @save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

def main():
    params=getparam(4580,512,torch.device("cuda"))
    print(params)
    torch.save(params,'my_model_params.pth')

