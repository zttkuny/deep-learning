import torch

from functional import *


def func():
    corups, vocab = load_corups_time_machine()

    net = RNNModelScratch(len(vocab), 512, torch.device("cuda"), getparam, init_rnn_state, rnn)

    num_epochs = 10
    lr = 0.1  # 学习率

    # 优化算法

    loss = nn.CrossEntropyLoss()

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    out = []

    for epoch in range(num_epochs):
        train_iter = seq_data_iter_sequential(corups, 2, 5)
        lst = train_epoch_ch8(net, train_iter, loss, updater, torch.device("cuda"), False)
        if len(lst) != 0:
            mean = sum(lst) / len(lst)
            print(f'第{epoch + 1}轮损失均值：{mean}')
            out.append(mean)

    torch.save(net.params, 'my_model_param.pth')
    d2l.plot([i for i in range(len(out))], out, xlabel="time", ylabel="loss", fmts="r-", figsize=(6, 4))
    plt.show()





