import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from Decoder_Encoder.Encoder import *
import MachineTrans.Trans as trans


def xavier_init_weight(m):
    if isinstance(m,nn.Linear): # 如果是线性层
        nn.init.xavier_uniform_(m.weight)

    if isinstance(m,nn.Embedding): # 如果是嵌入层
        nn.init.xavier_uniform_(m.weight)

    if isinstance(m,nn.GRU):
        for param in m._flat_weights_names:
            if 'weight' in param:
                nn.init.xavier_uniform_(m._parameters[param])

def train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device):
    net.apply(xavier_init_weight)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器

    loss = MaskedSoftmaxCELoss()

    net.train()  # 开启训练模式

    loss_list = []
    for epoch in range(num_epochs):
        loss_sum = 0
        for batch in train_iter:
            optimizer.zero_grad()  # 梯度清零
            X, X_len, Y, Y_len = [X.to(device) for X in batch]

            bos_ = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0]).unsqueeze(-1).to(device)
            dec_input = torch.cat([bos_, Y[:, :-1]], dim=1)  # 解码器输入中每个句子添加bos

            Y_hat = net(X, dec_input,X_len)

            l = loss(Y_hat, Y, Y_len)

            loss_sum += l.sum().item()
            l.sum().backward()

            optimizer.step()

            grad_clipping(net, 1)  # 防止梯度爆炸
        print(f'第{epoch + 1}轮，损失值：{loss_sum}\n')
        if epoch % 5 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'loss': loss_sum,
                        }, f'check_point_{epoch}.pth')
        loss_list.append(loss_sum)
    d2l.plot(X=[i for i in range(num_epochs)], Y=loss_list, xlabel='轮次', ylabel='loss', figsize=(6, 4), xlim=[0, 300])
    plt.show()


# def main():
#     # 训练迭代器
#     train_iter, src_vocab, tgt_vocab = trans.load_data_nmt(batch_size=64, num_steps=10, num_examples=100000)
#     net=EncoderDecoder(len(src_vocab),32,len(tgt_vocab),32,512,2,drop_out=0.5)
#     net.apply(xavier_init_weight)
#
#     lr,num_epochs=0.1,300
#
#     device=torch.device("cuda")
#     net.to(device)
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.005) # 优化器
#
#     loss = MaskedSoftmaxCELoss()
#
#     net.train() # 开启训练模式
#
#
#     loss_list = []
#     for epoch in range(num_epochs):
#         loss_sum=0
#         for batch in train_iter:
#             optimizer.zero_grad() # 梯度清零
#             X,X_len,Y,Y_len=[X.to(device) for X in batch]
#
#             bos_=torch.tensor([tgt_vocab['<bos>']]*Y.shape[0]).unsqueeze(-1).to(device)
#             dec_input=torch.cat([bos_,Y[:,:-1]],dim=1) # 解码器输入中每个句子添加bos
#
#             Y_hat,_=net(X,dec_input)
#
#             l=loss(Y_hat,Y,Y_len)
#
#             loss_sum+=l.sum().item()
#             l.sum().backward()
#
#             optimizer.step()
#
#             grad_clipping(net, 1) #防止梯度爆炸
#         print(f'第{epoch+1}轮，损失值：{loss_sum}\n')
#         if epoch%5==0:
#             torch.save({'epoch':epoch,
#                         'state_dict':net.state_dict(),
#                         'loss':loss_sum,
#                         },f'check_point_{epoch}.pth')
#         loss_list.append(loss_sum)
#     d2l.plot(X=[i for i in range(num_epochs)],Y=loss_list,xlabel='轮次',ylabel='loss',figsize=(6,4),xlim=[0,300])
#     plt.show()
#
#
# if __name__=='__main__':
#     main()