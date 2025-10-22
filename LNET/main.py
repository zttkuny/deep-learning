import torch
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l
import evaluate_accuracy

# class Reshape(nn.Module):
#     def __init__(self):
#         super(Reshape,self).__init__()
#     def forward(self,X):
#         return X.reshape((1,1)+X.shape)
class LNET(nn.Module):
    def __init__(self):
        super(LNET,self).__init__()
        self.AlexNet=nn.Sequential(
             # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
             nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2),
             # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2),
             # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
             nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
             nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2),
             nn.Flatten(),
             # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
             nn.Dropout(p=0.5),
             nn.Linear(4096, 4096), nn.ReLU(),
             nn.Dropout(p=0.5),
             # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
             nn.Linear(4096, 10))

    def forward(self,X):
        return self.AlexNet(X)

# 初始化参数的函数
def init(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m)==nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():

    # 指定GPU
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    lnet=LNET()
    lnet.apply(init)
    lnet.to(device)

    x=torch.randn(1,1,224,224,device=device)
    for layer in lnet.AlexNet:
        x=layer(x)
        print(f'{layer.__class__.__name__}的输出形状：{x.shape}\n')

    # 三个超参数
    batch_size ,lr,num_epochs= 128,0.1,10

    # 构造数据迭代器
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)

    # 定义优化器和损失函数
    optimizer=torch.optim.SGD(params=lnet.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()

    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        lnet.train() # 开启训练模式（开启 drop_out和batch_norm）
        # 训练损失之和，训练准确率之和
        metric =d2l.Accumulator(3)
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=lnet(X)
            l=loss(y_hat,y)
            l.backward() #反向传播计算梯度
            optimizer.step() #更新参数

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy.evaluate_accuracy_gpu(lnet, test_iter,device)
        print(f'在第{epoch}轮次  loss :{train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')







if __name__=='__main__':
    main()
