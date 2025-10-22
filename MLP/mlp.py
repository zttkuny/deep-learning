from torch import nn
import torch
from d2l import torch as d2l

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用的设备:{torch.cuda.get_device_name(0)}')

    # 定义模型结构
    net = nn.Sequential(nn.Flatten()
                        , nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    # 定义初始化函数
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weight)
    net=net.to(device)

    # 定义超参数：批次大小，学习率，迭代轮次
    batch_size, lr, num_epochs = 2048, 0.1, 100

    # 定义损失函数（交叉熵函数）
    loss = nn.CrossEntropyLoss(reduction='none')

    # 定义优化算法（梯度下降）
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 定义测试集和训练集迭代器
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    for epoch in range(num_epochs):
        train_accu = 0.0
        for X, Y in train_iter:
            X,Y=X.to(device),Y.to(device)
            y_hat = net(X)  # 前向传播
            l = loss(y_hat, Y)  # 计算损失
            trainer.zero_grad()
            l.mean().backward()  # 反向传播
            trainer.step()

            # 计算训练正确预测数量
            train_accu += (y_hat.argmax(axis=1) == Y).sum().item()

        # 开始测试
        test_accu,test_num=0.0,0
        with torch.no_grad():
            for X,Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat=net(X)
                test_num+=(y_hat.argmax(axis=1)==Y).sum().item()

        train_acc = train_accu / (len(train_iter) * batch_size)  # 平均训练准确率
        test_acc=test_num/(len(test_iter)*batch_size)
        print(f"平均训练准确率:{train_acc:.5f}   平均训练准确率：{test_acc:.5f}")
if __name__=='__main__':
    main()