import torch
from d2l import torch as d2l
import torch.nn as nn
from matplotlib import pyplot as plt


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 1000  # 1000个样本
    x = torch.arange(1, T + 1, dtype=torch.float32)
    y_lable = torch.sin(0.01 * x) + torch.normal(0, 0.2, (T,))


    tau=4 # 马尔可夫假设中当前数据和只和前tau个数据有关

    # 有T-tau个样本，因为前tau个样本没有完整的过去数据，舍弃
    features=torch.zeros(size=(T-tau,tau),dtype=torch.float32)

    for i in range(tau):
        features[:,i]=y_lable[i:i+T-tau]
    labels=y_lable[tau:].reshape((-1,1))

    batch_size,n_train=16,600 # 训练时只用前n_train条数据，后面400条做验证

    net=get_net() # 得到模型
    net.to(device)

    # 定义优化器和损失函数
    loss=nn.MSELoss()
    trainer=torch.optim.Adam(net.parameters(),lr=1e-3)

    num_epochs=10
    train_losses=[] # 记录每轮epoch的平均损失


    # 开始训练
    for epoch in range(num_epochs):
        iter_count = 0 # 记录每轮epoch的总迭代次数
        loss_sum = 0.0  # 记录每轮epoch的总损失值
        for X,y in train_iter(features,labels,batch_size,n_train):
            iter_count=iter_count+1
            X=X.to(device)
            y=y.to(device)
            trainer.zero_grad()
            y_hat=net(X)
            l=loss(y,y_hat)
            loss_sum+=l.item()
            l.backward()
            trainer.step()
        print(f'第{epoch+1}轮的平均损失值：{loss_sum/iter_count}\n')
        train_losses.append(loss_sum/iter_count)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制模型预测值
    with torch.no_grad():
        prediction=[net(features[i].to(device)).item() for i in range(T-tau)]

        multistep_prediction=list(prediction[:601-tau])
        train_set=features[598,:].to(device)
        for i in range(598,T-tau+1):
            y_hat=net(train_set[-4:])
            multistep_prediction.append((y_hat.item()))
            train_set=torch.cat([train_set,y_hat])
        print(f'multistep_prediction.shape:{len(multistep_prediction)}')

    d2l.plot([[i for i in range(tau + 1, T + 1)], x.numpy(),[i for i in range(4,T)]],  # x列表
             [prediction, y_lable,multistep_prediction],  # y列表
             'time', 'value',
             xlim=[1, 1000],
             axes=ax,
             fmts=['b-', 'r--','g*'])  # 蓝色实线 + 红色虚线
    plt.show()









def train_iter(features,lables,batch_size,n_train):
    indices=torch.randperm(n_train)

    for i in range(0,n_train,batch_size):
        yield features[indices[i:i+batch_size]],lables[indices[i:i+batch_size]]

def get_net():
    net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_param)
    return  net

def init_param(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

if __name__=='__main__':
    main()

