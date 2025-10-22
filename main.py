import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 生成真实数据集

true_w=torch.tensor([2,-3.4])
true_b=4.2
"""生成特征集和对应的标签"""
features,lables=d2l.synthetic_data(true_w,true_b,1000)
print(f"features:{features.shape}  labels:{lables.shape}")

# 生成数据迭代器
dataset=data.TensorDataset(features,lables)
data_iter=data.DataLoader(dataset,batch_size=10)

# 定义线性模型
net=nn.Sequential(nn.Linear(2,1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 权重初始化为N(0, 0.01)
net[0].bias.data.fill_(0)            # 偏置初始化为0

# 定义损失函数
loss=nn.MSELoss()

# 定义优化算法
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

# 训练过程
num_epochs=100
for i in range(num_epochs):
    for X,Y in data_iter:
        l=loss(net(X),Y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 计算当前epoch上所有样本上的平均损失
    loss(net(features),lables)
    print(f"epoch:{i+1}  loss:{l:f}")