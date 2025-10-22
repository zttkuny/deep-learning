import torch

from LNET import *

# 向多个设备分发参数
def get_params(params,device):
    new_params=[param.to(device) for param in params]
    for param in new_params:
        param.requires_grad_(True)
    return new_params

# 跨GPU设备求和
def allreduces(data):
    for i in range(1,len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1,len(data)):
        data[i][:]=data[0].to(data[i].device)

def split_batch(X,y,device):
    assert X.shape[0]==y.shape[0] #确保数据和标签的个数相同

    return (nn.parallel.scatter(X,device),nn.parallel.scatter(y,device))

def train_batch(X,y,device_params,devices,lr):# 训练小批次的函数
    # 切割函数
    X_shared,y_shared=split_batch(X,devices),split_batch(y,devices)

    # 每个设备的损失值之和组成的列表
    ls=[loss(lenet(X_shares,params),y_shares).sum()
        for X_shares,y_shares,params in zip(X_shared,y_shared,device_params)]

    for l in ls: #每个设备分别求梯度
        l.backward()

    with torch.no_grad:
       for i in range(len(device_params[0])):
           allreduces([device_params[c][i].grad for c in range(len(device_params))])

# 每个GPU上分别更新梯度
    for params in device_params:
        d2l.sgd(params,lr,X.shape[0])

def train(nums_gpu,lr,batch_size):
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

    devices=[d2l.try_gpu(i) for i in range(nums_gpu)]

    devices_params=[get_params(params,device) for device in devices]

    nums_epoch=10

    for epoch in range(nums_epoch):
        for X,y in train_iter:
            train_batch(X,y,devices_params,devices,lr)

def main():
    data=[1,2,3]
    data1=data[:]
    print(data[0] is data1[0])
if __name__=="__main__":
    main()