import torch
import torch.nn as nn
from d2l import torch as d2l

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式，关闭 dropout/batchnorm 更新

    if device is None:
        # 自动获取模型参数所在的设备
        device = next(iter(net.parameters())).device

    # metric[0] 累加正确预测数，metric[1] 累加样本总数
    metric = d2l.Accumulator(2)

    with torch.no_grad():  # 评估阶段不计算梯度
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微调时 X 可能是多个输入
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            # 累加正确预测数和样本总数
            metric.add(d2l.accuracy(y_hat, y), y.numel())

    return metric[0] / metric[1]  # 返回整体准确率
