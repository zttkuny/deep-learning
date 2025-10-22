import torch
import torch.nn as nn
from d2l import  torch as d2l

#批量归一化函数
def batch_norm(X,gamma,beta,moving_mean,moving_var,momentum,eps):
    if not torch.is_grad_enabled():
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2: #如果输入是二维的，说明是全连接层输出
            mean=X.mean(axis=0)
            var=((X-mean)**2).mean(dim=0)
        else: #如果输出是四维的，说明是卷积层输出
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)

        X_hat=(X-mean)/torch.sqrt(var+eps)

        moving_var=momentum*moving_var+(1.0-momentum)*mean
        moving_mean=momentum*moving_mean+(1.0-momentum)*var
        Y=gamma*X_hat+beta
        return  Y,moving_mean,moving_var

# 批量归一化层
class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super().__init__()
        assert num_dims in (2,4)
        if num_dims==2:
           shape=(1,num_features)
        else:
           shape=(1,num_features,1,1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self,X):
        if self.moving_var.device!=X.device:
            self.moving_var=self.moving_var.to(X.device)
            self.moving_mean=self.moving_mean.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(X,self.gamma,self.beta,
                                                      self.moving_mean,
                                                      self.moving_var,
                                                      0.9,
                                                      1e-3
                                                      )
        return Y
