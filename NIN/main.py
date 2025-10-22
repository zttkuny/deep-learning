import matplotlib.pyplot as plt
from NiN import *
from d2l import torch as d2l

class NIN(nn.Module):
    def __init__(self):
        super(NIN,self).__init__()
        self.net=nn.Sequential(
                 nin_block(1, 96, kernel_size=11, strides=4, padding=0),
                    nn.MaxPool2d(3, stride=2),
                 nin_block(96, 256, kernel_size=5, strides=1, padding=2),
                 nn.MaxPool2d(3, stride=2),
                 nin_block(256, 384, kernel_size=3, strides=1, padding=1),
                 nn.MaxPool2d(3, stride=2),
                 nn.Dropout(0.5),
                 # 标签类别数是10
                 nin_block(384, 10, kernel_size=3, strides=1, padding=1),
                 nn.AdaptiveAvgPool2d((1, 1)),
                 # 将四维的输出转成二维的输出，其形状为(批量大小,10)
                 nn.Flatten())
    def forward(self,X):
        return self.net(X)

def show_shape(m):
    X=torch.randn(size=(1,1,224,224))
    for layer in m.net:
        X=layer(X)
        print(f'{layer.__class__.__name__}的输出形状为:{X.shape}')

def init_params(m):
    if isinstance(m,(nn.Linear,nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
def main()->None:
    plt.ion()
    net=NIN()
    net.apply(init_params)
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, "cuda:0")

if __name__=='__main__':
    main()