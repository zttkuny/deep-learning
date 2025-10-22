import torch

def softmax(X):
    X_exp=torch.exp(X)
    x=X_exp.sum(axis=1,keepdim=True)
    return X_exp/x #运用广播机制

def main():
    x=-1
    name='李乾坤' if x>0 else '哈哈'
    print(name)


if __name__=='__main__':
    main()