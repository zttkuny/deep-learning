import torch
import torch.nn as nn

def pool2d(X,kernel_size,mode='max'):
    k_h,k_w=kernel_size
    out=torch.zeros(size=(X.shape[0]-k_h+1,X.shape[1]-k_w+1))

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if mode=='max':
                out[i, j] = X[i:i + k_h, j:j + k_w].max().item()
            else:
                out[i, j] = X[i:i + k_h, j:j + k_w].mean().item()
    return  out


def main():
    X=torch.randn((5,5))
    print(f'X={X}\n')
    print(pool2d(X,(2,2)))
if __name__=='__main__':
    main()