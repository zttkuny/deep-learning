import torch.nn as nn
import  torch

# X是输入矩阵，K是卷积核
def corr2D(X,K):
    h,w=K.shape
    outM=torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))#输出矩阵大小
    for i in range(outM.shape[0]):
        for j in range(outM.shape[1]):
            outM[i][j]=(X[i:i+h,j:j+w]*K).sum()

    return outM

def main():
    t=(1,2,3)
    x,y,z=t
    print(f'x:{x},y:{y},z:{z}\n')

if __name__=='__main__':
    main()