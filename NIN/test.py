import torch

def main()->None:
    print(torch.cuda.device_count())

if __name__=='__main__':
    main()
