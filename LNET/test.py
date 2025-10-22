import  torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'第{i}块GPU:{torch.cuda.get_device_name(i)} ')
if __name__=='__main__':
    main()