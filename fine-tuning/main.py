import torchvision
import torch
import torch.nn as nn

pretrained_model=torchvision.models.resnet18(pretrained=True)

finetune_net=torchvision.models.resnet18(pretrained=True)
finetune_net.fc=nn.Linear(pretrained_model.fc.in_features,2)
nn.init.zeros_(finetune_net.fc.weight)
nn.init.zeros_(finetune_net.fc.bias)
with torch.no_grad():
    finetune_net.fc.weight[0][:]=pretrained_model.fc.weight[943]

def main():
    words="lqkwqwqaa"
    print(list(words))

if __name__=='__main__':
    main()