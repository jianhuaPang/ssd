import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

def vgg(pretrained = False):
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    model = nn.ModuleList(layers)

    if pretrained:
        model_state = torch.load('../model_data/ssd_weights.pth', map_location=torch.device('cpu'))
    else:
        model_state = torch.load('model_data/ssd_weights.pth', map_location=torch.device('cuda'))



    state_dict = {k.replace('vgg.', '') : v for k, v in model_state.items()}
    model.load_state_dict(state_dict, strict = False)
    return model

if __name__ == "__main__":
    net = vgg(pretrained=True)
    input = torch.rand(2, 3, 300, 300)
    for i, layer in enumerate(net):
        print(i, layer)
        input = layer(input)
        print(input.shape)
        print('*'*30)

