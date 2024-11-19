import torch
import torch.nn as nn



class IMNVMODEL(nn.Module) :
    def __init__(self, in_channels, adjust= False):
        super(IMNVMODEL, self).__init__()
        self.adjust = adjust
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        self.conv2 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1, stride=1, padding=0)
        self.rule2 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2_3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adjustCon = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0)
        self.rule3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x1_3 = self.conv1_3(x)

        x = torch.cat((x1_1, x1_2, x1_3), dim=1)
        x = self.rule2(self.conv2(x))


        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        xpool = self.pooling1(x)

        if self.adjust:
            xpool = self.adjustCon(xpool)

        x = torch.cat((x2_1, x2_2, x2_3, xpool), dim=1)
        x = self.rule3(self.conv3(x))
        return x



base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

def myvgg02(pretrained = False):
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [IMNVMODEL(in_channels)]
        elif v == 'C':
            layers += [IMNVMODEL(in_channels, adjust=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v


    layers += [nn.Conv2d(512, 1024, kernel_size=3, padding=1)]
    layers += [nn.Conv2d(1024, 1024, kernel_size=3, padding=6, dilation=6)]
    layers += [nn.ReLU(inplace=True)]
    layers += [IMNVMODEL(1024, adjust=True)]

    layers += [nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    layers += [IMNVMODEL(512)]

    layers += [nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    layers += [IMNVMODEL(256, adjust=True)]

    layers += [nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    layers += [nn.ReLU(inplace=True)]

    model = nn.ModuleList(layers)
    return model

if __name__ == "__main__":

    net = myvgg02()
    input = torch.rand(2, 3, 300, 300)
    for i, layer in enumerate(net):
        print(i, layer)
        print(input.shape)
        input = layer(input)
        print(input.shape)
        print('*' * 30)