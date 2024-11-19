import torch.nn as nn
from torch.hub import load_state_dict_from_url

from mytest.IVDNET.Blocks import *


class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1,
                                                  padding=0, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,
                                                  padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 4, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3,  conv_2_4], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3



base = [64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'M',512]

def myvgg01(pretrained = False):

    layers = []
    in_channels = 3
    act_fn = nn.LeakyReLU(0.2, inplace=True)
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = Conv_residual_conv_Inception_Dilation_asymmetric(in_channels, v, act_fn)
            layers += [conv2d]

            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    conv6 = Conv_residual_conv_Inception_Dilation_asymmetric(512, 1024, act_fn)

    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, conv7]

    model = nn.ModuleList(layers)
    if pretrained:
        print('没有预训练权重')
    return model

if __name__ == "__main__":

    input = torch.rand(1, 3, 300, 300)


    act_fn = nn.LeakyReLU(0.2, inplace=True)
    con = Conv_residual_conv_Inception_Dilation_asymmetric(3, 64, act_fn)
    input = con(input)
    print(input.shape)