import torch
import torch.nn as nn
import torch.nn.init as init


from mytest.SELayer import SELayer
from nets.vgg import vgg as add_vgg
from nets.convnextBone import Block


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # Block 6
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    return nn.ModuleList(layers)

# 创建 上采样的部分
def add_up(dp_rates=0.):
    layer_scale_init_value = 1e-6

    uplayers = []
    convlayers = []

    # uplayersBlock 1
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)]  # 用来调整尺寸的

    # convlayersBlock 1
    convlayers += [nn.Conv2d(512, 256, kernel_size=3, padding=1)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(256)]
    convlayers += [nn.ReLU()]

    # uplayersBlock 2
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)]  # 用来调整尺寸的

    # convlayersBlock 2
    convlayers += [nn.Conv2d(512, 256, kernel_size=3, padding=1)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(256)]
    convlayers += [nn.ReLU()]

    # uplayersBlock 3
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)]  # 用来调整通道数的

    # convlayersBlock 3
    convlayers += [nn.Conv2d(1024, 512, kernel_size=3, padding=1)]
    convlayers += [Block(dim=512, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=512, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(512)]
    convlayers += [nn.ReLU()]

    # uplayersBlock 4
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0)]  # 用来调整尺寸的

    # convlayersBlock 4
    convlayers += [nn.Conv2d(2048, 1024, kernel_size=3, padding=1)]
    convlayers += [Block(dim=1024, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=1024, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(1024)]
    convlayers += [nn.ReLU()]

    # uplayersBlock 5
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)]  # 用来调整通道数的

    # convlayersBlock 5
    convlayers += [nn.Conv2d(1024, 512, kernel_size=3, padding=1)]
    convlayers += [Block(dim=512, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=512, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(512)]
    convlayers += [nn.ReLU()]

    # uplayersBlock 6
    uplayers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    uplayers += [nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=0)]  # 用来调整通道数和尺寸


    # convlayersBlock 6
    convlayers += [nn.Conv2d(512, 256, kernel_size=3, padding=1)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [Block(dim=256, drop_path=dp_rates,layer_scale_init_value=layer_scale_init_value)]
    convlayers += [nn.BatchNorm2d(256)]
    convlayers += [nn.ReLU()]


    uplayersmodel = nn.ModuleList(uplayers)
    convlayersmodel = nn.ModuleList(convlayers)

    return uplayersmodel, convlayersmodel

def add_conv1x1():
    layers = []
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    return nn.ModuleList(layers)

def add_selayer(falg = True):
    layers = []
    layers += [SELayer(64)]
    layers += [SELayer(128)]
    layers += [SELayer(256)]
    layers += [SELayer(512)]
    layers += [SELayer(1024)]
    layers += [SELayer(512)]
    layers += [SELayer(256)]
    layers += [SELayer(256)]

    if not falg:
        layers.reverse()
    return nn.ModuleList(layers)


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size, 1], padding=tuple([padding, 0]), dilation=(dilation, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0, padding]), dilation=(1, dilation)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model

class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1,padding=0, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,padding=2, dilation=2)
        self.conv_2_5 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,padding=4, dilation=4)
        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,dilation=1)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


def add_LVDNet():

    act_fn = nn.LeakyReLU(0.01, inplace=True)
    layers = []
    # Block 0

    layers += [nn.Identity()]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(3, 64, act_fn)]
    layers += [nn.BatchNorm2d(64)]
    layers += [act_fn]

    # Block 1

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(64, 128, act_fn)]
    layers += [nn.BatchNorm2d(128)]
    layers += [act_fn]

    # Block 2

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(128, 256, act_fn)]
    layers += [nn.BatchNorm2d(256)]
    layers += [act_fn]

    # Block 3

    layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(256, 512, act_fn)]
    layers += [nn.BatchNorm2d(512)]
    layers += [act_fn]

    # Block 4

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(512, 1024, act_fn)]
    layers += [nn.BatchNorm2d(1024)]
    layers += [act_fn]

    # Block 5

    layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(1024, 512, act_fn)]
    layers += [nn.BatchNorm2d(512)]
    layers += [act_fn]

    # Block 6
    # 10,10,512 -> 10,10,1024 -> 5,5,256
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(512, 256, act_fn)]
    layers += [nn.BatchNorm2d(256)]
    layers += [act_fn]

    # Block 7
    # 5,5,256 -> 3,3,256 -> 3,3,256
    layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)]
    layers += [Conv_residual_conv_Inception_Dilation_asymmetric(256, 256, act_fn)]
    layers += [nn.BatchNorm2d(256)]
    layers += [act_fn]

    return nn.ModuleList(layers)


class MYMODEL37(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(MYMODEL37, self).__init__()
        self.num_classes    = num_classes
        print('----------------------   应用 nets.mynet37.py 模型的 MYMODEL37 网络  ----------------------')

        self.vgg        = add_vgg(pretrained)
        self.extras     = add_extras(1024, backbone_name)
        self.L2Norm00 = L2Norm(128, 20)
        self.L2Norm01 = L2Norm(256, 20)
        self.L2Norm02     = L2Norm(512, 20)
        mbox            = [2, 2, 2, 2, 2, 2]

        loc_layers      = []
        conf_layers     = []
        backbone_source = [14, 21, -2]

        for k, v in enumerate(backbone_source):
            loc_layers  += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
            conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]

        for k, v in enumerate(self.extras[1::2], 2):
            loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]


        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name
        self.up, self.upconv = add_up()
        self.downselayer = add_selayer(True)
        self.upselayer = add_selayer(False)

        self.conv1x1 = add_conv1x1()
        self.lvdnet = add_LVDNet()

    def forward(self, x):
        sources = list()
        loc     = list()
        conf    = list()
        countSE=0

        x1 = self.lvdnet[0](x)
        x1 = self.lvdnet[1](x1)
        x = self.vgg[0](x)
        x = self.vgg[1](x)
        x = self.vgg[2](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        x = x + x1
        x = self.lvdnet[2](x)
        x = self.lvdnet[3](x)

        x1 = self.lvdnet[4](x)
        x1 = self.lvdnet[5](x1)
        x = self.vgg[4](x)
        x = self.vgg[5](x)
        x = self.vgg[6](x)
        x = self.vgg[7](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        x = x + x1
        x = self.lvdnet[6](x)
        x = self.lvdnet[7](x)

        x1 = self.lvdnet[8](x)
        x1 = self.lvdnet[9](x1)
        for i in range(9, 15):
            x = self.vgg[i](x)
        x = x + x1
        x = self.lvdnet[10](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[11](x)


        x1 = self.lvdnet[12](x)
        x1 = self.lvdnet[13](x1)
        for i in range(16, 22):
            x = self.vgg[i](x)
        x = x + x1
        x = self.lvdnet[14](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[15](x)


        x1 = self.lvdnet[16](x)
        x1 = self.lvdnet[17](x1)
        for i in range(23,34):
            x = self.vgg[i](x)
        x = x + x1
        x = self.lvdnet[18](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[19](x)

        x1 = self.lvdnet[20](x)
        x1 = self.lvdnet[21](x1)
        x = self.extras[0](x)
        x = self.extras[1](x)
        x = x + x1
        x = self.lvdnet[22](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[23](x)

        x1 = self.lvdnet[24](x)
        x1 = self.lvdnet[25](x1)
        x = self.extras[2](x)
        x = self.extras[3](x)
        x = x + x1
        x = self.lvdnet[26](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[27](x)

        x1 = self.lvdnet[28](x)
        x1 = self.lvdnet[29](x1)
        x = self.extras[4](x)
        x = self.extras[5](x)
        x = x + x1
        x = self.lvdnet[30](x)
        x = self.downselayer[countSE](x)
        countSE = countSE + 1
        sources.append(x)
        x = self.lvdnet[31](x)

        for i in range(len(self.conv1x1)):
            x = self.conv1x1[i](x)

        sources.append(x)
        downlenth = len(sources)

        countSE = 0
        for i in range(len(self.up) // 2):
            # 上采样部分
            x = self.up[2 * i](x)
            x = self.up[2 * i + 1](x)

            x = torch.cat((x, sources[downlenth - 2 - i]), dim=1)

            for k in range(5):
                x = self.upconv[5 * i + k](x)
                if k == 4:
                    x = self.upselayer[countSE](x)
                    countSE = countSE + 1
                    sources.append(x)

        sources = sources[-6:]
        sources.reverse()


        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

if __name__ == '__main__':
    net = MYMODEL37(2, 'vgg', pretrained=True)
    net.cpu()
    print(net)
    input = torch.rand(2, 3, 300, 300).cpu()
    output = net(input)


