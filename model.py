import torch
import torch.nn as nn


class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_kernel_size=3, instance_norm=False, channel_attention=True):
        super(MultiConv, self).__init__()
        self.multiconv_layers = nn.ModuleList()
        for kernel_size in range(3, max_kernel_size+1, 2):
            if instance_norm:
                doubleconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True, padding=kernel_size//2, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False, padding=kernel_size//2, padding_mode='reflect'),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2))
            else:
                doubleconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True, padding=kernel_size//2, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=True, padding=kernel_size//2, padding_mode='reflect'),
                nn.LeakyReLU(0.2))
            self.multiconv_layers.append(doubleconv)
        if channel_attention:
            self.channel_attention = ChannelAttention(out_channels*len(self.multiconv_layers), reduction_ratio=len(self.multiconv_layers))
        else:
            self.channel_attention = None


    def forward(self, x):
        if self.channel_attention is None:
            return torch.cat([multiconv(x) for multiconv in self.multiconv_layers], 1)
        else:
            return self.channel_attention(torch.cat([multiconv(x) for multiconv in self.multiconv_layers], 1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction_ratio, kernel_size=1), nn.ReLU(), nn.Conv2d(channels//reduction_ratio, channels, kernel_size=1))


    def forward(self, x):
        return x * torch.sigmoid(self.channel_attention(x))


class PyNetCA(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=16, instance_norm=True, channel_attention=True):
        super(PyNetCA, self).__init__()
        self.level0_conv1 = MultiConv(hidden_channels*2, hidden_channels*2, 9, False, channel_attention)
        self.level0_conv2 = nn.Conv2d(hidden_channels*8, out_channels*2*2, 1)
        self.level0_up1 = nn.PixelShuffle(2)

        self.level1_conv1 = MultiConv(in_channels, hidden_channels*2, 3, False, channel_attention)
        self.level1_conv2 = MultiConv(hidden_channels*4, hidden_channels*2, 5, False, channel_attention)
        self.level1_conv3 = MultiConv(hidden_channels*6, hidden_channels*2, 7, instance_norm, channel_attention)
        self.level1_conv4 = MultiConv(hidden_channels*6, hidden_channels*2, 9, instance_norm, channel_attention)
        self.level1_conv5 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
        self.level1_conv6 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
        self.level1_conv7 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
        self.level1_conv8 = MultiConv(hidden_channels*8, hidden_channels*2, 7, instance_norm, channel_attention)
        self.level1_conv9 = MultiConv(hidden_channels*8, hidden_channels*2, 5, instance_norm, channel_attention)
        self.level1_conv10 = MultiConv(hidden_channels*8, hidden_channels*2, 3, False, channel_attention)
        self.level1_conv11 = nn.Conv2d(hidden_channels*2, out_channels, 3, padding=3//2, padding_mode='reflect')
        self.level1_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*4, hidden_channels*2, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
        self.level1_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*4, hidden_channels*2, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))

        self.level2_conv1 = nn.Sequential(nn.MaxPool2d(2, 2),MultiConv(hidden_channels*2, hidden_channels*4, 3, instance_norm, channel_attention))
        self.level2_conv2 = MultiConv(hidden_channels*8, hidden_channels*4, 5, instance_norm, channel_attention)
        self.level2_conv3 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
        self.level2_conv4 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
        self.level2_conv5 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
        self.level2_conv6 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
        self.level2_conv7 = MultiConv(hidden_channels*16, hidden_channels*4, 5, instance_norm, channel_attention)
        self.level2_conv8 = MultiConv(hidden_channels*12, hidden_channels*4, 3, instance_norm, channel_attention)
        self.level2_conv9 = nn.Conv2d(hidden_channels*4, out_channels, 3, padding=3//2, padding_mode='reflect')
        self.level2_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*8, hidden_channels*4, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
        self.level2_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*8, hidden_channels*4, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))

        self.level3_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*4, hidden_channels*8, 3, instance_norm, channel_attention))
        self.level3_conv2 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
        self.level3_conv3 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
        self.level3_conv4 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
        self.level3_conv5 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
        self.level3_conv6 = MultiConv(hidden_channels*32, hidden_channels*8, 3, instance_norm, channel_attention)
        self.level3_conv7 = nn.Conv2d(hidden_channels*8, out_channels, 3, padding=3//2, padding_mode='reflect')
        self.level3_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*16, hidden_channels*8, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
        self.level3_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*16, hidden_channels*8, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))

        self.level4_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*8, hidden_channels*16, 3, instance_norm, channel_attention))
        self.level4_conv2 = MultiConv(hidden_channels*32, hidden_channels*16, 3, instance_norm, channel_attention)
        self.level4_conv3 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
        self.level4_conv4 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
        self.level4_conv5 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
        self.level4_conv6 = MultiConv(hidden_channels*32, hidden_channels*16, 3, instance_norm, channel_attention)
        self.level4_conv7 = nn.Conv2d(hidden_channels*16, out_channels, 3, padding=3//2, padding_mode='reflect')
        self.level4_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*32, hidden_channels*16, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
        self.level4_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*32, hidden_channels*16, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))

        self.level5_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*16, hidden_channels*32, 3, instance_norm, channel_attention))
        self.level5_conv2 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
        self.level5_conv3 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
        self.level5_conv4 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
        self.level5_conv5 = nn.Conv2d(hidden_channels*32, out_channels, 3, padding=3//2, padding_mode='reflect')


    def get_parameters(self, level=None):
        assert level in [None, 0, 1, 2, 3, 4, 5]
        if level is None:
            return self.parameters()
        else:
            params = []
            for name, param in self.named_parameters():
                if (f'level{level}' in name) or ('conv1' in name):
                    params.append(param)
            return params


    def forward(self, x, level):
        if level<=5:
            x = self.level1_conv1(x)
            x1 = x
            x = self.level2_conv1(x)
            x2 = x
            x = self.level3_conv1(x)
            x3 = x
            x = self.level4_conv1(x)
            x4 = x
            x = self.level5_conv1(x)
            x = self.level5_conv2(x)
            x = self.level5_conv3(x)
            x = self.level5_conv4(x)
            y5 = x

        if level<=4:
            x = torch.cat([x4, self.level4_up1(y5)], 1)
            x = self.level4_conv2(x)
            x = self.level4_conv3(x) + x
            x = self.level4_conv4(x) + x
            x = self.level4_conv5(x)
            x = torch.cat([x, self.level4_up2(y5)], 1)
            x = self.level4_conv6(x)
            y4 = x

        if level<=3:
            x = torch.cat([x3, self.level3_up1(y4)], 1)
            x = self.level3_conv2(x) + x
            x = self.level3_conv3(x) + x
            x = self.level3_conv4(x) + x
            x = self.level3_conv5(x)
            x = torch.cat([x, x3, self.level3_up2(y4)], 1)
            x = self.level3_conv6(x)
            y3 = x

        if level<=2:
            x = torch.cat([x2, self.level2_up1(y3)], 1)
            x = self.level2_conv2(x)
            x = torch.cat([x, x2], 1)
            x = self.level2_conv3(x) + x
            x = self.level2_conv4(x) + x
            x = self.level2_conv5(x) + x
            x = self.level2_conv6(x)
            x = torch.cat([x, x2], 1)
            x = self.level2_conv7(x)
            x = torch.cat([x, self.level2_up2(y3)], 1)
            x = self.level2_conv8(x)
            y2 = x

        if level<=1:
            x = torch.cat([x1, self.level1_up1(y2)], 1)
            x = self.level1_conv2(x)
            x = torch.cat([x, x1], 1)
            x = self.level1_conv3(x)
            x = self.level1_conv4(x)
            x = self.level1_conv5(x) + x
            x = self.level1_conv6(x) + x
            x = self.level1_conv7(x) + x
            x = self.level1_conv8(x)
            x = torch.cat([x, x1], 1)
            x = self.level1_conv9(x)
            x = torch.cat([x, self.level1_up2(y2), x1], 1)
            x = self.level1_conv10(x)

        if level==0:
            x = self.level0_conv1(x)
            x = self.level0_conv2(x)
            x = self.level0_up1(x)
        elif level==1:
            x = self.level1_conv11(x)
        elif level==2:
            x = self.level2_conv9(x)
        elif level==3:
            x = self.level3_conv7(x)
        elif level==4:
            x = self.level4_conv7(x)
        elif level==5:
            x = self.level5_conv5(x)
        return torch.tanh(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### Baseline PyNet model is implemented but not used ###
# class PyNet(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, hidden_channels=16, instance_norm=True, channel_attention=True):
#         super(PyNet, self).__init__()
#         self.level0_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#         self.level0_conv1 = nn.Conv2d(hidden_channels, out_channels, 3, padding=3//2, padding_mode='reflect')
#
#         self.level1_conv1 = MultiConv(in_channels, hidden_channels*2, 3, False, channel_attention)
#         self.level1_conv2 = MultiConv(hidden_channels*4, hidden_channels*2, 5, False, channel_attention)
#         self.level1_conv3 = MultiConv(hidden_channels*6, hidden_channels*2, 7, instance_norm, channel_attention)
#         self.level1_conv4 = MultiConv(hidden_channels*6, hidden_channels*2, 9, instance_norm, channel_attention)
#         self.level1_conv5 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
#         self.level1_conv6 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
#         self.level1_conv7 = MultiConv(hidden_channels*8, hidden_channels*2, 9, instance_norm, channel_attention)
#         self.level1_conv8 = MultiConv(hidden_channels*8, hidden_channels*2, 7, instance_norm, channel_attention)
#         self.level1_conv9 = MultiConv(hidden_channels*8, hidden_channels*2, 5, instance_norm, channel_attention)
#         self.level1_conv10 = MultiConv(hidden_channels*8, hidden_channels*2, 3, False, channel_attention)
#         self.level1_conv11 = nn.Conv2d(hidden_channels*2, out_channels, 3, padding=3//2, padding_mode='reflect')
#         self.level1_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*4, hidden_channels*2, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#         self.level1_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*4, hidden_channels*2, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#
#         self.level2_conv1 = nn.Sequential(nn.MaxPool2d(2, 2),MultiConv(hidden_channels*2, hidden_channels*4, 3, instance_norm, channel_attention))
#         self.level2_conv2 = MultiConv(hidden_channels*8, hidden_channels*4, 5, instance_norm, channel_attention)
#         self.level2_conv3 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
#         self.level2_conv4 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
#         self.level2_conv5 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
#         self.level2_conv6 = MultiConv(hidden_channels*12, hidden_channels*4, 7, instance_norm, channel_attention)
#         self.level2_conv7 = MultiConv(hidden_channels*16, hidden_channels*4, 5, instance_norm, channel_attention)
#         self.level2_conv8 = MultiConv(hidden_channels*12, hidden_channels*4, 3, instance_norm, channel_attention)
#         self.level2_conv9 = nn.Conv2d(hidden_channels*4, out_channels, 3, padding=3//2, padding_mode='reflect')
#         self.level2_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*8, hidden_channels*4, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#         self.level2_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*8, hidden_channels*4, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#
#         self.level3_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*4, hidden_channels*8, 3, instance_norm, channel_attention))
#         self.level3_conv2 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
#         self.level3_conv3 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
#         self.level3_conv4 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
#         self.level3_conv5 = MultiConv(hidden_channels*16, hidden_channels*8, 5, instance_norm, channel_attention)
#         self.level3_conv6 = MultiConv(hidden_channels*32, hidden_channels*8, 3, instance_norm, channel_attention)
#         self.level3_conv7 = nn.Conv2d(hidden_channels*8, out_channels, 3, padding=3//2, padding_mode='reflect')
#         self.level3_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*16, hidden_channels*8, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#         self.level3_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*16, hidden_channels*8, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#
#         self.level4_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*8, hidden_channels*16, 3, instance_norm, channel_attention))
#         self.level4_conv2 = MultiConv(hidden_channels*32, hidden_channels*16, 3, instance_norm, channel_attention)
#         self.level4_conv3 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
#         self.level4_conv4 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
#         self.level4_conv5 = MultiConv(hidden_channels*16, hidden_channels*16, 3, instance_norm, channel_attention)
#         self.level4_conv6 = MultiConv(hidden_channels*32, hidden_channels*16, 3, instance_norm, channel_attention)
#         self.level4_conv7 = nn.Conv2d(hidden_channels*16, out_channels, 3, padding=3//2, padding_mode='reflect')
#         self.level4_up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*32, hidden_channels*16, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#         self.level4_up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(hidden_channels*32, hidden_channels*16, 3, padding=3//2, padding_mode='reflect'), nn.LeakyReLU(0.2))
#
#         self.level5_conv1 = nn.Sequential(nn.MaxPool2d(2, 2), MultiConv(hidden_channels*16, hidden_channels*32, 3, instance_norm, channel_attention))
#         self.level5_conv2 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
#         self.level5_conv3 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
#         self.level5_conv4 = MultiConv(hidden_channels*32, hidden_channels*32, 3, instance_norm, channel_attention)
#         self.level5_conv5 = nn.Conv2d(hidden_channels*32, out_channels, 3, padding=3//2, padding_mode='reflect')
#
#
#     def get_parameters(self, level=None):
#         assert level in [None, 0, 1, 2, 3, 4, 5]
#         if level is None:
#             return self.parameters()
#         else:
#             params = []
#             for name, param in self.named_parameters():
#                 if (f'level{level}' in name) or ('conv1' in name):
#                     params.append(param)
#             return params
#
#     def count_parameters(self):
#         return count_parameters(self)
#
#
#     def forward(self, x, level):
#         if level<=5:
#             x = self.level1_conv1(x)
#             x1 = x
#             x = self.level2_conv1(x)
#             x2 = x
#             x = self.level3_conv1(x)
#             x3 = x
#             x = self.level4_conv1(x)
#             x4 = x
#             x = self.level5_conv1(x)
#             x = self.level5_conv2(x)
#             x = self.level5_conv3(x)
#             x = self.level5_conv4(x)
#             y5 = x
#
#         if level<=4:
#             x = torch.cat([x4, self.level4_up1(y5)], 1)
#             x = self.level4_conv2(x)
#             x = self.level4_conv3(x) + x
#             x = self.level4_conv4(x) + x
#             x = self.level4_conv5(x)
#             x = torch.cat([x, self.level4_up2(y5)], 1)
#             x = self.level4_conv6(x)
#             y4 = x
#
#         if level<=3:
#             x = torch.cat([x3, self.level3_up1(y4)], 1)
#             x = self.level3_conv2(x) + x
#             x = self.level3_conv3(x) + x
#             x = self.level3_conv4(x) + x
#             x = self.level3_conv5(x)
#             x = torch.cat([x, x3, self.level3_up2(y4)], 1)
#             x = self.level3_conv6(x)
#             y3 = x
#
#         if level<=2:
#             x = torch.cat([x2, self.level2_up1(y3)], 1)
#             x = self.level2_conv2(x)
#             x = torch.cat([x, x2], 1)
#             x = self.level2_conv3(x) + x
#             x = self.level2_conv4(x) + x
#             x = self.level2_conv5(x) + x
#             x = self.level2_conv6(x)
#             x = torch.cat([x, x2], 1)
#             x = self.level2_conv7(x)
#             x = torch.cat([x, self.level2_up2(y3)], 1)
#             x = self.level2_conv8(x)
#             y2 = x
#
#         if level<=1:
#             x = torch.cat([x1, self.level1_up1(y2)], 1)
#             x = self.level1_conv2(x)
#             x = torch.cat([x, x1], 1)
#             x = self.level1_conv3(x)
#             x = self.level1_conv4(x)
#             x = self.level1_conv5(x) + x
#             x = self.level1_conv6(x) + x
#             x = self.level1_conv7(x) + x
#             x = self.level1_conv8(x)
#             x = torch.cat([x, x1], 1)
#             x = self.level1_conv9(x)
#             x = torch.cat([x, self.level1_up2(y2), x1], 1)
#             x = self.level1_conv10(x)
#
#         if level==0:
#             x = self.level0_up1(x)
#             x = self.level0_conv1(x)
#         elif level==1:
#             x = self.level1_conv11(x)
#         elif level==2:
#             x = self.level2_conv9(x)
#         elif level==3:
#             x = self.level3_conv7(x)
#         elif level==4:
#             x = self.level4_conv7(x)
#         elif level==5:
#             x = self.level5_conv5(x)
#         return torch.tanh(x)
