import numpy as np
import torch.nn as nn
import torch


class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()

        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)


class UpConvRelu(nn.Module):
    """
    二倍上采样
    """
    def __init__(self, channels_in, channels_out):
        super(UpConvRelu, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_in, channels_out, 3, 1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, imgs):
        imgs = self.up(imgs)
        return imgs


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(UpSample, self).__init__()

        self.conv_first = ConvRelu(1, in_channels)

        layers = [UpConvRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = UpConvRelu(out_channels, out_channels)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        self.conv_last = ConvRelu(in_channels, 1, init_zero=True)

    def forward(self, msgs):
        x = self.conv_first(msgs)
        x = self.layers(x)
        x =  self.conv_last(x)

        return x


class CopyUpNet(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(CopyUpNet, self).__init__()

        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))

        self.msg_up1 = UpSample(channels, channels, blocks=stride_blocks)
        self.msg_up2 = UpSample(channels, channels, blocks=stride_blocks)
        self.msg_up3 = UpSample(channels, channels, blocks=stride_blocks)

    def forward(self, msgs):
        h = int(np.sqrt(msgs.shape[1]))
        msgs_reshape = msgs.view(-1, 1, h, h)
        up1 = self.msg_up1(msgs_reshape)
        up2 = self.msg_up2(msgs_reshape)
        up3 = self.msg_up3(msgs_reshape)
        
        return torch.concat([up1, up2, up3], dim=1)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(DownSample, self).__init__()

        self.conv_first = ConvRelu(3, in_channels)

        layers = [ConvRelu(in_channels, out_channels, 2)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvRelu(out_channels, out_channels, 2)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        self.conv_last = ConvRelu(in_channels, 1, init_zero=True)

    def forward(self, imgs):
        x = self.conv_first(imgs)
        x = self.layers(x)
        x =  self.conv_last(x)

        return x


class DownNet(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(DownNet, self).__init__()

        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))

        self.img_down = DownSample(channels, channels, blocks=stride_blocks)

    def forward(self, imgs):
        down = self.img_down(imgs).view(imgs.shape[0], -1)

        return down
