import torch
import torch.nn as nn

import configs.config as config
from utils.utils import (
    TemporalAttention,
    SpatioTemporalAttention,
    SpatioTemporalFCAttention,
    inconv,
    down,
    downfirst,
    up,
    outconv,
)


def generate_model(args):

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.model == "UNet_Attention":
        model = UNet_Attention(config.img_channel, config.class_num).to(device)
    elif args.model == "UNet":
        model = UNet(config.img_channel, config.class_num).to(device)

    return model


class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_Attention, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = downfirst(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.inconv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
        self.outconv = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1))
        self.outc = outconv(64, n_classes)
        self.attention_module = TemporalAttention(input_size=128, hidden_size=128, device="cuda")

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.inconv(x5)
            data.append(x6.unsqueeze(0))
        data = torch.cat(data, dim=0)
        data = torch.flatten(data, start_dim=2)

        test, _ = self.attention_module(data)
        output_tensor = test.permute(1, 0, 2)
        output_tensor_new = output_tensor.reshape(
            output_tensor.shape[0], output_tensor.shape[1], 8, 16
        )
        output = self.outconv(output_tensor_new)
        x = self.up1(output, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
