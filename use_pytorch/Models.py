import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNN(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, data):
        res = self.encoder(data)
        return res.view(data.size(0), -1)

