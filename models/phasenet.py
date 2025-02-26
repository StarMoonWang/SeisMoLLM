import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
PhaseNet.

Reference:
    [1] Zhu, W., & Beroza, G. C. (2019)
        PhaseNet: a deep-neural-network-based seismic arrival-time picking method.
        Geophysical Journal International, 216(1), 261-273.
        doi: 10.1093/gji/ggy423
"""


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride,drop_rate, has_stride_conv=True
    ):
        super().__init__()

        self.stride = stride if has_stride_conv else 1
        self.kernel_padding = kernel_size - stride if has_stride_conv else 0

        self.conv0 = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            )
            if has_stride_conv
            else nn.Identity()
        )
        self.bn0 = (
            nn.BatchNorm1d(num_features=in_channels)
            if has_stride_conv
            else nn.Identity()
        )
        self.relu0 = nn.ReLU() if has_stride_conv else nn.Identity()
        self.drop0 = nn.Dropout(drop_rate)  if has_stride_conv else nn.Identity()

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop_rate)

    def forward(self, x):
        p = (
            self.stride - (x.size(-1) % self.stride)
        ) % self.stride + self.kernel_padding
        stride_conv_padding = (
            p // 2,
            p - p // 2,
        )
        x = F.pad(x, stride_conv_padding, "constant", 0)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.drop0(x)

        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        return x


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        drop_rate,
        has_conv_same=True,
        has_conv_trans=True,
    ):
        super().__init__()

        self.conv_padding_same = (
            (
                (kernel_size - 1) // 2,
                kernel_size - 1 - (kernel_size - 1) // 2,
            )
            if has_conv_same
            else (0, 0)
        )
        self.conv0 = (
            nn.Conv1d(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                bias=False,
            )
            if has_conv_same
            else nn.Identity()
        )
        self.bn0 = nn.BatchNorm1d(num_features=in_channels) if has_conv_same else nn.Identity()
        self.relu0 = nn.ReLU() if has_conv_same else nn.Identity()
        self.drop0 = nn.Dropout(drop_rate)  if has_conv_trans else nn.Identity()

        self.convt = (
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            )
            if has_conv_trans
            else nn.Identity()
        )
        self.bn1 = (
            nn.BatchNorm1d(num_features=out_channels)
            if has_conv_trans
            else nn.Identity()
        )
        self.relu1 = nn.ReLU() if has_conv_trans else nn.Identity()
        self.drop1 = nn.Dropout(drop_rate)  if has_conv_same else nn.Identity()

    def forward(self, x):
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.drop0(x)

        x = self.convt(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        return x


class PhaseNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        kernel_size=7,
        stride=4,
        conv_channels=[8, 16, 32, 64, 128], 
        drop_rate=0.1,
        **kwargs
    ):
        """PhaseNet

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            kernel_size (int): Kernel size. Defaults to 7.
            stride (int): Stride. Defaults to 4.
            conv_channels (list): Number of output channels of each convolution layer. Defaults to [8, 16, 32, 64, 128].
            drop_rate (list): Dropout rate. Defauts to 0.1.
        """        
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_channels = conv_channels
        self.depth = len(conv_channels)

        # Input
        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.conv_in = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.conv_channels[0],
            kernel_size=self.kernel_size,
        )
        self.bn_in = nn.BatchNorm1d(num_features=self.conv_channels[0])
        self.relu_in = nn.ReLU()
        self.drop_in = nn.Dropout(drop_rate)

        # Down sampling
        self.down_convs = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    drop_rate=drop_rate,
                    has_stride_conv=(i != 0),
                )
                for i, inc, outc in zip(
                    range(self.depth),
                    self.conv_channels[:1] + self.conv_channels[:-1],
                    self.conv_channels,
                )
            ]
        )

        # Up sampling
        self.up_convs = nn.ModuleList(
            [
                ConvTransBlock(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    drop_rate=drop_rate,
                    has_conv_same=(i < self.depth - 1),
                    has_conv_trans=(i > 0),
                )
                for i, inc, outc in zip(
                    range(self.depth)[::-1],
                    self.conv_channels[::-1],
                    self.conv_channels[-2::-1] + [None],
                )
            ]
        )

        # Output
        self.conv_out = nn.Conv1d(self.conv_channels[0], 3, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.drop_in(x)

        shortcuts = []
        for conv in self.down_convs[:-1]:
            x = conv(x)
            shortcuts.append(x)

        x = self.down_convs[-1](x)

        for convt, shortcut in zip(self.up_convs[:-1], shortcuts[::-1]):
            x = convt(x)
            p = (
                (self.stride - (shortcut.size(-1) % self.stride)) % self.stride
                + self.kernel_size
                - self.stride
            )
            lp = p // 2
            rp = p - lp
            x = torch.cat([shortcut, x[:, :, lp:-rp]], dim=1)

        x = self.up_convs[-1](x)

        x = self.conv_out(x)
        x = self.softmax(x)

        return x




@register_model
def phasenet(**kwargs):
    model = PhaseNet(**kwargs)
    return model

