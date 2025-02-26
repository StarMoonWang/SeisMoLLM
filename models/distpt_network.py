import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
dist-PT network.

Reference:
    [1] S. M. Mousavi and G. C. Beroza. (2020) 
        Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations.
        IEEE Transactions on Geoscience and Remote Sensing, 58, 11, 8211-8224.
        doi: 10.1109/TGRS.2020.2988770.
"""


def _causal_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    dilation: int,
    dim: int = -1,
    padding_value: float = 0.0,
):
    """
    Auto pad for causal conv layer.
    """
    assert stride == 1

    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (kernel_size -1) * dilation 
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds, 0)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, drop_rate):
        super().__init__()
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn0 = nn.BatchNorm1d(out_channels)

        self.relu0 = nn.ReLU()

        self.dropout0 = nn.Dropout1d(drop_rate)

        self.conv1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout1d(drop_rate)

        self.conv_out = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = _causal_pad_1d(
            x, self.conv0.kernel_size[0], self.conv0.stride[0], self.conv0.dilation[0]
        )
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.dropout0(x)

        x = _causal_pad_1d(
            x, self.conv1.kernel_size[0], self.conv1.stride[0], self.conv1.dilation[0]
        )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x1 = x + self.conv_out(x)

        return x1, x


class TemporalConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_size: int = 2,
        num_conv_blocks: int = 1,
        dilations: list = [1, 2, 4, 8, 16, 32],
        drop_rate: float = 0.0,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        self.conv_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    drop_rate=drop_rate,
                )
                for dilation in dilations * num_conv_blocks
            ]
        )

        self.return_sequences = return_sequences

    def forward(self, x):
        x = self.conv_in(x)

        shortcuts = []
        for conv in self.conv_blocks:
            x, sc = conv(x)
            shortcuts.append(sc)

        x = sum(shortcuts)

        if not self.return_sequences:
            x = x[:, :, -1]

        return x


class DistPT_Network(nn.Module):
    """
    dist-PT network
    """

    def __init__(
        self,
        in_channels: int,
        tcn_channels: int=20,
        kernel_size: int=6,
        num_conv_blocks: int=1,
        dilations: list=[2**i for i in range(11)],
        drop_rate: float=0.1,
        **kwargs
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            tcn_channels (int): Number of TCN channels.
            kernel_size (int): Convolution kernel size.
            num_conv_blocks (int): Number of convolution blocks in each TCN layer.
            dilations (list): dilation list.
            drop_rate (float): dropout rate.
        """
        super().__init__()

        self.tcn = TemporalConvLayer(
            in_channels=in_channels,
            out_channels=tcn_channels,
            kernel_size=kernel_size,
            num_conv_blocks=num_conv_blocks,
            dilations=dilations,
            drop_rate=drop_rate,
        )

        self.lin_dist = nn.Linear(in_features=tcn_channels, out_features=2)
        self.lin_ptrvl = nn.Linear(in_features=tcn_channels, out_features=2)

    def forward(self, x):
        x = self.tcn(x)

        do = self.lin_dist(x)
        po = self.lin_ptrvl(x)

        return do, po


@register_model
def distpt_network(**kwargs):
    model = DistPT_Network(**kwargs)
    return model

