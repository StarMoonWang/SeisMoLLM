import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
MagNet.

Reference:
    [1] Mousavi, S. M., & Beroza, G. C. (2020)
        A machine-learning approach for earthquake magnitude estimation. 
        Geophysical Research Letters, 47, e2019GL085976. 
        https://doi.org/10.1029/2019GL085976
"""

def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.
    """
    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, drop_rate
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
        )
        self.dropout = nn.Dropout(drop_rate)
        self.pool = nn.MaxPool1d(pool_kernel_size, ceil_mode=True)

    def forward(self, x):
        N, C, L = x.size()

        x = _auto_pad_1d(x,self.conv.kernel_size[0])

        x = self.conv(x)

        x = self.dropout(x)
        x = self.pool(x)

        return x


class MagNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list = [64, 32],
        lstm_dim: int = 100,
        drop_rate: float = 0.2,
        **kwargs
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            conv_channels (list, optional): Number of output channels of each convolution layer. Defaults to [64, 32].
            lstm_dim (int, optional): Hidden size of LSTM layer. Defaults to 100.
            drop_rate (float, optional): Dropout rate. Defaults to 0.2.
        """        
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    conv_kernel_size=3,
                    pool_kernel_size=4,
                    drop_rate=drop_rate,
                )
                for inc, outc in zip([in_channels] + conv_channels[:-1], conv_channels)
            ]
        )

        self.lstm = nn.LSTM(
            conv_channels[-1],
            lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lin = nn.Linear(in_features=lstm_dim * 2, out_features=2)

    def forward(self, x):
        x = self.conv_layers(x)
        hs, (h, c) = self.lstm(x.transpose(-1, -2))
        h = h.transpose(0, 1).flatten(1)
        out = self.lin(h)

        return out



@register_model
def magnet(**kwargs):
    model = MagNet(**kwargs)
    return model

