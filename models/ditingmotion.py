import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
DiTingMotion.

Reference:
    [1] Zhao M, Xiao Z, Zhang M, Yang Y, Tang L and Chen S.
        DiTingMotion: A deep-learning first-motion-polarity classifier 
        and its application to focal mechanism inversion.(2023)
        Front. Earth Sci. 11:1103914. 
        doi: 10.3389/feart.2023.1103914
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


class CombConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_sizes, out_kernel_size, drop_rate
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kers,
                    ),
                    nn.ReLU(),
                )
                for kers in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(drop_rate)
        self.out_conv = nn.Conv1d(
            in_channels=(in_channels + len(kernel_sizes) * out_channels),
            out_channels=out_channels,
            kernel_size=out_kernel_size,
        )
        self.out_relu = nn.ReLU()

    def forward(self, x):
        
        outs = [x]
        for conv_relu in self.convs:
            xi = _auto_pad_1d(x, conv_relu[0].kernel_size[0])
            xi = conv_relu(xi)
            outs.append(xi)

        x = torch.cat(outs, dim=1)
        x = self.dropout(x)
        x = _auto_pad_1d(x, self.out_conv.kernel_size[0])
        x = self.out_conv(x)
        x = self.out_relu(x)

        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layer_channels: list,
        comb_kernel_sizes,
        comb_out_kernel_size,
        drop_rate,
        pool_size,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                CombConvLayer(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_sizes=comb_kernel_sizes,
                    out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                )
                for inc, outc in zip(
                    [in_channels] + layer_channels[:-1], layer_channels
                )
            ]
        )
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x1 = self.conv_layers(x)
        x1 = torch.cat([x, x1], dim=1)
        x1 = self.pool(x1)

        return x1


class SideLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: list,
        comb_kernel_sizes,
        comb_out_kernel_size,
        drop_rate,
        linear_in_dim,
        linear_hidden_dim,
        linear_out_dim,
    ):
        super().__init__()

        self.conv_layer = CombConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=comb_kernel_sizes,
            out_kernel_size=comb_out_kernel_size,
            drop_rate=drop_rate,
        )

        self.flatten = nn.Flatten(1)

        self.lin0 = nn.Linear(in_features=linear_in_dim, out_features=linear_hidden_dim)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(
            in_features=linear_hidden_dim, out_features=linear_out_dim
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_out_channels = out_channels
        self.linear_in_dim = linear_in_dim

    def forward(self, x):
        x = self.conv_layer(x)
        N, C, L = x.size()

        if C * L != self.linear_in_dim:
            # The input shape of the official DiTingMotion-model is fixed to (2, 128).
            # In order to accommodate different shapes of input, interpolation is used here.
            tartget_size = self.linear_in_dim // self.conv_out_channels
            x = F.interpolate(x, tartget_size)

        x1 = self.flatten(x)

        x2 = self.lin0(x1)
        x2 = self.relu(x2)

        x3 = self.lin1(x2)
        x3 = self.sigmoid(x3)

        return x1, x2, x3


class DiTingMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        blocks_layer_channels:list=[
            [8, 8],
            [8, 8],
            [8, 8, 8],
            [8, 8, 8],
            [8, 8, 8],
        ],
        side_layer_conv_channels:int=2,
        blocks_sidelayer_linear_in_dims:list=[None, None, 32, 16, 16],
        blocks_sidelayer_linear_hidden_dims:list=[None, None, 8, 8, 8],
        comb_kernel_sizes:list=[3, 3, 5, 5],
        comb_out_kernel_size:int=3,
        pool_size:int=2,
        drop_rate: float = 0.2,
        fuse_hidden_dim:int=8,
        num_polarity_classes:int=2,
        num_clarity_classes:int=2,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            blocks_layer_channels (list): Layer channels of each block. Defaults to [ [8, 8], [8, 8], [8, 8, 8], [8, 8, 8], [8, 8, 8] ].
            side_layer_conv_channels (int): Number of output channels of `conv` in the side layers. Defaults to 2.
            blocks_sidelayer_linear_in_dims (list): Input dimension of the `linear` in each side layer of each block. Defaults to [ None, None, 32, 16, 16 ].
            blocks_sidelayer_linear_hidden_dims (list): Hidden dimension of the `linear` in each side layer of each block. Defaults to [ None, None, 8, 8, 8 ].
            comb_kernel_sizes (list): Kernel sizes of `CombConvLayer`. Defaults to [3, 3, 5, 5].
            comb_out_kernel_size (int): Kernel sizes of the last `conv` in  `CombConvLayer`. Defaults to 3.
            pool_size (int): Kernel size of pool layer. Defaults to 2.
            drop_rate (float): Dropout rate. Defaults to 0.2.
            fuse_hidden_dim (int): Hidden dimension of fuse-layer. Defaults to 8.
            num_polarity_classes (int): Number of polarity classes. Defaults to 2.
            num_clarity_classes (int): Number of clarity classes. Defaults to 2.
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        self.clarity_side_layers = nn.ModuleList()
        self.polarity_side_layers = nn.ModuleList()

        # In channels of each block
        blocks_in_channels = [in_channels]
        for blc in blocks_layer_channels[:-1]:
            blocks_in_channels.append(blc[-1] + blocks_in_channels[-1])

        fuse_polarity_in_dim = fuse_clarity_in_dim = 0

        # Blocks and side layers
        for inc, layer_channels, side_lin_in_dim, side_lin_hidden_dim in zip(
            blocks_in_channels,
            blocks_layer_channels,
            blocks_sidelayer_linear_in_dims,
            blocks_sidelayer_linear_hidden_dims,
        ):
            # Block
            block = BasicBlock(
                in_channels=inc,
                layer_channels=layer_channels,
                comb_kernel_sizes=comb_kernel_sizes,
                comb_out_kernel_size=comb_out_kernel_size,
                drop_rate=drop_rate,
                pool_size=pool_size,
            )

            if side_lin_in_dim is not None:
                # Side layers
                clarity_side_layer = SideLayer(
                    in_channels=layer_channels[-1] + inc,
                    out_channels=side_layer_conv_channels,
                    comb_kernel_sizes=comb_kernel_sizes,
                    comb_out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                    linear_in_dim=side_lin_in_dim,
                    linear_hidden_dim=side_lin_hidden_dim,
                    linear_out_dim=num_clarity_classes,
                )

                polarity_side_layer = SideLayer(
                    in_channels=layer_channels[-1] + inc,
                    out_channels=side_layer_conv_channels,
                    comb_kernel_sizes=comb_kernel_sizes,
                    comb_out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                    linear_in_dim=side_lin_in_dim,
                    linear_hidden_dim=side_lin_hidden_dim,
                    linear_out_dim=num_polarity_classes,
                )

                fuse_clarity_in_dim += side_lin_in_dim
                fuse_polarity_in_dim += side_lin_hidden_dim

            else:
                clarity_side_layer = polarity_side_layer = None

            self.blocks.append(block)
            self.clarity_side_layers.append(clarity_side_layer)
            self.polarity_side_layers.append(polarity_side_layer)

        # Fuse
        self.fuse_polarity = nn.Sequential(
            *[
                nn.Linear(in_features=indim, out_features=outdim)
                for indim, outdim in zip(
                    [fuse_polarity_in_dim, fuse_hidden_dim],
                    [fuse_hidden_dim, num_polarity_classes],
                )
            ],
            nn.Sigmoid(),
        )

        self.fuse_clarity = nn.Sequential(
            *[
                nn.Linear(in_features=indim, out_features=outdim)
                for indim, outdim in zip(
                    [fuse_clarity_in_dim, fuse_hidden_dim],
                    [fuse_hidden_dim, num_clarity_classes],
                )
            ],
            nn.Sigmoid(),
        )

    def forward(self, x):
        clarity_to_fuse = list()
        polarity_to_fuse = list()

        clarity_outs = list()
        polarity_outs = list()

        for block, clarity_side_layer, polarity_side_layer in zip(
            self.blocks, self.clarity_side_layers, self.polarity_side_layers
        ):
            # Basic block
            x = block(x)

            # Side layer
            if clarity_side_layer is not None and polarity_side_layer is not None:
                c0, _, c2 = clarity_side_layer(x)
                clarity_to_fuse.append(c0)
                clarity_outs.append(c2)

                _, p1, p2 = polarity_side_layer(x)
                polarity_to_fuse.append(p1)
                polarity_outs.append(p2)

        # Fuse
        x = torch.cat(clarity_to_fuse, dim=-1)
        x = self.fuse_clarity(x)
        clarity_outs.append(x)

        x = torch.cat(polarity_to_fuse, dim=-1)
        x = self.fuse_polarity(x)
        polarity_outs.append(x)

        # Final output
        final_clarity = sum(clarity_outs) / len(clarity_outs)
        final_polarity = sum(polarity_outs) / len(polarity_outs)

        return final_clarity, final_polarity


@register_model
def ditingmotion(**kwargs):
    model = DiTingMotion(num_polarity_classes=2, num_clarity_classes=2, **kwargs)
    return model


