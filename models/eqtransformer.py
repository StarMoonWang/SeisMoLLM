import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

"""
EQTransformer. 

Reference:
    [1] Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L, Y., and Beroza, G, C.
        Earthquake transformer—an attentive deep- learning model for simultaneous
        earthquake detection and phase picking. Nat Commun 11, 3952 (2020).
        doi: 10.1038/s41467-020-17591-w
"""


class ConvBlock(nn.Module):
    """Convolution block
    Input shape: (N,C,L)
    """

    _epsilon = 1e-6

    def __init__(
        self, in_channels, out_channels, kernel_size, kernel_l1_alpha, bias_l1_alpha
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, padding=0)

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)
        x = F.pad(x, (0, x.size(-1) % 2), "constant", -1 / self._epsilon)
        x = self.pool(x)
        return x


class ResConvBlock(nn.Module):
    """Residual convolution block
    Input shape: (N,C,L)
    """

    def __init__(self, io_channels, kernel_size, drop_rate):
        super().__init__()

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.bn0 = nn.BatchNorm1d(num_features=io_channels)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout1d(p=drop_rate)
        self.conv0 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

        self.bn1 = nn.BatchNorm1d(io_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(p=drop_rate)
        self.conv1 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        x1 = self.bn0(x)
        x1 = self.relu0(x1)
        x1 = self.dropout0(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv0(x1)

        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv1(x1)
        out = x + x1
        return out


class BiLSTMBlock(nn.Module):
    """Bi-LSTM block
    Input shape: (N,C,L)
    """

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=drop_rate)
        self.conv = nn.Conv1d(
            in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        return x


class AttentionLayer(nn.Module):
    """Single head attention
    Input shape: (N,C,L)
    """

    _epsilon = 1e-6

    def __init__(self, in_channels, d_model, attn_width=None):
        super().__init__()
        self.attn_width = attn_width
        self.Wx = nn.Parameter(torch.empty((in_channels, d_model)))
        self.Wt = nn.Parameter(torch.empty((in_channels, d_model)))
        self.bh = nn.Parameter(torch.empty(d_model))
        self.Wa = nn.Parameter(torch.empty((d_model, 1)))
        self.ba = nn.Parameter(torch.empty(1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.xavier_uniform_(self.Wa)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.ba)

    def forward(self, x):
        # (N,C,L) -> (N,L,C)
        x = x.permute(0, 2, 1)

        # (N,L,C),(C,d) -> (N,L,1,d)
        q = torch.matmul(x, self.Wt).unsqueeze(2)

        # (N,L,C),(C,d) -> (N,1,L,d)
        k = torch.matmul(x, self.Wx).unsqueeze(1)

        # (N,L,1,d),(N,1,L,d),(d,) -> (N,L,L,d)
        h = torch.tanh(q + k + self.bh)

        # (N,L,d),(d,1) -> (N,L,L,1) -> (N,L,L)
        e = (torch.matmul(h, self.Wa) + self.ba).squeeze(-1)

        # (N,L,L)
        e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)

        # Masked attention
        if self.attn_width is not None:
            mask = (
                torch.ones(e.shape[-2:], dtype=torch.bool, device=e.device)
                .tril(self.attn_width // 2 - 1)
                .triu(-self.attn_width // 2)
            )
            e = e.where(mask, 0)

        # (N,L,L)
        s = torch.sum(e, dim=-1, keepdim=True)
        a = e / (s + self._epsilon)

        # (N,L,L),(N,L,C) -> (N,L,C)
        v = torch.matmul(a, x)

        # (N,L,C) -> (N,C,L)
        v = v.permute(0, 2, 1)

        return v, a


class FeedForward(nn.Module):
    """MLP
    Input shape: (N,L,C)
    """

    def __init__(self, io_channels, feedforward_dim, drop_rate):
        super().__init__()

        self.lin0 = nn.Linear(in_features=io_channels, out_features=feedforward_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.lin1 = nn.Linear(in_features=feedforward_dim, out_features=io_channels)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin0.weight)
        nn.init.zeros_(self.lin0.bias)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin1(x)

        return x


class TransformerLayer(nn.Module):
    """Transformer Layer
    Input shape: (N,C,L)
    """

    def __init__(
        self, io_channels, d_model, feedforward_dim, drop_rate, attn_width=None
    ):
        super().__init__()

        self.attn = AttentionLayer(
            in_channels=io_channels, d_model=d_model, attn_width=attn_width
        )
        self.ln0 = nn.LayerNorm(normalized_shape=io_channels)

        self.ff = FeedForward(
            io_channels=io_channels,
            feedforward_dim=feedforward_dim,
            drop_rate=drop_rate,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=io_channels)

    def forward(self, x):
        x1, w = self.attn(x)
        x2 = x1 + x
        # (N,C,L) -> (N,L,C)
        x2 = x2.permute(0, 2, 1)
        x2 = self.ln0(x2)
        x3 = self.ff(x2)
        x4 = x3 + x2
        x4 = self.ln1(x4)
        # (N,L,C) -> (N,C,L)
        x4 = x4.permute(0, 2, 1)

        return x4, w


class Encoder(nn.Module):
    """
    Encoder layers:
        Conv * 7
        ResConv * 5
        BiLSTM * 3
        TransformerLayer * 2

    Note:
        L1 regularization was only applied to the convolution blocks of the first stage.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: list,
        conv_kernels: list,
        resconv_kernels: list,
        num_lstm_blocks: int,
        num_transformer_layers: int,
        transformer_io_channels: int,
        transformer_d_model: int,
        feedforward_dim: int,
        drop_rate: float,
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
    ):
        super().__init__()

        # Conv 1D
        self.convs = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, kers in zip(
                    [in_channels] + conv_channels[:-1], conv_channels, conv_kernels
                )
            ]
        )

        # Res CNN
        self.res_convs = nn.Sequential(
            *[
                ResConvBlock(
                    io_channels=conv_channels[-1], kernel_size=kers, drop_rate=drop_rate
                )
                for kers in resconv_kernels
            ]
        )

        # Bi-LSTM
        self.bilstms = nn.Sequential(
            *[
                BiLSTMBlock(in_channels=inc, out_channels=outc, drop_rate=drop_rate)
                for inc, outc in zip(
                    [conv_channels[-1]]
                    + [transformer_io_channels] * (num_lstm_blocks - 1),
                    [transformer_io_channels] * num_lstm_blocks,
                )
            ]
        )

        # Transformer
        self.transformers = nn.ModuleList(
            [
                TransformerLayer(
                    io_channels=transformer_io_channels,
                    d_model=transformer_d_model,
                    feedforward_dim=feedforward_dim,
                    drop_rate=drop_rate,
                )
                for _ in range(num_transformer_layers)
            ]
        )

    def forward(self, x):
        x = self.convs(x)

        x = self.res_convs(x)

        x = self.bilstms(x)

        for transformer_ in self.transformers:
            x, w = transformer_(x)

        return x, w


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_samples,
        kernel_size,
        kernel_l1_alpha,
        bias_l1_alpha,
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.out_samples = out_samples

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = self.upsampling(x)
        x = x[:, :, : self.out_samples]
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)

        return x


class IdentityNTuple(nn.Identity):
    def __init__(self, *args, ntuple: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ntuple >= 1
        self.ntuple = ntuple

    def forward(self, input: torch.Tensor):
        if self.ntuple > 1:
            return (super().forward(input),) * self.ntuple
        else:
            return super().forward(input)


class Decoder(nn.Module):
    """
    Decoder layers:
        LSTM * 1 (opt.)
        TransormerLayer *1 (opt.)
        UpSampling * 7
        Conv * 1
    """

    def __init__(
        self,
        conv_channels: list,
        conv_kernels: list,
        transformer_io_channels: int,
        transformer_d_model: int,
        feedforward_dim: int,
        drop_rate: float,
        out_samples,
        has_lstm: bool = True,
        has_local_attn: bool = True,
        local_attn_width: int = 3,
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
    ):
        super().__init__()

        self.lstm = (
            nn.LSTM(
                input_size=transformer_io_channels,
                hidden_size=transformer_io_channels,
                batch_first=True,
                bidirectional=False,
            )
            if has_lstm
            else IdentityNTuple(ntuple=2)
        )

        self.lstm_dropout = nn.Dropout(p=drop_rate) if has_lstm else nn.Identity()

        self.transformer = (
            TransformerLayer(
                io_channels=transformer_io_channels,
                d_model=transformer_d_model,
                feedforward_dim=feedforward_dim,
                drop_rate=drop_rate,
                attn_width=local_attn_width,
            )
            if has_local_attn
            else IdentityNTuple(ntuple=2)
        )

        crop_sizes = [out_samples]
        for _ in range(len(conv_kernels) - 1):
            crop_sizes.insert(0, math.ceil(crop_sizes[0] / 2))

        self.upsamplings = nn.Sequential(
            *[
                UpSamplingBlock(
                    in_channels=inc,
                    out_channels=outc,
                    out_samples=crop,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, crop, kers in zip(
                    [transformer_io_channels] + conv_channels[:-1],
                    conv_channels,
                    crop_sizes,
                    conv_kernels,
                )
            ]
        )

        self.conv_out = nn.Conv1d(
            in_channels=conv_channels[-1], out_channels=1, kernel_size=11, padding=5
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)

        x, _ = self.transformer(x)

        x = self.upsamplings(x)

        x = self.conv_out(x)

        x = x.sigmoid()

        return x


class EQTransformer(nn.Module):
    _epsilon = 1e-6

    def __init__(
        self,
        in_channels=3,
        in_samples=8192,
        conv_channels=[8, 16, 16, 32, 32, 64, 64],
        conv_kernels=[11, 9, 7, 7, 5, 5, 3],
        resconv_kernels=[3, 3, 3, 2, 2],
        num_lstm_blocks=3,
        num_transformer_layers=2,
        transformer_io_channels=16,
        transformer_d_model=32,
        feedforward_dim=128,
        local_attention_width=3,
        drop_rate=0.1,
        decoder_with_attn_lstm=[False, True, True],
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
        **kwargs
    ):
        """
        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            in_samples (int): Number of input samples. Defaults to 8192.
            conv_channels (list): Number of output channels of each convolution layer. Defaults to [8, 16, 16, 32, 32, 64, 64].
            conv_kernels (list): Kernel sizes of each convolution layer. Defaults to [11, 9, 7, 7, 5, 5, 3].
            resconv_kernels (list): Kernel sizes of each residual convolution layer. Defaults to [3, 3, 3, 2, 2].
            num_lstm_blocks (int): Number of LSTM blocks in encoder. Defaults to 3.
            num_transformer_layers (int): Number of transformer layers in encoder. Defaults to 2.
            transformer_io_channels (int): Number of input and output channels in transformer layers. Defaults to 16.
            transformer_d_model (int): Number of features in transformer layers. Defaults to 32.
            feedforward_dim (int): Dimension of the feedforward network. Defaults to 128.
            local_attention_width (int): Attention width in local attention layer. Defaults to 3.
            drop_rate (float): Dropout rate. Defaults to 0.1.
            decoder_with_attn_lstm (tuple): Whether decoder branch has attention layer and lstm layer. Defaults to [False, True, True].
            conv_kernel_l1_regularization (float): Alpha of l1 regularization. Defaults to 0.0.
            conv_bias_l1_regularization (float): Alpha of l1 regularization. Defaults to 0.0.
        """        

        super().__init__()

        assert len(conv_channels) == len(conv_kernels)

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.resconv_kernels = resconv_kernels
        self.num_lstm_blocks = num_lstm_blocks
        self.num_transformer_layers = num_transformer_layers
        self.transformer_io_channels = transformer_io_channels
        self.transformer_d_model = transformer_d_model
        self.feedforward_dim = feedforward_dim
        self.decoder_with_attn_lstm = decoder_with_attn_lstm

        self.encoder = Encoder(
            in_channels=self.in_channels,
            conv_channels=self.conv_channels,
            conv_kernels=self.conv_kernels,
            resconv_kernels=self.resconv_kernels,
            num_lstm_blocks=self.num_lstm_blocks,
            num_transformer_layers=self.num_transformer_layers,
            transformer_io_channels=self.transformer_io_channels,
            transformer_d_model=self.transformer_d_model,
            feedforward_dim=self.feedforward_dim,
            drop_rate=self.drop_rate,
            conv_kernel_l1_regularization=conv_kernel_l1_regularization,
            conv_bias_l1_regularization=conv_bias_l1_regularization,
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    conv_channels=self.conv_channels[::-1],
                    conv_kernels=self.conv_kernels[::-1],
                    transformer_io_channels=self.transformer_io_channels,
                    transformer_d_model=self.transformer_d_model,
                    feedforward_dim=self.feedforward_dim,
                    drop_rate=self.drop_rate,
                    out_samples=self.in_samples,
                    has_lstm=has_attn_lstm,
                    has_local_attn=has_attn_lstm,
                    local_attn_width=local_attention_width,
                    conv_kernel_l1_regularization=conv_kernel_l1_regularization,
                    conv_bias_l1_regularization=conv_bias_l1_regularization,
                )
                for has_attn_lstm in self.decoder_with_attn_lstm
            ]
        )

    def forward(self, x):
        feature, _ = self.encoder(x)

        outputs = [decoder(feature) for decoder in self.decoders]

        return torch.cat(outputs, dim=1)



@register_model
def eqtransformer(**kwargs):
    model = EQTransformer(**kwargs)
    return model
