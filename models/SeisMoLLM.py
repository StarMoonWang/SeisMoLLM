from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo.config
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import transformers.models.gpt2 as GPT2
from peft import get_peft_model, LoraConfig
from functools import partial
from ._factory import register_model

torch._dynamo.config.cache_size_limit = 1024


def lora_setting(target_modules, r=16, lora_alpha=16, lora_dropout=0.1, bias="lora_only"):
    return LoraConfig(target_modules=target_modules, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias)

GPT2_lora = lora_setting("all-linear")
GPT_file_path = '/ailab/user/wangxinghao/project/EQLLM/LLM/'  # the path of pre-trained model files


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.
    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.
    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.

    Args:
        x (torch.Tensor): N-dimensional tensor.
        input (Tensor): N-dimensional tensor
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
        dim (int): Dimension to pad.
        padding_value (float): fill value.

    Raises: AssertionError: `kernel_size` is less than `stride`.

    Returns: torch.Tensor : padded tensor.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )
        
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, 
                              stride=stride, bias=False)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Multi_Scale_Conv_Block(nn.Module):
    def __init__(
        self, scale_num, scale_stride, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num)
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=scale_num * out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list()
        for conv in self.convs:
            xi = conv(x)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class LLM_Block(nn.Module):
    def __init__(self, start_layer, end_layer, patch_size, lora_config, pretrain=True, freeze=True):
        super(LLM_Block, self).__init__()
        
        self.pretrain = pretrain
        self.freeze = freeze
        self.lora_config = lora_config
        self.patch_size = patch_size

        if pretrain:
            self.llm = GPT2.GPT2Model.from_pretrained(
                GPT_file_path+'GPT2', output_hidden_states=True, 
                vocab_size=0, ignore_mismatched_sizes=True
            )  # loads a pretrained GPT-2 small base model
        else:
            print("------------------no pretrain------------------")
            self.llm = GPT2.GPT2Model(GPT2.configuration_gpt2.GPT2Config(vocab_size=0))
        self.llm.h = self.llm.h[start_layer : end_layer]
        
        # print("LLM blocks = {}".format(self.llm))
        # print(f"using LLM layers: {end_layer - start_layer}")

        if self.freeze and self.pretrain:
            self.llm = get_peft_model(self.llm, self.lora_config)  # apply LoRA to finetune the base model
            for name, param in self.llm.named_parameters():
                if "ln" in name or "wpe" in name or "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # wte is "Word Token Embeddings", won't participate in producing loss
            for name, param in self.llm.named_parameters():
                if "wte" in name:  
                    param.requires_grad = False

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = rearrange(x, 'b c n p -> b n (c p)')
        x = self.llm(inputs_embeds = x).last_hidden_state
        x = rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size)
        return x


class HeadDetectionPicking(nn.Module):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels,
        layer_channels,  # dp_head_channels: [128, 160, 192, 224]
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)

        self.depth = len(layer_channels)

        self.up_layers = nn.ModuleList()

        for inc, outc, kers in zip(
                [feature_channels] + layer_channels[:-1],
                layer_channels[:-1] + [out_channels * 2],
                layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        N, C, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadClassification(nn.Module):
    """Head of classification."""

    def __init__(self, feature_channels, num_classes, out_act_layer, **kwargs):
        super().__init__()
        
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , num_classes)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadRegression(nn.Module):
    """Head of regression."""

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        # x : [b c (n p)]
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x
    
class HeadBAZ(nn.Module):
    """Head of Back-Azimuth Estimation."""

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , 2)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        # x : [b c (n p)]
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x[:, :1], x[:, 1:]


class SeisMoLLM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=[8, 6, 4, 2],
        conv_channels=[16, 48, 96],
        conv_kernel_sizes=[16, 8, 6, 1],
        conv_strides=[2, 2, 2, 1],
        llm_layers = 3,
        d_model=768,
        patch_size=8,
        dp_head_channels = [128, 160, 192, 224],
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        output_head=HeadRegression,
        **kwargs
    ):
        super().__init__()

        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels.append(d_model // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]

        # Multi-Scale Convolutional Embedder
        self.convs = nn.Sequential(
            *[
                Multi_Scale_Conv_Block(
                    scale_num=conv_scale_num,
                    scale_stride=ss,
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for ss, inc, outc, kers, strd in zip(
                    conv_scale_strides,
                    [in_channels] + conv_channels[:-1],
                    conv_channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )

        # Pre-trained LLM Blocks
        self.llm_blocks = LLM_Block(
            start_layer=0,
            end_layer=llm_layers,
            patch_size=patch_size,
            lora_config=GPT2_lora
        )
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=12,
        #                                        dropout=0.1, batch_first=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=12, dim_feedforward=d_model * 4, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel)
                out_layer_kernel_sizes.insert(0, kernel)

            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias
            )

        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x):
        x_input = x
        # Multi Scale Conv Embedder
        x = self.convs(x)
        # LLM Blocks
        x = self.llm_blocks(x)
        
        '''
        for ablations of changing LLM to attention layer or Transformer layer

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = rearrange(x, 'b c n p -> b n (c p)')

        x, _ = self.attention(x, x, x)
        x = self.transformer(x)

        x = rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size)
        '''

        # Output head
        x = self.out_head(x, x_input)
        return x


@register_model
def SeisMoLLM_dpk(**kwargs):
    """Detection and Phase-Picking."""
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_pmp(**kwargs):
    """P-motion-polarity classification."""
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_emg(**kwargs):
    """Magnitude estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_baz(**kwargs):
    """Azimuth estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadBAZ,
            out_act_layer=partial(
                nn.Tanh
            ),
        ),
        **kwargs,
    )
    return model

@register_model
def SeisMoLLM_dis(**kwargs):
    """Epicentral distance estimation."""
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        **kwargs,
    )
    return model
