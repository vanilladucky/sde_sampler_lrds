# Unet implementation specific to MNIST 14x14
# Based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py

# Libraries
import torch
from ..models.mlp import TimeEmbed
from ..models.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_


def init_at_zero(m):
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        kaiming_uniform_zeros_(m.weight)
        init_bias_uniform_zeros(m.bias, weight=m.weight)


class AttentionBlock(torch.nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = torch.nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = torch.nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = torch.nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class ResidualBlock(torch.nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 16, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = torch.nn.GroupNorm(n_groups, in_channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = torch.nn.GroupNorm(n_groups, out_channels)
        self.act2 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = torch.nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = torch.nn.Linear(time_channels, out_channels)
        self.time_act = torch.nn.SiLU()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(torch.nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool = False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(torch.nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool = False):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(torch.nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(torch.nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(torch.nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Unet(torch.nn.Module):

    def __init__(self, n_channels, image_channels=1, rev_proj_channels=None, init_last_layer_with_zeros=False):
        # Call the parent constructor
        super().__init__()
        # Project image into feature map
        self.image_proj = torch.nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        # Make the time embedding
        time_channels = n_channels * 4
        self.time_emb = TimeEmbed(
            dim_out=time_channels,
            activation=torch.nn.SiLU(),
            num_layers=2,
            channels=n_channels,
        )
        # Make the down part
        self.down_block1 = DownBlock(n_channels, n_channels, time_channels)
        self.down_sample1 = Downsample(n_channels)
        self.down_block2 = DownBlock(n_channels, 2 * n_channels, time_channels, has_attn=True)
        # Make the mid part
        self.mid_block1 = MiddleBlock(2 * n_channels, time_channels)
        # Make the up part
        self.up_block1 = UpBlock(2 * n_channels, 2 * n_channels, time_channels, has_attn=True)
        self.up_sample1 = Upsample(2 * n_channels)
        self.up_block2 = UpBlock(2 * n_channels, n_channels, time_channels)
        # Project back to the image map
        if rev_proj_channels is None:
            self.image_proj_rev = torch.nn.Sequential(
                torch.nn.GroupNorm(16, n_channels),
                torch.nn.SiLU(),
                torch.nn.Conv2d(n_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
            )
        else:
            channels_proj_rev = [n_channels] + list(rev_proj_channels) + [image_channels]
            layers_proj_rev = []
            for i in range(len(channels_proj_rev)-1):
                layers_proj_rev.append(torch.nn.GroupNorm(min(16, channels_proj_rev[i]), channels_proj_rev[i]))
                layers_proj_rev.append(torch.nn.SiLU())
                layers_proj_rev.append(torch.nn.Conv2d(
                    channels_proj_rev[i], channels_proj_rev[i+1], kernel_size=(3, 3), padding=(1, 1)))
            self.image_proj_rev = torch.nn.Sequential(*layers_proj_rev)
        # Initialize the last layer
        if init_last_layer_with_zeros:
            if rev_proj_channels is None:
                self.image_proj_rev.apply(init_at_zero)
            else:
                self.image_proj_rev[-1].apply(init_at_zero)

    def forward(self, t, x):
        # Reshape the image
        orig_shape = x.shape
        x = x.view((-1, 1, 14, 14))
        # Embed time and image
        t = self.time_emb(t)
        x = self.image_proj(x)  # (n_channels, 14, 14)
        # Run the down part
        x = self.down_block1(x, t)  # (n_channels, 14, 14)
        x1 = x.clone()
        x = self.down_sample1(x, t)  # (n_channels, 7, 7)
        x = self.down_block2(x, t)  # (2 * n_channels, 7, 7)
        x2 = x.clone()
        # Run the mid part
        x = self.mid_block1(x, t)  # (2 * n_channels, 7, 7)
        # Run the up part
        x = self.up_block1(torch.cat((x, x2), dim=1), t)  # (2 * n_channels, 7, 7)
        # Run the up part
        x = self.up_sample1(x, t)  # (2 * n_channels, 14, 14)
        x = self.up_block2(torch.cat((x, x1), dim=1), t)  # (n_channels, 14, 14)
        # Project back
        return self.image_proj_rev(x).view(orig_shape)  # (1, 14, 14)

    def load_state_dict_except_proj_rev(self, d):
        self.load_state_dict({k: v for k, v in d.items() if 'image_proj_rev' not in k}, strict=False)

    def freeze_except_image_proj_rev(self):
        for name, param in self.named_parameters():
            if 'image_proj_rev' not in name:
                param.requires_grad = False
