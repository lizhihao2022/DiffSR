import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .Basic import MLP


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, n_layers=1, act='gelu', res=True)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinSR(nn.Module):
    """
    Swin Transformer for Super-Resolution
    
    Input: [B, H, W, C]
    Output: [B, H*scale_h, W*scale_w, out_channels]
    """
    def __init__(self, model_params):  # 【修改1】统一使用 model_params
        super().__init__()
        """
        Args:
            model_params: dict containing:
                - in_channels: int, number of input channels, default: 3
                - out_channels: int, number of output channels, default: 1
                - n_hidden: int, hidden dimension, default: 96
                - n_layers: int, number of Swin layers, default: 3
                - depth: int, number of blocks per layer, default: 2
                - num_heads: int, number of attention heads, default: 4
                - window_size: int, window size for attention, default: 4
                - mlp_ratio: float, MLP expansion ratio, default: 4.0
                - qkv_bias: bool, whether to use bias in QKV projection, default: True
                - qk_scale: float or None, default: None
                - drop_rate: float, dropout rate, default: 0.0
                - attn_drop_rate: float, attention dropout rate, default: 0.0
                - drop_path_rate: float, stochastic depth rate, default: 0.1
                - act: str, activation function {gelu, relu, tanh}, default: 'gelu'
                - upsample_factor: list of int or int, upsampling scale, default: [2, 2]
        """
        self.__name__ = 'SwinSR'
        
        # 【修改2】从 model_params 读取所有参数
        self.in_channels = model_params.get('in_channels', 3)
        self.out_channels = model_params.get('out_channels', 1)
        self.n_hidden = model_params.get('n_hidden', 96)
        n_layers = model_params.get('n_layers', 3)
        depth = model_params.get('depth', 2)
        num_heads = model_params.get('num_heads', 4)
        self.window_size = model_params.get('window_size', 4)
        mlp_ratio = model_params.get('mlp_ratio', 4.0)
        qkv_bias = model_params.get('qkv_bias', True)
        qk_scale = model_params.get('qk_scale', None)
        drop_rate = model_params.get('drop_rate', 0.0)
        attn_drop_rate = model_params.get('attn_drop_rate', 0.0)
        drop_path_rate = model_params.get('drop_path_rate', 0.1)
        act = model_params.get('act', 'gelu')
        
        # 【修改3】上采样倍数支持动态配置
        upsample_factor = model_params.get('upsample_factor', [2, 2])
        if isinstance(upsample_factor, int):
            upsample_factor = [upsample_factor, upsample_factor]
        self.upsample_factor = upsample_factor
        
        norm_layer = nn.LayerNorm

        # 【修改4】移除固定的 input_resolution 和 output_resolution
        # 这些将在 forward 中动态计算

        self.preprocess = MLP(
            1,
            self.n_hidden * 2,
            self.n_hidden,
            n_layers=0,
            res=False,
            act=act
        )

        self.placeholder = nn.Parameter(torch.rand(self.n_hidden) / self.n_hidden)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers * depth)]

        # 【修改5】layers 将在 forward 中根据输入尺寸动态创建
        # 这里保存配置以便动态创建
        self.n_layers = n_layers
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.dpr = dpr
        self.norm_layer = norm_layer
        
        # 缓存层以避免重复创建
        self.cached_layers = None
        self.cached_input_resolution = None

        self.norm = norm_layer(self.n_hidden)
        
        # 【修改6】上采样支持任意倍数
        scale_h, scale_w = self.upsample_factor
        if scale_h == scale_w == 2:
            # 亚像素卷积（Pixel Shuffle）for 2x
            self.upsample = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden * 4),
                nn.GELU(),
            )
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale_h == scale_w:
            # 通用上采样
            self.upsample = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden * scale_h * scale_w),
                nn.GELU(),
            )
            self.pixel_shuffle = nn.PixelShuffle(scale_h)
        else:
            # 非对称上采样，使用插值
            self.upsample = None
            self.pixel_shuffle = None

        self.output_proj = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden // 2),
            nn.GELU(),
            nn.Linear(self.n_hidden // 2, self.out_channels)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _build_layers(self, input_resolution):
        """【修改7】根据输入分辨率动态构建 Swin Transformer 层"""
        layers = nn.ModuleList([
            BasicLayer(
                dim=self.n_hidden,
                input_resolution=input_resolution,
                depth=self.depth,
                num_heads=self.num_heads,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i*self.depth:(i+1)*self.depth],
                norm_layer=self.norm_layer
            )
            for i in range(self.n_layers)
        ])
        return layers
            
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] 其中 C 只包含物理场
        Returns:
            out: [B, H*scale_h, W*scale_w, out_channels]
        """
        # 【修改8】动态获取输入分辨率
        B, H, W, C = x.shape
        
        assert C == self.in_channels, f"Input channels mismatch: {C} vs {self.in_channels}"

        fx = x  # [B, H, W, 1] 取最后一个通道作为物理场
        fx = fx.reshape(B, H * W, 1)  # [B, N, 1]

        fx = self.preprocess(fx)  # [B, N, n_hidden]
        fx = fx + self.placeholder[None, None, :]
        
        input_resolution = (H, W)
        if self.cached_layers is None or self.cached_input_resolution != input_resolution:
            self.cached_layers = self._build_layers(input_resolution).to(x.device)
            self.cached_input_resolution = input_resolution

        for layer in self.cached_layers:
            fx = layer(fx)  # [B, N, n_hidden]

        fx = self.norm(fx)
        
        # 【修改10】动态上采样处理
        if self.upsample is not None and self.pixel_shuffle is not None:
            # 使用 Pixel Shuffle
            fx = self.upsample(fx)  # [B, N, n_hidden*scale^2]
            fx = fx.view(B, H, W, -1)  # [B, H, W, n_hidden*scale^2]
            fx = fx.permute(0, 3, 1, 2)  # [B, n_hidden*scale^2, H, W]
            fx = self.pixel_shuffle(fx)  # [B, n_hidden, scale*H, scale*W]
            fx = fx.permute(0, 2, 3, 1)  # [B, scale*H, scale*W, n_hidden]
        else:
            # 使用插值上采样（非对称情况）
            H_out = H * self.upsample_factor[0]
            W_out = W * self.upsample_factor[1]
            fx = fx.view(B, H, W, -1)  # [B, H, W, n_hidden]
            fx = fx.permute(0, 3, 1, 2)  # [B, n_hidden, H, W]
            fx = F.interpolate(fx, size=(H_out, W_out), mode='bilinear', align_corners=False)
            fx = fx.permute(0, 2, 3, 1)  # [B, H_out, W_out, n_hidden]
        
        H_out, W_out = fx.shape[1], fx.shape[2]
        fx = fx.reshape(B, H_out * W_out, -1)  # [B, N_out, n_hidden]

        out = self.output_proj(fx)  # [B, N_out, out_channels]
        out = out.view(B, H_out, W_out, self.out_channels)

        return out