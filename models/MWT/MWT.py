import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers.Basic import MLP
from .layers.MWT_Layers import MWT_CZ2d

class MWT_SuperResolution(nn.Module):
    """
    Multi-Wavelet Transform for 2D Super-Resolution
    Input: [B, H, W, C] - Low-resolution field with optional positional encoding
    Output: [B, H*scale_h, W*scale_w, out_channels] - High-resolution field
    """
    def __init__(self, model_params):
        super(MWT_SuperResolution, self).__init__()
        """
        Args:
            model_params: dict containing:
                - in_channels: int, number of input channels, default: 1
                - out_channels: int, number of output channels, default: 1
                - hidden_dim: int, hidden dimension, default: 128
                - k: int, wavelet degree, default: 6
                - n_layers: int, number of MWT layers, default: 6
                - c: int, number of wavelet channels, default: 1
                - alpha: int, wavelet parameter, default: 2
                - L: int, wavelet parameter, default: 0
                - base: str, wavelet basis {legendre, chebyshev}, default: 'legendre'
                - act: str, activation function {gelu, relu, tanh}, default: 'gelu'
                - upsample_factor: list of int or int, upsampling scale, default: [2, 2]
        """
        # 从配置文件读取参数
        self.in_channels = model_params.get('in_channels', 1)
        self.out_channels = model_params.get('out_channels', 1)
        self.hidden_dim = model_params.get('hidden_dim', 128)
        self.k = model_params.get('k', 6)
        self.n_layers = model_params.get('n_layers', 6)
        self.c = model_params.get('c', 1)
        alpha = model_params.get('alpha', 2)
        L = model_params.get('L', 0)
        base = model_params.get('base', 'legendre')
        act = model_params.get('act', 'gelu')
        
        # 上采样倍数
        upsample_factor = model_params.get('upsample_factor', 2)
        if isinstance(upsample_factor, int):
            upsample_factor = [upsample_factor, upsample_factor]
        self.upsample_factor = upsample_factor
        
        # MWT 维度
        self.WMT_dim = self.c * self.k ** 2
        
        # Padding 和增强分辨率将在 forward 中动态计算
        
        # 预处理层
        self.preprocess = MLP(
            self.in_channels, 
            self.hidden_dim * 2, 
            self.WMT_dim, 
            n_layers=0,
            res=False,
            act=act
        )
        
        # MWT 谱层
        self.spectral_layers = nn.ModuleList([
            MWT_CZ2d(k=self.k, alpha=alpha, L=L, c=self.c, base=base) 
            for _ in range(self.n_layers)
        ])
        
        # 上采样模块
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_factor[0], mode='bilinear', align_corners=True),
            nn.Conv2d(self.WMT_dim, self.WMT_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.WMT_dim, self.WMT_dim, kernel_size=3, padding=1)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.WMT_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_channels)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, H, W, C] 低分辨率输入
            
        Returns:
            output: [B, H*scale_h, W*scale_w, out_channels] 高分辨率输出
        """
        B, H, W, C = x.shape
        
        # 动态验证输入通道
        if C != self.in_channels:
            # 自动截取或填充到 in_channels
            if C > self.in_channels:
                x = x[..., :self.in_channels]
            else:
                # 如果通道数不足,用零填充
                padding = torch.zeros(B, H, W, self.in_channels - C, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
        
        # 动态计算 padding (确保尺寸是 2 的幂次)
        target_h = 2 ** (math.ceil(np.log2(H)))
        target_w = 2 ** (math.ceil(np.log2(W)))
        padding_h = target_h - H
        padding_w = target_w - W
        padding = [padding_h, padding_w]
        augmented_resolution = [target_h, target_w]
        
        # Flatten spatial dimensions
        x_flat = x.reshape(B, H * W, self.in_channels)
        
        # 预处理映射到 MWT 空间
        x_mwt = self.preprocess(x_flat)  # [B, H*W, WMT_dim]
        
        # 重塑为空间结构 [B, WMT_dim, H, W]
        x_mwt = x_mwt.permute(0, 2, 1).reshape(B, self.WMT_dim, H, W)
        
        # Padding 到 2 的幂次
        if not all(item == 0 for item in padding):
            x_mwt = F.pad(x_mwt, [0, padding_w, 0, padding_h])
        
        # 重塑为 MWT 输入格式 [B, H_aug, W_aug, c, k^2]
        x_mwt = x_mwt.reshape(B, self.WMT_dim, -1).permute(0, 2, 1).contiguous()
        x_mwt = x_mwt.reshape(B, *augmented_resolution, self.c, self.k ** 2)
        
        # MWT 谱层处理
        for i in range(self.n_layers):
            x_mwt = self.spectral_layers[i](x_mwt)
            if i < self.n_layers - 1:
                x_mwt = F.gelu(x_mwt)
        
        # 重塑回 [B, WMT_dim, H_aug, W_aug]
        x_mwt = x_mwt.reshape(B, -1, self.WMT_dim).permute(0, 2, 1).contiguous()
        x_mwt = x_mwt.reshape(B, self.WMT_dim, *augmented_resolution)
        
        # 移除 padding
        if not all(item == 0 for item in padding):
            x_mwt = x_mwt[..., :H, :W]
        
        # 上采样 [B, WMT_dim, H, W] -> [B, WMT_dim, H*scale, W*scale]
        x_up = self.upsample(x_mwt)
        
        # 计算输出分辨率
        H_out = H * self.upsample_factor[0]
        W_out = W * self.upsample_factor[1]
        
        # 重塑为 [B, H_out*W_out, WMT_dim]
        x_up = x_up.permute(0, 2, 3, 1).reshape(B, H_out * W_out, self.WMT_dim)
        
        # 输出投影
        out = self.output_projection(x_up)  # [B, H_out*W_out, out_channels]
        out = out.reshape(B, H_out, W_out, self.out_channels)
        
        return out