import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .layers.Basic import MLP, LinearAttention
from .layers.Embedding import unified_pos_embedding
from einops import rearrange


class Galerkin_Transformer_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1a = nn.LayerNorm(hidden_dim)
        self.Attn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                    dropout=dropout, attn_type='galerkin')
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Galerkin_Transformer(nn.Module):
    """
    Galerkin Transformer for 2D Super-Resolution
    Input:  [B, H, W, C] - Low-resolution field
    Output: [B, H*scale, W*scale, out_dim] - High-resolution field
    """
    def __init__(self, model_params):
        super(Galerkin_Transformer, self).__init__()
        """
        Args:
            model_params: dict containing:
                - in_dim: int, number of input channels, default: 1
                - out_dim: int, number of output channels, default: 1
                - ref: int, position encoding reference resolution, default: 8
                - n_hidden: int, hidden dimension, default: 256
                - n_layers: int, number of transformer layers, default: 6
                - n_heads: int, number of attention heads, default: 8
                - dropout: float, dropout rate, default: 0.0
                - mlp_ratio: int, MLP expansion ratio, default: 4
                - act: str, activation function {gelu, relu, tanh}, default: 'gelu'
                - upsample_factor: list of int or int, upsampling scale, default: [2, 2]
        """
        self.in_dim = model_params.get('in_dim', 1)
        self.out_dim = model_params.get('out_dim', 1)
        self.ref = model_params.get('ref', 8)
        self.n_hidden = model_params.get('n_hidden', 256)
        n_layers = model_params.get('n_layers', 6)
        n_heads = model_params.get('n_heads', 8)
        dropout = model_params.get('dropout', 0.0)
        mlp_ratio = model_params.get('mlp_ratio', 4)
        act = model_params.get('act', 'gelu')
        
        upsample_factor = model_params.get('upsample_factor', [2, 2])
        if isinstance(upsample_factor, int):
            upsample_factor = [upsample_factor, upsample_factor]
        self.upsample_factor = upsample_factor
        
        self.pos_embedding = None
        
        self.preprocess = MLP(
            self.in_dim + self.ref ** 2,
            self.n_hidden * 2,
            self.n_hidden,
            n_layers=0, 
            res=False, 
            act=act
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Galerkin_Transformer_block(
                num_heads=n_heads,
                hidden_dim=self.n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=self.out_dim,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        
        self.placeholder = nn.Parameter(
            (1 / self.n_hidden) * torch.rand(self.n_hidden, dtype=torch.float)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, H, W, C] 低分辨率输入
            
        Returns:
            output: [B, H*scale_h, W*scale_w, out_dim] 高分辨率输出
        """
        B, H_in, W_in, C_in = x.shape
        
        H_out = H_in * self.upsample_factor[0]
        W_out = W_in * self.upsample_factor[1]
        
        if self.pos_embedding is None or self.pos_embedding.shape[1] != H_out * W_out:
            self.pos_embedding = unified_pos_embedding(
                shapelist=[H_out, W_out], 
                ref=self.ref,
                batchsize=1
            ).to(x.device)
        
        # 双线性插值上采样 [B, H, W, C] → [B, H*scale, W*scale, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.interpolate(
            x, 
            size=(H_out, W_out), 
            mode='bilinear', 
            align_corners=True
        )
        x = x.permute(0, 2, 3, 1)  # [B, H_out, W_out, C]
        
        if C_in > self.in_dim:
            x = x[..., :self.in_dim]
        
        # Reshape: [B, H_out, W_out, in_dim] → [B, H_out*W_out, in_dim]
        fx = rearrange(x, 'b h w c -> b (h w) c')
        
        # 拼接位置编码 [B, H_out*W_out, in_dim] + [1, H_out*W_out, ref^2] → [B, H_out*W_out, in_dim+ref^2]
        pos_emb = self.pos_embedding.repeat(B, 1, 1)
        fx = torch.cat([fx, pos_emb], dim=-1)
        
        fx = self.preprocess(fx)  # [B, H_out*W_out, n_hidden]
        fx = fx + self.placeholder[None, None, :]
        
        # Transformer blocks
        for block in self.blocks:
            fx = block(fx)  # [B, H_out*W_out, out_dim]
        
        # Reshape 回空间维度
        output = rearrange(fx, 'b (h w) c -> b h w c', h=H_out, w=W_out)
        
        return output