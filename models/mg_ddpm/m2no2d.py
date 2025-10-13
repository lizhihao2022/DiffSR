import math

import torch
import torch.nn as nn

from .grid_operator import LPFOperator2d, FourierOperator, VanillaOperator


def exists(x):
    return x is not None


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        sinusoid_in = torch.ger(t.view(-1).float(), self.inv_freq)
        emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        return emb


class GridBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_ite, time_emb_dim=None, bias=True, padding_mode='zeros'):
        super(GridBlock2d, self).__init__()
        self.num_ite = num_ite
        self.S = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias, padding_mode=padding_mode)
            for _ in range(num_ite)
        ])

        # 时间调制 (FiLM-like)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)  # 输出 gamma, beta
            )
        else:
            self.time_mlp = None

    def forward(self, A, f, u=None, t_emb=None):
        for i in range(self.num_ite):
            if u is None:
                u = self.S[i](f)
            else:
                u = u + self.S[i](f - A(u))
            r = f - A(u)

            # === 时间调制 ===
            if exists(self.time_mlp) and exists(t_emb):
                gamma, beta = self.time_mlp(t_emb).chunk(2, dim=-1)
                # 形状匹配 (B, C, 1, 1)
                gamma = gamma[:, :, None, None]
                beta = beta[:, :, None, None]
                u = u * (1 + gamma) + beta

        return u, r


class MultiGrid2d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_levels, op, time_emb_dim=None, **kwargs):
        super().__init__()
        self.op = op
        self.grid_levels = grid_levels
        self.num_level = len(grid_levels)

        self.A = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pre_S = GridBlock2d(in_channels, out_channels, 1, time_emb_dim=time_emb_dim)
        self.post_S = GridBlock2d(in_channels, out_channels, 1, time_emb_dim=time_emb_dim)
        self.grid_list = nn.ModuleList([
            GridBlock2d(in_channels, out_channels, grid_levels[i], time_emb_dim=time_emb_dim)
            for i in range(self.num_level)
        ])

    def forward(self, f, t_emb=None):
        u_n, r_n = self.pre_S(self.A, f, t_emb=t_emb)

        u_list = [None] * (self.num_level + 1)
        r_list = [None] * (self.num_level + 1)
        u_list[0], r_list[0] = u_n, r_n

        for i in range(self.num_level):
            u = self.op.restrict(u_list[i])
            u, r = self.grid_list[i](self.A, u, t_emb=t_emb)
            r_list[i + 1] = r
            u_list[i + 1] = u

        for i in range(self.num_level, 0, -1):
            u_list[i - 1] = u_list[i - 1] + self.op.prolongate(u_list[i])

        u_n = u_list[0]
        u_n, _ = self.post_S(self.A, f, u_n, t_emb=t_emb)
        return u_n


class M2NO2d(nn.Module):
    def __init__(self, model_args, **kwargs):
        super().__init__()
        in_channels = model_args['in_channels']
        out_channels = model_args['out_channels']
        k = model_args['k']
        c = model_args['c']
        num_layer = model_args['num_layer']
        grid_levels = model_args['grid_levels']
        time_emb_dim = model_args.get('time_emb_dim', 128)

        # === 时间编码器 ===
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.L_in = nn.Linear(in_channels, c * (k ** 2))
        self.L_hidden = nn.Linear(c * (k ** 2), c * (k ** 2))
        self.L_out = nn.Linear(c * (k ** 2), out_channels)

        self.wavelet_operator = LPFOperator2d(k=k, c=c, base='legendre')
        self.conv_list = nn.ModuleList([
            MultiGrid2d(c * (k ** 2), c * (k ** 2), grid_levels, self.wavelet_operator, time_emb_dim=time_emb_dim)
            for _ in range(num_layer)
        ])

        self.activate = nn.GELU()

    def forward(self, x, t):
        # (1) 时间嵌入
        t_emb = self.time_embed(t)

        # (2) 输入映射
        u = self.L_in(x.permute(0, 2, 3, 1))  # (B, H, W, C*k*k)
        u = u.permute(0, 3, 1, 2)  # (B, C, H, W)

        # (3) 多层 M2NO 传播 + 时间调制
        for conv in self.conv_list:
            u = conv(u, t_emb)
            u = self.activate(u)

        # (4) 输出层
        u = u.permute(0, 2, 3, 1)
        u = self.activate(self.L_hidden(u))
        u = self.L_out(u)
        return u.permute(0, 3, 1, 2)
