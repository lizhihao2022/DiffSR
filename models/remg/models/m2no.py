import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .basic_ops import normalization, timestep_embedding, zero_module
from .grid_operator import LPFOperator2d

# ---------------------------
# 1) 带 FiLM 的 GridBlock（不改求解样式，只加时间步调制）
# ---------------------------
class GridBlock2d(nn.Module):
    """
    语义与原 GridBlock2d 一致：迭代 u = u + S(f - A(u))。
    差异：S 里加入了 FiLM(t) 调制（scale/shift），让步数参与更新强度与方向。
    """
    def __init__(self, in_channels, out_channels, num_ite, emb_channels, bias=True, padding_mode='zeros'):
        super().__init__()
        self.num_ite = num_ite
        self.norm = normalization(in_channels)  # 对残差做个轻量 norm
        self.S = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias, padding_mode=padding_mode)
            for _ in range(num_ite)
        ])
        # FiLM: 用时间嵌入生成 per-channel 的 scale/shift
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels)
        )

    def forward(self, A, f, emb, u=None):
        """
        A: 近似算子（与原版相同）
        f: 右端项（我们会把它当“特征化的残差”）
        emb: [N, emb_channels] 时间步嵌入
        u: 初始解
        """
        N, C, H, W = f.shape
        gamma, beta = self.emb(emb).chunk(2, dim=1)  # [N,C]
        gamma = gamma[..., None, None]
        beta  = beta[...,  None, None]

        for i in range(self.num_ite):
            if u is None:
                h = self.S[i](self.norm(f))
                u = (1 + gamma) * h + beta
            else:
                res = f - A(u)
                h = self.S[i](self.norm(res))
                u = u + (1 + gamma) * h + beta
        r = f - A(u)
        return u, r

# ---------------------------
# 2) 多重网格：加入 per-level 的 t 调度权重
# ---------------------------
class MultiGrid2d(nn.Module):
    """
    与原 MultiGrid2d 基本一致，只是：
      - pre_S / grid_list / post_S 都换成带 FiLM 的 GridBlock2d_FiLM
      - 每个层级的 prolongation 融合时，乘上 W_l(t) 门控（可学习 + t 嵌入）
    """
    def __init__(self, in_channels, out_channels, grid_levels, op,
                 emb_channels, bias=True, padding_mode='zeros', resolutions=(64, 64), norm=False):
        super().__init__()
        self.op = op
        self.resolutions = resolutions
        self.num_level = len(grid_levels)
        self.norm = norm

        self.A = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias, padding_mode=padding_mode)
        self.pre_S  = GridBlock2d(in_channels, out_channels, 1, emb_channels, bias=bias, padding_mode=padding_mode)
        self.post_S = GridBlock2d(in_channels, out_channels, 1, emb_channels, bias=bias, padding_mode=padding_mode)
        self.grid_list = nn.ModuleList([
            GridBlock2d(in_channels, out_channels, grid_levels[i], emb_channels, bias=bias, padding_mode=padding_mode)
            for i in range(self.num_level)
        ])
        if norm:
            self.norm_list = nn.ModuleList([
                nn.LayerNorm([out_channels, resolutions[0] // (2 ** i), resolutions[1] // (2 ** i)])
                for i in range(self.num_level)
            ])

        # 每层一个可学习门控 + t 嵌入线性投影
        self.level_gate = nn.Parameter(th.zeros(self.num_level))  # a_l
        self.level_proj = nn.Linear(emb_channels, self.num_level)  # b_l^T * te

    def forward(self, f, emb):
        u_list = [None] * (self.num_level + 1)
        r_list = [None] * (self.num_level + 1)

        # 预平滑
        u_n, r_n = self.pre_S(self.A, f, emb)
        u_list[0], r_list[0] = u_n, r_n

        # 下行：粗化求解
        for i in range(self.num_level):
            u = self.op.restrict(u_list[i])
            u, r = self.grid_list[i](self.A, u, emb)
            u_list[i+1], r_list[i+1] = u, r

        # 计算层级权重 W_l(t)（sigmoid 门控，随 t 变化）
        gates = th.sigmoid(self.level_gate[None, :] + self.level_proj(emb))  # [N, L]

        # 上行：延拓 + 门控融合
        for i in range(self.num_level, 0, -1):
            up = self.op.prolongate(u_list[i])
            w = gates[:, i-1].view(-1, 1, 1, 1)  # [N,1,1,1]
            if self.norm:
                u_list[i-1] = self.norm_list[i-1](u_list[i-1] + w * up)
            else:
                u_list[i-1] = u_list[i-1] + w * up

        # 后平滑
        u_n, _ = self.post_S(self.A, f, emb, u_list[0])
        return u_n  # 注意：这里返回的是“解”的校正结果；外层用 1x1 映射成 e

# ---------------------------
# 3) 新的 M2NO：直接作为 S(x,r,t)，输出校正 e
# ---------------------------
class M2NO2d(nn.Module):
    """
    直接用作 S(x, r, t):
      输入: x 当前解, r 残差, t 步数
      输出: e 校正方向，与 x 同形状
    整体仍是 M2NO/V-cycle；只在入口/中间加入时间步调制与多尺度门控。
    """
    def __init__(self, model_args):
        super().__init__()
        C            = model_args['in_channels']           # 与 x/r 的通道一致
        base         = model_args.get('base_channels', 64)
        grid_levels  = model_args.get('grid_levels', [1,1,1])
        resolutions  = model_args.get('resolutions', (64, 64))
        bias         = model_args.get('bias', True)
        padding_mode = model_args.get('padding_mode', 'zeros')

        # 时间嵌入
        self.time_dim = base
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, base*4), nn.SiLU(),
            nn.Linear(base*4, base*4)
        )
        emb_channels = base*4

        # 把 [r, x] 做一次 1x1 投影：残差感知
        self.in_proj = nn.Conv2d(2*C, base, kernel_size=1, bias=True)

        # M2NO/V-cycle 核心
        self.core_op = LPFOperator2d(
            k=model_args.get('k', 3),
            c=model_args.get('c', 4),
            base=model_args.get('base', 'legendre'),
            bias=bias, padding_mode=padding_mode
        )
        self.mg = MultiGrid2d(
            in_channels=base, out_channels=base,
            grid_levels=grid_levels, op=self.core_op,
            emb_channels=emb_channels, bias=bias,
            padding_mode=padding_mode, resolutions=resolutions, norm=True
        )

        # 输出：把“解的校正”映射回 e（zero init 稳定）
        self.out_proj = nn.Sequential(
            normalization(base),
            nn.SiLU(),
            zero_module(nn.Conv2d(base, C, kernel_size=1, bias=True))
        )

    def forward(self, x, r, t, cond=None):
        """
        x, r: [N,C,H,W], t: [N] (long)
        """
        dev = x.device
        x_dtype = x.dtype
        # 时间步嵌入（与原 UNet 的 timestep_embedding 对齐）
        te = timestep_embedding(t, self.time_dim)  # [N, base]
        te = te.to(device=dev, dtype=th.float32)
        emb = self.time_mlp(te)                    # [N, emb_channels]
        
        # 残差感知输入：[r, x] -> base 通道
        h = self.in_proj(th.cat([r, x], dim=1))    # [N, base, H, W]
        # V-cycle（多尺度权重随 t 调度）
        h = self.mg(h, emb)                        # [N, base, Hc, Wc] （与 resolutions 对齐后又回到同尺度）

        # 输出校正 e
        e = self.out_proj(h)                       # [N, C, H, W]
        return e
