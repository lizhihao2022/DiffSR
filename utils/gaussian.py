import torch
import torch.nn.functional as F

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm


def build_scaling_rotation_3d(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation_3d(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    return R @ L


def build_rotation_3d(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


def build_scaling_rotation_2d(s, r):
    # s: [B, N, 2]
    # r: [B, N, 3]
    B = s.shape[0]
    N = s.shape[1]
    s = s.view(B*N, 2)
    r = r.view(B*N, 3)
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device="cuda")
    R = build_rotation_2d(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]

    out = R @ L
    
    return out.view(B, N, 2, 2)


def build_rotation_2d(r):
    # r: [B, N, 2]
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2])
    
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 2, 2), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]

    R[:, 0, 0] = 1 - 2 * (y*y)
    R[:, 0, 1] = 2 * (x*y - r)
    R[:, 1, 0] = 2 * (x*y + r)
    R[:, 1, 1] = 1 - 2 * (x*x)

    return R


def build_scaling_rotation_1d(s, r):
    L = torch.zeros((s.shape[0], 1, 1), dtype=torch.float, device="cuda")
    R = build_rotation_1d(r)

    L[:, 0, 0] = s[:, 0]

    return R @ L


def build_rotation_1d(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 1, 1), device='cuda')

    r = q[:, 0]
    x = q[:, 1]

    R[:, 0, 0] = 1 - 2 * (x*x)

    return R


def quick_gs_init(                         #
        x: torch.Tensor,                   # (B, N, pos_dim)  坐标
        a: torch.Tensor,                   # (B, N, feat_dim) 系数或初始场
        const_sigma: float = 0.15,         # 全局统一初始尺度
        pos_dim: int = 2,                  # 2D 或 3D
        out_dim: int = 1                   # 权重通道
) -> Dict[str, torch.Tensor]:
    """
    直接把每个采样点 x_i 作为一个 Gaussian μ_i，不做 FPS/KMeans。
    σ、旋转 r、权重 w 用常数或简单规则初始化，端到端可训练。
    """
    B, N, _ = x.shape
    device  = x.device

    # --- 1. μ 直接等于 x ---
    mu = x.clone()                         # (B, N, pos_dim)

    # --- 2. σ 设为常数 ---
    sigma = torch.full((B, N, pos_dim),    #
                       const_sigma,        #
                       device=device)      # (B, N, pos_dim)

    # --- 3. rotation: 单位旋转 (2D: θ=0；3D: quaternion=[1,0,0,0]) ---
    if pos_dim == 2:
        # 单角度 0
        quat = torch.zeros(B, N, 3, device=device)
        quat[..., 0] = 1.                 # [cos θ, sin θ] → 简写为 [1,0]
    else:  # 3D
        quat = torch.zeros(B, N, 4, device=device)
        quat[..., 0] = 1.                 # w=1, x=y=z=0

    # --- 4. weight: 用 a(x) 平均 or 直接复制 ---
    #   如果 feat_dim == out_dim，可直接拷贝；否则取 L2 范数
    if a.size(-1) == out_dim:
        weight = a.clone()                # (B, N, out_dim)
    else:
        # 例：取范数作为标量权重，再 broadcast
        w0 = a.norm(dim=-1, keepdim=True) # (B, N, 1)
        weight = w0.repeat(1, 1, out_dim)

    # 返回 dict，方便直接写入 GaussianField
    return {
        'mu':     mu,        # (B, N, pos_dim)
        'sigma':  sigma,     # (B, N, pos_dim)   正数
        'quat':   quat,      # (B, N, pos_dim+1) or (B,N,3)
        'weight': weight     # (B, N, out_dim)
    }


def vis_gaussian(mu, sigma, weight, ax, factor=1):
    mu = mu.squeeze(0)
    sigma = sigma.squeeze(0)

    mu = mu.cpu().detach().numpy()
    sigma = sigma.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    N, G, d = mu.shape

    # fig, ax = plt.subplots(figsize=(10,8))
    # heatmap = ax.imshow(u.reshape(h, w), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1, origin='lower', extent=[0.0, 1.0, 0.0, 1.0])
    # plt.colorbar(heatmap)

    sample_idx = np.arange(N)
    sample_idx = sample_idx[::factor]

    for n in tqdm(sample_idx):
        # normalize weights
        w_n = np.abs(weight) + 1e-6
        w_n = np.ones_like(w_n)
        w_n = w_n / np.sum(w_n)  # (G,)

        mu_n = mu[n]  # (G,2)
        sigma_n = sigma[n]  # (G,2)

        # 融合中心
        mu_fused = (w_n[:,None] * mu_n).sum(axis=0)  # (2,)

        # 融合尺度
        sigma_fused = (w_n[:,None] * (sigma_n**2 + (mu_n - mu_fused)**2)).sum(axis=0)  # (2,)

        # # 融合中心
        # mu_fused = (mu_n).sum(axis=0)  # (2,)

        # # 融合尺度
        # sigma_fused = ((sigma_n**2 + (mu_n - mu_fused)**2)).sum(axis=0)  # (2,)

        width = np.sqrt(sigma_fused[0]) * 0.05
        height = np.sqrt(sigma_fused[1]) * 0.05
        angle = 0

        ellipse = Ellipse(
            xy=(mu_fused[0], mu_fused[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor='red',
            facecolor='none',
            lw=1,
            alpha=0.8
        )
        ax.add_patch(ellipse)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted u(x) with Gaussian Ellipses')
    plt.axis('equal')
    plt.show()
