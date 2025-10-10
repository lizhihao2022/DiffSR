import torch
import torch.nn.functional as F
import math


def mod_crop(x, scale):
    B, C, H, W = x.shape
    H2 = H - (H % scale)
    W2 = W - (W % scale)
    return x[:, :, :H2, :W2]


def gaussian_kernel2d(ks=7, sigma=1.2, device='cpu', dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k  # (ks, ks)


def depthwise_gaussian_blur(x, ks=7, sigma=1.2, padding='same'):
    """
    x: (B,C,H,W)  每通道独立高斯模糊（深度可分卷积）
    """
    B, C, H, W = x.shape
    k = gaussian_kernel2d(ks, sigma, x.device, x.dtype).view(1,1,ks,ks)
    weight = k.repeat(C, 1, 1, 1)              # (C,1,ks,ks)
    if padding == 'same':
        pad = ks // 2
    else:
        pad = 0
    return F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=C)


# --------- HR -> 模糊 LR ---------
def make_lr_blur(HR, scale=4, ks=7, sigma=1.2, down_mode='bicubic'):
    """
    HR: (B,C,H,W)  建议先做标准化（z-score 或 [0,1]），再生成 LR
    返回: LR_blur: (B,C,h,w)
    """
    x = mod_crop(HR, scale)                                  # 尺度对齐
    x = depthwise_gaussian_blur(x, ks=ks, sigma=sigma)       # 高斯模糊
    h = x.shape[-2] // scale
    w = x.shape[-1] // scale
    LR = F.interpolate(x, size=(h, w), mode=down_mode, align_corners=False, antialias=True)
    return LR
