import torch
import torch.nn.functional as F
from typing import Optional

# ---------- 基础算子 ----------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device, dtype=dtype)/8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=device, dtype=dtype)/8.0
    return kx[None,None], ky[None,None]

def _depthwise(x, w):
    C = x.shape[1]
    W = w.expand(C,1,*w.shape[-2:])
    return F.conv2d(x, W, padding=w.shape[-1]//2, groups=C)

def _laplacian_kernel(device, dtype):
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=device, dtype=dtype)
    return k[None,None]

def _apply_mask(x, mask):
    return x if mask is None else x * mask

# ---------- 1) Laplacian / Biharmonic 抑振 ----------
def rho_lap(u: torch.Tensor, mask: Optional[torch.Tensor]=None, scale: float=1.0) -> torch.Tensor:
    """ ρ_lap(u) = -Δu ，抑制高频振荡但不强糊 """
    dev, dt = u.device, u.dtype
    kL = _laplacian_kernel(dev, dt)
    r = -_depthwise(u, kL)
    return _apply_mask(r, mask) * scale

def rho_biharm(u: torch.Tensor, mask: Optional[torch.Tensor]=None, scale: float=1.0) -> torch.Tensor:
    """ ρ_biharm(u) = +Δ^2 u 的负梯度 ≈ -Δ(Δu) = -Δ^2 u（更强的去波纹/去棋盘） """
    dev, dt = u.device, u.dtype
    kL = _laplacian_kernel(dev, dt)
    Lu = _depthwise(u, kL)
    r  = -_depthwise(Lu, kL)
    return _apply_mask(r, mask) * scale

# ---------- 2) 各向异性平滑（Perona–Malik） ----------
def rho_aniso(u: torch.Tensor, u_anchor: Optional[torch.Tensor]=None,
              mask: Optional[torch.Tensor]=None, kappa: float=0.1) -> torch.Tensor:
    """
    ρ_aniso(u) = -div( g(|∇(u_anchor)|) ∇u ), 其中 g(s)=1/(1+(s/kappa)^2)
    u_anchor: 可用粗场/低频 u0，引导边缘保护（无则用 u）
    """
    dev, dt = u.device, u.dtype
    kx, ky = _sobel_kernels(dev, dt)
    ua = u if u_anchor is None else u_anchor

    # edge indicator from anchor
    gx = _depthwise(ua, kx)
    gy = _depthwise(ua, ky)
    g  = 1.0 / (1.0 + (gx**2 + gy**2) / (kappa**2) + 1e-12)   # [N,1,H,W] per-channel broadcast later

    # flux = g * ∇u
    ux = _depthwise(u, kx); uy = _depthwise(u, ky)
    fx = g * ux;            fy = g * uy

    # div(flux) ≈ Dx^T fx + Dy^T fy ；Sobel 近似下再卷一次同核
    rx = _depthwise(fx, -kx)   # transpose conv kernel ≈ -kx for centered difference
    ry = _depthwise(fy, -ky)
    r  = -(rx + ry)            # 负号：下降方向
    return _apply_mask(r, mask)

# ---------- 3) 标量无通量边界（Neumann） ----------
def rho_noflux_scalar(u: torch.Tensor, mask: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    在固体边界 enforce ∂u/∂n = 0: ρ = -(1-M) * (∇u·n)
    mask: [N,1,H,W], 1=fluid, 0=solid
    """
    dev, dt = u.device, u.dtype
    kx, ky = _sobel_kernels(dev, dt)
    solid = 1.0 - mask
    nx = _depthwise(solid, kx); ny = _depthwise(solid, ky)
    nrm = torch.clamp(torch.sqrt(nx*nx + ny*ny), min=eps)
    nx /= nrm; ny /= nrm

    ux = _depthwise(u, kx)
    uy = _depthwise(u, ky)
    du_n = ux*nx + uy*ny
    r = -(solid * du_n)   # 方向同 u 的标量场
    return r

# ---------- 4) 谱一致（标量） ----------
def rho_spec_scalar(u: torch.Tensor,
                    target_spec: Optional[torch.Tensor]=None,
                    u0_for_anchor: Optional[torch.Tensor]=None,
                    num_bins: int=32, huber_delta: float=0.1,
                    mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    对每通道做 rfft2，按径向 bin 的 log-power 差构造权重后 iFFT 回像素域。
    """
    N, C, H, W = u.shape
    dev, dt = u.device, u.dtype
    x = u if mask is None else u * mask

    # 频域
    U = torch.fft.rfft2(x, norm='ortho')        # [N,C,H,W//2+1]
    P = (U.real**2 + U.imag**2)                 # power

    ky = torch.fft.fftfreq(H, d=1.0).to(dev).reshape(-1,1)
    kx = torch.fft.rfftfreq(W, d=1.0).to(dev).reshape(1,-1)
    kr = torch.sqrt(kx**2 + ky**2)              # [H, W//2+1]
    kmax = float(kr.max()) + 1e-12
    edges = torch.linspace(0., kmax, steps=num_bins+1, device=dev, dtype=dt)
    idx = torch.bucketize(kr.reshape(-1), edges, right=False) - 1
    idx = idx.clamp(0, num_bins-1).reshape(kr.shape)

    # bin 平均
    bins = torch.arange(num_bins, device=dev)
    counts = torch.zeros(num_bins, dtype=dt, device=dev)
    P_bins = torch.zeros((N, C, num_bins), dtype=dt, device=dev)
    for b in bins:
        m = (idx == int(b)).float()                     # [H, W//2+1]
        counts[b] = m.sum()
        if counts[b] > 0:
            P_bins[:,:,b] = (P * m).sum(dim=(-2,-1), keepdim=False) / counts[b]
    P_bins = torch.clamp(P_bins, min=1e-12)
    logP = torch.log(P_bins)                            # [N,C,B]

    if target_spec is None:
        assert u0_for_anchor is not None, "Provide target_spec or u0_for_anchor."
        U0 = torch.fft.rfft2(u0_for_anchor if mask is None else u0_for_anchor*mask, norm='ortho')
        P0 = (U0.real**2 + U0.imag**2)
        P0_bins = torch.zeros_like(P_bins)
        for b in bins:
            m = (idx == int(b)).float()
            if counts[b] > 0:
                P0_bins[:,:,b] = (P0 * m).sum(dim=(-2,-1)) / counts[b]
        P0_bins = torch.clamp(P0_bins, min=1e-12)
        logP_tgt = torch.log(P0_bins)
    else:
        # target_spec: [N,1 or C,B] or [N,B]
        if target_spec.ndim == 2:
            logP_tgt = target_spec[:,None,:].expand_as(logP)
        else:
            logP_tgt = target_spec

    diff = logP - logP_tgt                              # [N,C,B]
    # Huber-like 权重（在频域更稳）
    absd = diff.abs()
    huber = torch.where(absd <= huber_delta, diff, huber_delta*diff.sign())

    # 展到频率平面
    Wk = torch.zeros_like(P)
    for b in bins:
        if counts[b] > 0:
            Wb = huber[:,:,b]                          # [N,C]
            Wb = Wb[...,None,None]                     # [N,C,1,1]
            Wk += (idx == int(b)).float() * Wb

    # 对每通道频谱做修正并 iFFT
    U_corr = U * Wk
    u_corr = torch.fft.irfft2(U_corr, s=(H,W), norm='ortho')
    return u_corr if mask is None else u_corr*mask

# ---------- 5) 物理范围（软投影） ----------
def rho_range(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    软约束 [lo, hi]: ρ = -(ReLU(u-hi) - ReLU(lo-u))
    """
    over = torch.relu(u - hi)
    under = torch.relu(lo - u)
    return -(over - under)

# ---------- 总装：单量场 ρ(u) ----------
def rho(
    u: torch.Tensor,
    t: Optional[torch.Tensor]=None,
    mask: Optional[torch.Tensor]=None,
    # anchors
    u0_for_anchor: Optional[torch.Tensor]=None,
    # weights
    w_lap: float=0.0,
    w_biharm: float=0.5,
    w_aniso: float=0.5,
    w_noflux: float=0.0,
    w_spec: float=0.5,
    w_range: float=0.0,
    # params
    aniso_kappa: float=0.1,
    spec_bins: int=32,
    spec_huber: float=0.1,
    range_lo: Optional[float]=None,
    range_hi: Optional[float]=None,
):
    parts = []

    if w_lap > 0:
        parts.append(w_lap * rho_lap(u, mask=mask))

    if w_biharm > 0:
        parts.append(w_biharm * rho_biharm(u, mask=mask))

    if w_aniso > 0:
        parts.append(w_aniso * rho_aniso(u, u_anchor=u0_for_anchor, mask=mask, kappa=aniso_kappa))

    if w_noflux > 0 and mask is not None:
        parts.append(w_noflux * rho_noflux_scalar(u, mask))

    if w_spec > 0:
        parts.append(w_spec * rho_spec_scalar(u, u0_for_anchor=u0_for_anchor, num_bins=spec_bins,
                                              huber_delta=spec_huber, mask=mask))

    if w_range > 0 and (range_lo is not None) and (range_hi is not None):
        parts.append(w_range * rho_range(u, range_lo, range_hi))

    if len(parts) == 0:
        return torch.zeros_like(u)

    r = torch.stack(parts, dim=0).sum(dim=0)

    # 可选：时间门控（早期更强抑振，后期减弱）
    if t is not None:
        T = float(t.max().item()) if t.numel()>0 else 1.0
        s = 1.0 - (t.float()/max(T,1.0))           # 0(晚)->1(早)
        gate = (0.6 + 0.4*s).view(-1,1,1,1)        # 早期 1.0, 晚期 0.6
        r = r * gate
    return r
