import torch
import torch.nn.functional as F

from typing import Dict, List, Any
from .loss import LossRecord

import importlib

@torch.no_grad()
def mse(pred, target, *args, **kwargs):
    return F.mse_loss(pred, target, reduction="mean")


@torch.no_grad()
def rmse(pred, target, *args, **kwargs):
    return torch.sqrt(mse(pred, target) + 1e-12)


@torch.no_grad()
def psnr(pred, target, shape, data_range=None, eps=1e-12):    
    m = mse(pred, target)
    pred = pred.permute(0, 3, 1, 2)  # BCHW
    target = target.permute(0, 3, 1, 2)  # BCHW
    
    L = (target.max() - target.min()).clamp_min(eps)

    return 20.0 * torch.log10(L) - 10.0 * torch.log10(m + eps)


@torch.no_grad()
def ssim(pred, target, shape, data_range=None, K1=0.01, K2=0.03, eps=1e-12):
    """
    仅自适配 C1/C2：
      C1 = (K1*L)^2, C2 = (K2*L)^2, 其中 L=data_range
    其他计算保持你原版不变（3x3 平均池化）
    """
    pred = pred.permute(0, 3, 1, 2)  # BCHW
    target = target.permute(0, 3, 1, 2)  # BCHW

    # 自适配 L
    if data_range is None:
        L = (target.max() - target.min()).clamp_min(eps)
    else:
        L = torch.as_tensor(data_range, device=pred.device, dtype=pred.dtype).clamp_min(eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + eps
    )
    return ssim_map.mean()


METRIC_REGISTRY = {
    "mse": mse,
    "rmse": rmse,
    "psnr": psnr,
    "ssim": ssim,
}


class Evaluator:
    def __init__(self, shape: List[int], **metric_kwargs: Any):
        self.kw = metric_kwargs
        self.shape = shape

    def init_record(self, loss_list: List[str] = []) -> LossRecord:
        loss_list = loss_list + list(METRIC_REGISTRY.keys())
        return LossRecord(loss_list)

    @torch.no_grad()
    def __call__(self, pred, target, record=None, batch_size=None, **batch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, fn in METRIC_REGISTRY.items():
            out[name] = fn(pred, target, self.shape, **self.kw).item()
        if record is not None:
            record.update(out)

        return out


#用于resshift的模型构建函数：
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)