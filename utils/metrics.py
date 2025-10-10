# metrics.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Any

@torch.no_grad()
def mse(pred, target):
    return F.mse_loss(pred, target, reduction="mean")

@torch.no_grad()
def rmse(pred, target):
    return torch.sqrt(mse(pred, target) + 1e-12)

@torch.no_grad()
def psnr(pred, target, data_range=1.0):
    m = mse(pred, target)
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(m + 1e-12)

@torch.no_grad()
def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(1), target.unsqueeze(1)
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred*pred, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target*target, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred*target, 3, 1, 1) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x.pow(2)+mu_y.pow(2)+C1) * (sigma_x+sigma_y+C2) + 1e-12)
    return ssim_map.mean()

METRIC_REGISTRY = {
    "mse": mse,
    "rmse": rmse,
    "psnr": psnr,
    "ssim": ssim,
}

class Evaluator:
    def __init__(self, names: List[str], **metric_kwargs: Any):
        self.names = names
        self.kw = metric_kwargs

    @torch.no_grad()
    def __call__(self, pred, target, record=None, batch_size=None, **batch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in self.names:
            fn = METRIC_REGISTRY[name]
            if name == "psnr":
                val = fn(pred, target, **self.kw)      # 允许传 data_range
            else:
                val = fn(pred, target)                  # 默认调用
            out[name] = float(val.detach().item())
        # 可选：直接写入记录器（用于分布式全局均值的 sum/count 聚合）
        if record is not None:
            n = int(batch_size) if batch_size is not None else int(pred.size(0))
            record.update(out, n=n)
        return out
