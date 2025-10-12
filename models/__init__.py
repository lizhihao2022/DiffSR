from .fno import FNO2d
from .unet import UNet2d
from . import ddpm
from . import sr3
from .galerkin import Galerkin_Transformer
from .MWT import MWT_SuperResolution
from .sronet import SRNO
from .swin_Transformer import SwinSR
    
_model_dict = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "Galerkin_Transformer": Galerkin_Transformer,
    "MWT2d": MWT_SuperResolution,
    "SRNO": SRNO,
    "Swin_Transformer": SwinSR,
    "DDPM": {
        "model": ddpm.UNet,
        "diffusion": ddpm.GaussianDiffusion,
    },
    "SR3": {
        "model": sr3.UNet,
        "diffusion": sr3.GaussianDiffusion,
    },
}

_ddpm_dict = {
    "DDPM": {
        "model": ddpm.UNet,
        "diffusion": ddpm.GaussianDiffusion,
    },
    "SR3": {
        "model": sr3.UNet,
        "diffusion": sr3.GaussianDiffusion,
    },
}
