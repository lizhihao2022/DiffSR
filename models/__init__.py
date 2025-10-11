from .fno import FNO2d
from .unet import UNet2d
from . import ddpm
from . import sr3
    
_model_dict = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
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
