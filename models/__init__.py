from .fno import FNO2d
from .unet import UNet2d
from .m2no import M2NO2d

from . import ddpm
from . import sr3
# from . import wdno
from . import mg_ddpm

    
_model_dict = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "M2NO2d": M2NO2d,
    "DDPM": {
        "model": ddpm.UNet,
        "diffusion": ddpm.GaussianDiffusion,
    },
    "SR3": {
        "model": sr3.UNet,
        "diffusion": sr3.GaussianDiffusion,
    },
    # "WDNO": {
    #     "model": wdno.Unet3D_with_Conv3D,
    #     "diffusion": wdno.GaussianDiffusion,
    # },
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
    "MG-DDPM": {
        "model": mg_ddpm.M2NO2d,
        "diffusion": mg_ddpm.GaussianDiffusion,
    },
}
