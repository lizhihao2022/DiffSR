from .fno import FNO2d
from .unet import UNet2d
from .m2no import M2NO2d
from .galerkin import Galerkin_Transformer
from .MWT import MWT_SuperResolution
from .sronet import SRNO
from .swin_Transformer import SwinSR
from .EDSR import EDSR_net
from .HiNOTE import HiNOTE_net
from .swinIR import SwinIR_net

from . import ddpm
from . import sr3
from . import mg_ddpm
from . import remig


_model_dict = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "M2NO2d": M2NO2d,
    "Galerkin_Transformer": Galerkin_Transformer,
    "MWT2d": MWT_SuperResolution,
    "SRNO": SRNO,
    "Swin_Transformer": SwinSR,
    "EDSR": EDSR_net,
    "HiNOTE": HiNOTE_net,
    "SwinIR": SwinIR_net,
    "DDPM": {
        "model": ddpm.UNet,
        "diffusion": ddpm.GaussianDiffusion,
    },
    "SR3": {
        "model": sr3.UNet,
        "diffusion": sr3.GaussianDiffusion,
    },
    "ReMiG": {
        "model": remig.UNetModelSwin,
        "diffusion": remig.GaussianDiffusion,
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
    "MG-DDPM": {
        "model": mg_ddpm.M2NO2d,
        "diffusion": mg_ddpm.GaussianDiffusion,
    },
    "ReMiG": {
        "model": remig.UNetModelSwin,
        "diffusion": remig.GaussianDiffusion,
    },
}
