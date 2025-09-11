from .fno import FNO2d
from .unet import UNet2d
from .ddpm import GaussianDiffusion as DDPM

    
ALL_MODELS = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "DDPM": DDPM,
}
