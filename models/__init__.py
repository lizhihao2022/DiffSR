from .fno import FNO2d
from .unet import UNet2d
from .ddpm import GaussianDiffusion as DDPM
from .sr3 import GaussianDiffusion as SR3
    
_model_dict = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "DDPM": DDPM,
    "SR3": SR3,
}
