from .base import BaseTrainer
from .ddpm import DDPMTrainer

_trainer_dict = {
    'FNO2d': BaseTrainer,
    'UNet2d': BaseTrainer, 
    'M2NO2d': BaseTrainer,
    'DDPM': DDPMTrainer,
    'SR3': DDPMTrainer,
    "MG-DDPM": DDPMTrainer,
}
