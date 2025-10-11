from .base import BaseTrainer
from .ddpm import DDPMTrainer
from .sr3 import SR3Trainer

_trainer_dict = {
    'FNO2d': BaseTrainer,
    'UNet2d': BaseTrainer, 
    'DDPM': DDPMTrainer,
    'SR3': SR3Trainer,
}
