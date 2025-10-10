from .base import BaseTrainer
from .ddpm import DDPMTrainer

_trainer_dict = {
    'FNO2d': BaseTrainer,
    'UNet2d': BaseTrainer, 
    'DDPM': DDPMTrainer,
}
