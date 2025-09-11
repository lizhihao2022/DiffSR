from .fno2d import FNO2DTrainer
from .ddpm import DDPMTrainer

TRAINER_DICT = {
    'FNO2d': FNO2DTrainer,  
    'DDPM': DDPMTrainer,
}