from .gpo import GPOTrainer
from .dgpo import DGPOTrainer
from .gs import GSTrainer

TRAINER_DICT = {
    'GPO': GPOTrainer,
    'DGPO': DGPOTrainer,
    'GaussianField': GSTrainer,
}