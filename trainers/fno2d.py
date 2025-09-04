from models import FNO2d
from .base import BaseTrainer


class FNO2DTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'],
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
    
    def build_model(self, args, **kwargs):        
        model = FNO2d(
            modes1=args['modes1'],
            modes2=args['modes2'],
            width=args['width'],
            fc_dim=args['fc_dim'],
            layers=args['layers'],
            in_dim=args['in_dim'],
            out_dim=args['out_dim'],
            act=args['act'],
            )
        return model
