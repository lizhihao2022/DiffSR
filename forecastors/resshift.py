from .base import BaseForecaster
from models import _ddpm_dict
import torch
import torch.nn.functional as F
import numpy as np
from utils.metrics import get_obj_from_str
class ResshiftForecaster(BaseForecaster):
    def __init__(self, path):
        super().__init__(path)
        
    def build_model(self, **kwargs):
        
        self.resshift_cfg = self.args['resshift']
        
        params = self.resshift_cfg["model"]['params']
        model =get_obj_from_str(self.resshift_cfg['model']['target'])(**params)
        
        params = self.resshift_cfg["diffusion"]['params']
        self.base_diffusion = get_obj_from_str(self.resshift_cfg['diffusion']['target'])(**params)
        return model

    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=y.shape[2:], mode='bicubic', align_corners=False)
        
        indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()        

        if not (self.base_diffusion.num_timesteps-1) in indices:
            indices.append(self.base_diffusion.num_timesteps-1)     
            
        im_lq = x  
        
        model_kwargs = {'lq':x,}
        
        tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        device= x.device
                        )
        
        
        y_pred = self.base_diffusion.p_sample_loop(
                        y=im_lq,
                        model=self.model,
                        first_stage_model=None,
                        noise=None,
                        clip_denoised=None,
                        model_kwargs=model_kwargs,
                        device=x.device,
                        progress=True,
                        )
        
        # y_pred = self._unwrap().super_resolution(x, continous=False)
        y_pred = y_pred.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        return y_pred

