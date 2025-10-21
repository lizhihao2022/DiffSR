import torch
import torch.distributed as dist
import functools
import numpy as np
from models import _ddpm_dict
from utils.loss import LossRecord
import torch.nn.functional as F
from .base import BaseTrainer

from utils.metrics import get_obj_from_str

class ResshiftTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        
    def build_model(self, **kwargs):
        
        self.resshift_cfg = self.args['resshift']
        
        params = self.resshift_cfg["model"]['params']
        model =get_obj_from_str(self.resshift_cfg['model']['target'])(**params)
        
        params = self.resshift_cfg["diffusion"]['params']
        self.base_diffusion = get_obj_from_str(self.resshift_cfg['diffusion']['target'])(**params)
        return model

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=y.shape[2:], mode='bicubic', align_corners=False)

            micro_data = {
                'lq': x,
                'gt': y
            }
            B, C, H, W = x.shape
            
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device= x.device,
                    )

            model_kwargs = {'lq':micro_data['lq'],}

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=None,
                model_kwargs=model_kwargs,
                noise=None,
            )
            
            losses, z0_pred, z_t = compute_losses()
            
            loss = losses['mse']
            # print(f"epoch:{epoch}, iter:{i}, loss:{loss.item():.6f}")
            loss_record.update({"train_loss": loss}, n=B)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record
    
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
        
        self._unwrap()
        
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
