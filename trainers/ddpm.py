import torch
import torch.distributed as dist

from models import _ddpm_dict
from utils.loss import LossRecord

from .base import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, args):
        self.beta_schedule = args['beta_schedule']
        super().__init__(args)
        
    def build_model(self, **kwargs):
        model = _ddpm_dict[self.model_name]["model"](self.model_args)
        diffusion = _ddpm_dict[self.model_name]["diffusion"](
            model,
            model_args=self.model_args,
            schedule_opt=self.beta_schedule['train']
        )
        diffusion.set_new_noise_schedule(
            self.beta_schedule['train'],
            device=self.device
        )
        diffusion.set_loss(self.device)

        return diffusion

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
            data = {
                'SR': x,
                'HR': y
            }
            B, C, H, W = x.shape
            pix_loss = self.model(data)
            loss = pix_loss.sum() / int(B * C * H * W)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_record.update({"train_loss": loss.item()}, n=1)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record
    
    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y_pred = self._unwrap().super_resolution(x, continous=False)
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(y.shape)
        return y_pred
