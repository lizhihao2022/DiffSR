import torch

from models.ddpm import UNet, GaussianDiffusion
from utils.loss import LossRecord

from .base import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, args):
        self.beta_schedule = args['beta_schedule']
        super().__init__(args)
        
    def build_model(self, **kwargs):
        model = UNet(
            in_channel=self.model_args['in_channel'],
            out_channel=self.model_args['out_channel'],
            inner_channel=self.model_args['inner_channel'],
            channel_mults=self.model_args['channel_mults'],
            attn_res=self.model_args['attn_res'],
            res_blocks=self.model_args['res_blocks'],
            dropout=self.model_args['dropout'],
            image_size=self.model_args['image_size'],
            )
        diffusion = GaussianDiffusion(
            model,
            image_size=self.model_args['image_size'],
            channels=self.model_args['channels'],
            loss_type=self.model_args['loss_type'],
            conditional=self.model_args['conditional'],
            schedule_opt=self.beta_schedule['train'],
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
            loss_record.update({"train_loss": loss.item()}, n=B)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record
    
    def evaluate(self, split="valid", **kwargs):
        if split == "valid":
            eval_loader = self.valid_loader
        elif split == "test":
            eval_loader = self.test_loader
        else:
            raise ValueError("split must be 'valid' or 'test'")
        
        loss_record = LossRecord(["{}_loss".format(split)])
        self.model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                x = x.permute(0, 3, 1, 2)
                B, C, H, W = x.shape
                y_pred = self._unwrap().super_resolution(x, continous=False)
                y_pred = y_pred.reshape(B, C, -1).permute(0, 2, 1)
                y_pred = self.normalizer.decode(y_pred)
                y = self.normalizer.decode(y)
                loss = self.loss_fn(y_pred, y)
                loss_record.update({"{}_loss".format(split): loss.item()}, n=x.size(0))
        if self.dist and dist.is_initialized():              
            loss_record.dist_reduce()
        return loss_record
