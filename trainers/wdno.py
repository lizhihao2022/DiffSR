from .base import BaseTrainer
from accelerate import Accelerator
from models.wdno import Unet3D_with_Conv3D, GaussianDiffusion


class WDNOTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.acc_args = args['accelerator']
        self.build_accelerator()
        
        self.gradient_accumulate_every = self.train_args.get('gradient_accumulate_every', 1)
        
        
    def build_model(self, **kwargs):
        model = Unet3D_with_Conv3D(**self.model_args)
        diffusion = GaussianDiffusion(
            denoise_fn=model,
            model_args=self.model_args
        )
        return diffusion

    def build_accelerator(self, **kwargs):
        self.accelerator = Accelerator(
            split_batches=self.acc_args.get('split_batches', True),
            mixed_precision= 'fp16' if self.acc_args.get('fp16', False) else 'no'
        )

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
