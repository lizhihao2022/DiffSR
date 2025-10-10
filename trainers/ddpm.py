import os

import torch
from tqdm import tqdm
import wandb
import logging
from models.ddpm import UNet, GaussianDiffusion
from utils.loss import LossRecord

from .base import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.loss_type = args['loss_type']
        self.n_iter = args['n_iter']
        self.step = 0
        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-4,
                "linear_end": 2e-2
            },
            "val": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-4,
                "linear_end": 2e-2
            }
        }

    def build_model(self, args, **kwargs):
        model = UNet(
            in_channel=args['in_channel'],
            out_channel=args['out_channel'],
            # norm_groups=args['norm_groups'],
            inner_channel=args['inner_channel'],
            channel_mults=args['channel_mults'],
            attn_res=args['attn_res'],
            res_blocks=args['res_blocks'],
            dropout=args['dropout'],
            image_size=args['image_size'],
            )
        diffusion = GaussianDiffusion(
            model,
            image_size=args['image_size'],
            channels=args['channels'],
            loss_type=self.loss_type,
            conditional=args['conditional'],
            schedule_opt=self.beta_schedule,
            )
        
        # initializer = self.get_initializer('orthogonal')
        # diffusion.apply(initializer)
        
        diffusion.set_new_noise_schedule(
            self.beta_schedule['train'],
            device=self.device
        )
        diffusion.set_loss(self.device)

        return diffusion
    
    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss"])
        model.cuda()
        model.train()
        for (x, y) in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            # compute loss
            x = x.permute(0, 3, 1, 2)  # (b, c, h, w)
            y = y.permute(0, 3, 1, 2)
            data = {
                'SR': x,
                'HR': y
            }
            B, C, H, W = x.shape
            pix_loss = model(data)
            loss = pix_loss.sum() / int(B * C * H * W)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss and update progress bar
            loss_record.update({"train_loss": loss.item()}, n=B)

        if scheduler is not None:
            scheduler.step()
        return loss_record

    def evaluate(self, model, eval_loader, criterion, split='val', **kwargs):
        loss_record = LossRecord(["valid_loss", "PSNR", "SSIM"])
        model.cuda()
        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # compute loss
                x = x.permute(0, 3, 1, 2)  # (b, c, h, w)
                y = y.permute(0, 3, 1, 2)
                data = {
                    'SR': x,
                    'HR': y
                }
                B, C, H, W = x.shape
                y_pred = model.super_resolution(x, continous=False)
                loss = criterion(y_pred, y)
                psnr = calculate_psnr(y, y_pred)
                ssim = calculate_ssim(y, y_pred)
                loss_record.update({"valid_loss": loss.sum().item(), "PSNR": psnr, "SSIM": ssim}, n=B)
                break

        return loss_record
