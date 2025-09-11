import os

import torch
from tqdm import tqdm
import wandb
import logging
from models.ddpm import UNet, GaussianDiffusion
from utils.loss import LossRecord, calculate_psnr, calculate_ssim

from .base import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'],
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
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
    
    def process(self, model, train_loader, valid_loader, test_loader, optimizer, 
                criterion, regularizer=None, scheduler=None, **kwargs):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        with tqdm(total=self.epochs) as bar:
            for epoch in range(self.epochs):
                train_loss_record = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer=regularizer, **kwargs)
                if self.verbose:
                    # tqdm.write("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
                    self.logger("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
                if self.wandb:
                    wandb.log(train_loss_record.to_dict())
                
                if self.saving_checkpoint and (epoch + 1) % self.checkpoint_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.cpu().state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss_record': train_loss_record.to_dict(),
                        }, os.path.join(self.saving_path, "checkpoint_{}.pth".format(epoch)))
                    model.cuda()
                    if self.verbose:
                        # tqdm.write("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                        self.logger("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                    
                if (epoch + 1) % self.eval_freq == 0:
                    valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid", **kwargs)
                    if self.verbose:
                        # tqdm.write("Epoch {} | {}".format(epoch, valid_loss_record))
                        self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                    valid_metrics = valid_loss_record.to_dict()
                    
                    if self.wandb:
                        wandb.log(valid_loss_record.to_dict())
                    
                    if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
                        counter = 0
                        best_epoch = epoch
                        best_metrics = valid_metrics
                        torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                        model.cuda()
                        if self.verbose:
                            # tqdm.write("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                            self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                    elif self.patience != -1:
                        counter += 1
                        if counter >= self.patience:
                            if self.verbose:
                                self.logger("Early stop at epoch {}".format(epoch))
                            break
                bar.update(1)

        self.logger("Optimization Finished!")
        
        # load best model
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid", **kwargs)
        self.logger("Valid metrics: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(model, test_loader, criterion, split="test", **kwargs)
        self.logger("Test metrics: {}".format(test_loss_record))
        
        if self.wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())
            
        return model
    
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
