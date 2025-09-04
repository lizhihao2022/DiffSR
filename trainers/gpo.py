import torch

from models import GPO

from .base import BaseTrainer
from .gs import GSTrainer

from utils.loss import LossRecord


class GPOTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])

    def build_model(self, args):
        gs_field = GSTrainer(args).load_model(args['gs_path'])
        model = GPO(
            in_dim=args['in_dim'],
            out_dim=args['out_dim'],
            gs_field=gs_field,
            hidden_dim=args['hidden_dim'],
            pos_dim=args['pos_dim'],
            num_layers=args['num_layers'],
            num_gaussians=args['num_gaussians'],
            num_heads=args['num_heads'],
        )
        model.freeze_gs()
        return model

    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss", "data_loss", "z_loss"])
        model.cuda()
        model.train()
        for (x, y) in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            
            y_pred, z_pred = model(x)
            y = y.view(y.shape[0], 64, 64, y.shape[-1])
            y_pred = y_pred.reshape(y.shape)
            data_loss = criterion(y_pred, y)
            
            y = y.view(y.shape[0], -1, y.shape[-1])
            y = torch.cat([x[..., :2], y], dim=-1)
            z = model.gs_field.encode(y)
            z_loss = criterion(z_pred, z)

            loss = z_loss + data_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_record.update({
                "train_loss": loss.sum().item(),
                "data_loss": data_loss.sum().item(),
                "z_loss": z_loss.sum().item(),
                }, n=y_pred.shape[0])

        if scheduler is not None:
            scheduler.step()
    
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split), "data_loss", "z_loss"])
        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                y_pred, z_pred = model(x)
                y_pred = y_pred.reshape(y.shape)
                
                y = torch.cat([x[..., :2], y], dim=-1)
                z = model.gs_field.encode(y)
                z_loss = criterion(z_pred, z)
                
                y_pred = eval_loader.dataset.y_normalizer.decode(y_pred)
                y = eval_loader.dataset.y_normalizer.decode(y[..., -1:])
                data_loss = criterion(y_pred, y)
                loss = data_loss
                loss_record.update({
                    "{}_loss".format(split): loss.sum().item(),
                    "data_loss": data_loss.sum().item(),
                    "z_loss": z_loss.sum().item(),
                    }, n=y_pred.shape[0])

        return loss_record
