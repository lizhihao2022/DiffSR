import torch

from models import GaussianField
from .base import BaseTrainer
from utils.loss import LossRecord


class GSTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])

    def build_model(self, args, **kwargs):
        model = GaussianField(
            in_dim=args['in_dim'],
            out_dim=args['out_dim'],
            hidden_dim=args['hidden_dim'],
            pos_dim=args['pos_dim'],
            num_gaussians=args['num_gaussians'],
            sigma_init=args['sigma_init']
        )
        return model
    
    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss", "data", "global", "local"])
        model.cuda()
        model.train()
        
        for (x, y) in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')

            x_pred = model(x)
            x_pred = x_pred.reshape(x[..., -1:].shape)
            data_loss = criterion(x_pred, x[..., -1:])
            global_loss, local_loss = model.encoder.compute_loss(x)
            
            loss = data_loss + global_loss + local_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.update({
                "train_loss": loss.sum().item(),
                "data": data_loss.sum().item(),
                "global": global_loss.sum().item(),
                "local": local_loss.sum().item(),
                }, n=x_pred.shape[0])

        if scheduler is not None:
            scheduler.step()
    
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split), "data", "global", "local"])
        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')

                x_pred = model(x)
                x_pred = x_pred.reshape(x[..., -1:].shape)
                global_loss, local_loss = model.encoder.compute_loss(x)

                x_pred = eval_loader.dataset.x_normalizer.decode(x_pred)
                x = eval_loader.dataset.x_normalizer.decode(x[..., -1:])
                data_loss = criterion(x_pred, x)
                
                loss = data_loss
                loss_record.update({
                    "{}_loss".format(split): loss.sum().item(),
                    "data": data_loss.sum().item(),
                    "global": global_loss.sum().item(),
                    "local": local_loss.sum().item(),
                    }, n=x_pred.shape[0])

        return loss_record
