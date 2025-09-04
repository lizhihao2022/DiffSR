import torch

from utils.loss import LossRecord, kl_diag_gaussian
from .base import BaseTrainer

from models import DynamicGPO
from .gs import GSTrainer


class DGPOTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'],
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'],
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
    
    def build_model(self, args, **kwargs):
        gs_field = GSTrainer(args).load_model(args['gs_path'])
        model = DynamicGPO(
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

    def load_model(self, args, **kwargs):
        gs_field = GSTrainer(args).load_model(args['gs_path'])
        model = DynamicGPO(
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
        loss_record = LossRecord(["train_loss", "data", "z", "mu", "sigma"])
        model.cuda()
        model.train()

        gs_field = model.gs_field
        pos_dim = gs_field.pos_dim
        gs_field.eval()
        
        for (x, y) in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')

            x_pos = x[..., :pos_dim]

            y_pred, z_pred, mu_pred, sigma_pred, weight_pred, gaussian_response_pred, dist_sq_pred, diff_pred = model(x)
            y_pred = y_pred.reshape(y.shape)
            data_loss = criterion(y_pred, y)

            y = torch.cat([x_pos, y], dim=-1)
            mu, sigma, weight = gs_field.encode_gaussian(y)
            z, gaussian_response, dist_sq, diff = gs_field.decode_gaussian(x_pos, mu, sigma, with_params=True)

            # mu_loss = ((mu_pred - x_pos.unsqueeze(2))**2).mean()
            # sigma_penalty = torch.relu(sigma - 0.5) + torch.relu(0.01 - sigma)
            # sigma_loss = sigma_penalty.mean()

            z_loss = criterion(z_pred, z)
            mu_loss = criterion(mu_pred.view(mu_pred.shape[0], mu_pred.shape[1], -1), mu.view(mu.shape[0], mu.shape[1], -1))
            sigma_loss = criterion(sigma_pred.view(sigma_pred.shape[0], sigma_pred.shape[1], -1), sigma.view(sigma.shape[0], sigma.shape[1], -1))
            # gr_loss = criterion(gaussian_response_pred, gaussian_response)
            # dist_loss = criterion(dist_sq_pred, dist_sq)
            # diff_loss = criterion(diff_pred, diff)

            # kl_loss = kl_diag_gaussian(mu_pred, sigma_pred, mu, sigma)

            loss = mu_loss + sigma_loss + data_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.update({
                "train_loss": loss.sum().item(),
                "data": data_loss.sum().item(),
                "z": z_loss.sum().item(),
                "mu": mu_loss.sum().item(),
                "sigma": sigma_loss.sum().item(),
                # "kl": kl_loss.sum().item(),
                # "gr": gr_loss.sum().item(),
                # "dist": dist_loss.sum().item(),
                # "diff": diff_loss.sum().item(),
                }, n=y_pred.shape[0])

        if scheduler is not None:
            scheduler.step()
    
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split), "data", "z", "mu", "sigma"])
        model.eval()
        gs_field = model.gs_field
        pos_dim = gs_field.pos_dim
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                x_pos = x[..., :pos_dim]

                y_pred, z_pred, mu_pred, sigma_pred, weight_pred, gaussian_response_pred, dist_sq_pred, diff_pred = model(x)
                y_pred = y_pred.reshape(y.shape)

                y = torch.cat([x_pos, y], dim=-1)
                mu, sigma, weight = gs_field.encode_gaussian(y)
                z, gaussian_response, dist_sq, diff = gs_field.decode_gaussian(x_pos, mu, sigma, with_params=True)

                z_loss = criterion(z_pred, z)
                mu_loss = criterion(mu_pred, mu)
                sigma_loss = criterion(sigma_pred, sigma)
                # gr_loss = criterion(gaussian_response_pred, gaussian_response)
                # dist_loss = criterion(dist_sq_pred, dist_sq)
                # diff_loss = criterion(diff_pred, diff)

                # kl_loss = kl_diag_gaussian(mu_pred, sigma_pred, mu, sigma)

                y_pred = eval_loader.dataset.y_normalizer.decode(y_pred)
                y = eval_loader.dataset.y_normalizer.decode(y[..., -1:])
                data_loss = criterion(y_pred, y)
                loss = data_loss
                loss_record.update({
                    "{}_loss".format(split): loss.sum().item(),
                    "data": data_loss.sum().item(),
                    "z": z_loss.sum().item(),
                    "mu": mu_loss.sum().item(),
                    "sigma": sigma_loss.sum().item(),
                    # "kl": kl_loss.sum().item(),
                    # "gr": gr_loss.sum().item(),
                    # "dist": dist_loss.sum().item(),
                    # "diff": diff_loss.sum().item(),
                    }, n=y_pred.shape[0])

        return loss_record
