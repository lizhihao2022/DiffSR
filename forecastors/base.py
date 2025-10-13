import os
import yaml
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import _model_dict
from datasets import _dataset_dict
from utils.loss import LossRecord
from utils.metrics import Evaluator
from utils.sr import make_lr_blur


class BaseForecaster(object):
    def __init__(self, path):
        self.saving_path = path
        
        args_path = os.path.join(self.saving_path, 'config.yaml')
        self.args = yaml.load(open(args_path, 'r'), Loader=yaml.FullLoader)
        self.model_args = self.args['model']
        self.train_args = self.args['train']
        self.data_args = self.args['data']
        
        self.shape = self.data_args['shape']
        
        torch.manual_seed(self.train_args.get('seed', 42))
        self.device = self.train_args.get('device', 'cuda')
        
        self.model_name = self.model_args['name']
        self.model = self.build_model()
        self.load_model()
        self.build_evaluator()
        self.model.to(self.device)

    def build_model(self, **kwargs):
        if self.model_name not in _model_dict.keys():
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        print("Building model: {}".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model

    def load_model(self, **kwargs):
        model_path = os.path.join(self.saving_path, 'best_model.pth')
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(model_path))  
            
    def build_evaluator(self):  
        self.evaluator = Evaluator(self.shape)
        
    def build_data(self, **kwargs):
        if self.data_name not in _dataset_dict:
            raise NotImplementedError("Dataset {} not implemented".format(self.data_name))
        dataset = _dataset_dict[self.data_name](self.data_args, **kwargs)
        self.normalizer = dataset.normalizer
        self.train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=self.data_args.get('train_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            drop_last=True,
            pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(
            dataset.valid_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            pin_memory=True)
    
    def forecast(self, loader, normalizer, **kwargs):        
        loss_record = self.evaluator.init_record()
        all_y = []
        all_y_pred = []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                y_pred = normalizer.decode(y_pred)
                y = normalizer.decode(y)
                all_y.append(y)
                all_y_pred.append(y_pred)
        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        self.evaluator(y_pred, y, record=loss_record)
        print(loss_record)
        return loss_record
    
    def inference(self, x, y, **kwargs):
        return self.model(x).reshape(y.shape)

    def vis_ns(self, raw_x, raw_y, normalizer, save_path=None, 
               max_error=0.5, dpi=100, **kwargs):
        self.model.eval()
        with torch.no_grad():
            raw_x = raw_x.to(self.device)
            pred_y = self.inference(raw_x, raw_y, **kwargs)
        pred_y = normalizer.decode(pred_y)
        raw_y = normalizer.decode(raw_y)
        raw_x = make_lr_blur(raw_y.permute(0, 3, 1, 2), scale=self.data_args.get('sample_factor', 2)).permute(0, 2, 3, 1)
        
        pred_y = pred_y.cpu().reshape(self.shape)
        raw_y = raw_y.cpu().reshape(self.shape)
        raw_x = raw_x.cpu().reshape(self.shape[0] // self.data_args.get('sample_factor', 2), 
                                    self.shape[1] // self.data_args.get('sample_factor', 2))

        error_y = torch.abs(pred_y - raw_y)
        
        vmin = raw_y.min()
        vmax = raw_y.max()

        fig, axs = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True, dpi=dpi)

        axs[0].imshow(raw_x, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('Input LR')
        axs[0].axis('off')

        axs[1].imshow(raw_y, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_title('Ground Truth HR')
        axs[1].axis('off')

        heatmap = axs[2].imshow(pred_y, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[2].set_title('{} Prediction HR'.format(self.model_name))
        axs[2].axis('off')

        errormap = axs[3].imshow(error_y, cmap='hot', vmin=0, vmax=max_error)
        axs[3].set_title('Absolute Error')
        axs[3].axis('off')

        for ax, mappable in [(axs[2], heatmap), (axs[3], errormap)]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.04)  # 统一 size/pad
            fig.colorbar(mappable, cax=cax)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            
        plt.show()
