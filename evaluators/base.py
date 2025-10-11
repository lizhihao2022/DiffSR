import os
import torch

from models import _model_dict
from datasets import _dataset_dict
from utils.loss import LossRecord
from utils.metrics import Evaluator


class BaseEvaluator:
    def __init__(self, args):
        self.args = args
        self.model_args = args['model']
        self.train_args = args['train']
        self.data_args = args['data']
        
        self.model = self.build_model()
        self.load_model()
        self.build_evaluator()
        torch.manual_seed(self.train_args.get('seed', 42))
        self.device = self.train_args.get('device', 'cuda')
        self.model.to(self.device)

    def build_model(self, **kwargs):
        if self.model_name not in _model_dict:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model

    def load_model(self, **kwargs):
        model_path = os.path.join(self.saving_path, 'best_model.pth')
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(model_path))  
            
    def build_evaluator(self):  
        self.evaluator = Evaluator()
        
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

    def forcast(self, split='valid'):
        if split == 'train':
            data_loader = self.train_loader
        elif split == 'valid':
            data_loader = self.valid_loader
        elif split == 'test':
            data_loader = self.test_loader
        else:
            raise ValueError("Unknown split {}".format(split))
        
        loss_record = self.evaluator.init_record(['{}_loss'.format(split)])
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                pred_y = self.model(x)
                y = self.normalizer.decode(y)
                pred_y = self.normalizer.decode(pred_y)
                self.evaluator(pred_y, y, record=loss_record, batch_size=x.size(0))

        return loss_record
        
    def vis(self):
        pass
