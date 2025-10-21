import os
import torch
import wandb
import logging

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from utils.loss import LossRecord, LpLoss
from utils.helper import save_code
from utils.ddp import debug_barrier
from utils.metrics import Evaluator
from functools import partial
from models import _model_dict
from datasets import _dataset_dict


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model_args = args['model']
        self.data_args = args['data']
        self.optim_args = args['optimize']
        self.scheduler_args = args['schedule']
        self.train_args = args['train']
        self.log_args = args['log']
        
        self.set_distribute()

        self.logger = logging.info if self.log_args.get('log', True) else print
        self.wandb = self.log_args.get('wandb', False)
        if self.check_main_process() and self.wandb:
            wandb.init(
                project=self.log_args.get('wandb_project', 'default'), 
                name=self.train_args.get('saving_name', 'experiment'),
                tags=[self.model_args.get('name', 'model'), self.data_args.get('name', 'dataset')],
                config=args)
        
        self.model_name = self.model_args['name']
        self.main_log("Building {} model".format(self.model_name))
        self.model = self.build_model()
        self.apply_init()
        
        self.start_epoch = 0
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        

        if self.train_args.get('load_ckpt', False):
            self.load_ckpt(self.train_args['ckpt_path'])
        


        self.model = self.model.to(self.device)
        
        if self.dist:
            if self.dist_mode == 'DP':
                self.device_ids = self.train_args.get('device_ids', range(torch.cuda.device_count()))
                self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
                self.main_log("Using DataParallel with GPU: {}".format(self.device_ids))
            elif self.dist_mode == 'DDP':
                self.local_rank = self.train_args.get('local_rank', 0)
                torch.cuda.set_device(self.local_rank)
                self.model = self.model.to(self.local_rank)
                self.model = DDP(
                    self.model, 
                    device_ids=[self.local_rank], 
                    output_device=self.local_rank)
        
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()
        
        self.loss_fn = self.build_loss()
        self.evaluator = self.build_evaluator()

        self.main_log("Model: {}".format(self.model))
        self.main_log("Model parameters: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        self.main_log("Optimizer: {}".format(self.optimizer))
        self.main_log("Scheduler: {}".format(self.scheduler))

        self.data = self.data_args['name']
        self.main_log("Loading {} dataset".format(self.data))
        self.build_data()
        self.main_log("Train dataset size: {}".format(len(self.train_loader.dataset)))
        self.main_log("Valid dataset size: {}".format(len(self.valid_loader.dataset)))
        self.main_log("Test dataset size: {}".format(len(self.test_loader.dataset)))

        self.epochs = self.train_args['epochs']
        self.eval_freq = self.train_args['eval_freq']
        self.patience = self.train_args['patience']
        
        self.saving_best = self.train_args.get('saving_best', True)
        self.saving_ckpt = self.train_args.get('saving_ckpt', False)
        self.ckpt_freq = self.train_args.get('ckpt_freq', 100)
        self.ckpt_max = self.train_args.get('ckpt_max', 5)
        self.saving_path = self.train_args.get('saving_path', None)

    def _unwrap(self):
        if isinstance(self.model, (DDP, nn.DataParallel)):
            return self.model.module
        return self.model
    
    def set_distribute(self):
        self.dist = self.train_args.get('distribute', False)
        if self.dist:
            self.dist_mode = self.train_args.get('distribute_mode', 'DDP')
        if self.dist and self.dist_mode == 'DDP':
            self.local_rank = self.train_args.get('local_rank', 0)
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def get_initializer(self, name):
        if name is None:
            return None
        
        if name == 'xavier_normal':
            init_ = partial(torch.nn.init.xavier_normal_)
        elif name == 'kaiming_uniform':
            init_ = partial(torch.nn.init.kaiming_uniform_)
        elif name == 'kaiming_normal':
            init_ = partial(torch.nn.init.kaiming_normal_)
        return init_

    def apply_init(self, **kwargs):
        initializer = self.get_initializer(self.train_args.get('initializer', None))
        if initializer is not None:
            self.model.apply(initializer)
            self.main_log("Apply {} initializer".format(self.train_args.get('initializer', None)))
    
    def build_optimizer(self, **kwargs):
        if self.optim_args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                momentum=self.optim_args['momentum'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(self.optim_args['optimizer']))
        return optimizer
    
    def build_scheduler(self, **kwargs):
        if self.scheduler_args['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_args['milestones'],
                gamma=self.scheduler_args['gamma'],
            )
        elif self.scheduler_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optim_args['lr'],
                div_factor=self.scheduler_args['div_factor'],
                final_div_factor=self.scheduler_args['final_div_factor'],
                pct_start=self.scheduler_args['pct_start'],
                steps_per_epoch=self.scheduler_args['steps_per_epoch'],
                epochs=self.train_args['epochs'],
            )
        elif self.scheduler_args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_args['step_size'],
                gamma=self.scheduler_args['gamma'],
            )
        else:
            scheduler = None
            if self.scheduler_args['scheduler'] is not None:
                raise NotImplementedError("Scheduler {} not implemented".format(self.scheduler_args['scheduler']))
            
        return scheduler
    
    def build_model(self, **kwargs):
        if self.model_name not in _model_dict:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model
    
    def build_loss(self, **kwargs):
        loss_fn = LpLoss(size_average=False)
        return loss_fn
    
    def build_evaluator(self):
        return Evaluator(shape=self.data_args['shape'])
    
    def build_data(self, **kwargs):
        if self.data_args['name'] not in _dataset_dict:
            raise NotImplementedError("Dataset {} not implemented".format(self.data_args['name']))
        dataset = _dataset_dict[self.data_args['name']](self.data_args)
        if self.dist and self.dist_mode == 'DDP':
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset.train_dataset,
                shuffle=True,
                drop_last=True,
                )
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True
        
        self.train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=self.data_args.get('train_batchsize', 10),
            shuffle=shuffle,
            num_workers=self.data_args.get('num_workers', 0),
            sampler=self.train_sampler,
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
        
        self.normalizer = dataset.normalizer
    
    def _get_state_dict_cpu(self):
        if self.dist and self.dist_mode == 'DDP':
            model_to_save = self.model.module
        elif isinstance(self.model, torch.nn.DataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        return {k: v.detach().cpu() for k, v in model_to_save.state_dict().items()}
    
    def save_ckpt(self, epoch):
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        state_dict_cpu = self._get_state_dict_cpu()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict_cpu,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(self.saving_path, f"model_epoch_{epoch}.pth"))
        if self.ckpt_max is not None and self.ckpt_max > 0:
            ckpt_list = [f for f in os.listdir(self.saving_path) if f.startswith('model_epoch_') and f.endswith('.pth')]
            if len(ckpt_list) > self.ckpt_max:
                ckpt_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                os.remove(os.path.join(self.saving_path, ckpt_list[0]))
                    
    def save_model(self, model_path):
        state_dict_cpu = self._get_state_dict_cpu()
        torch.save(state_dict_cpu, model_path)
        self.main_log("Save model to {}".format(model_path))
        
    def load_model(self, model_path):
        state = torch.load(model_path, map_location="cpu")
        if self.dist and self.dist_mode == 'DDP':
            self.model.module.load_state_dict(state)
        elif isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        self.main_log("Load model from {}".format(model_path))
    
    def load_ckpt(self, ckpt_path):        
        state = torch.load(ckpt_path, map_location="cpu")
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            # ✅ 强制把optimizer中的状态迁移到GPU
            for state_tensor in self.optimizer.state.values():
                for k, v in state_tensor.items():
                    if isinstance(v, torch.Tensor):
                        state_tensor[k] = v.to(self.device)
        if 'scheduler_state_dict' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.start_epoch = state.get('epoch', 0) + 1
        self.main_log("Load checkpoint from {}, epoch {}".format(ckpt_path, state.get('epoch', 'N/A')))
    
    def check_main_process(self):
        if self.dist is False:
            return True
        if self.dist_mode == 'DP':
            return True
        if self.local_rank == 0:
            return True
        return False
    
    def main_log(self, msg):
        if self.check_main_process():
            self.logger(msg)
    
    def process(self, **kwargs):
        self.main_log("Start training")
        best_epoch = 0
        best_metrics = None
        best_path = os.path.join(self.saving_path, "best_model.pth")
        counter = 0
        if dist.is_initialized():
            dist.barrier()
        bar = tqdm(total=self.epochs - self.start_epoch) if self.check_main_process() else None

        for epoch in range(self.start_epoch, self.epochs):
            train_loss_record = self.train(epoch)
            self.main_log("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, self.optimizer.param_groups[0]["lr"]))
            if self.check_main_process() and self.wandb:
                wandb.log(train_loss_record.to_dict())
            
            if self.check_main_process() and self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                self.save_ckpt(epoch)
                self.main_log("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))

            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(split="valid")
                self.main_log("Epoch {} | {}".format(epoch, valid_loss_record))
                valid_metrics = valid_loss_record.to_dict()
                if self.check_main_process() and self.wandb:
                    wandb.log(valid_loss_record.to_dict())
                
                if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
                    counter = 0
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    if self.check_main_process() and self.saving_best:
                        self.save_model(best_path)
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        self.main_log("Early stop at epoch {}".format(epoch))
                        if not self.dist:
                            break
                        stop_flag = torch.tensor(0, device=self.device)
                        if self.check_main_process():
                            if self.patience != -1 and counter >= self.patience:
                                stop_flag += 1
                        if self.dist and dist.is_initialized():
                            dist.broadcast(stop_flag, src=0)
                        if stop_flag.item() > 0:
                            break                       
            if self.check_main_process():
                bar.update(1)
        if self.check_main_process():
            if bar is not None:
                bar.close()
        self.main_log("Optimization Finished!")
        
        if self.check_main_process() and not best_metrics:
            self.save_model(best_path)
        
        if self.dist and dist.is_initialized():
            dist.barrier()
        
        self.load_model(best_path)

        valid_loss_record = self.evaluate(split="valid")
        self.main_log("Valid metrics: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(split="test")
        self.main_log("Test metrics: {}".format(test_loss_record))

        if self.check_main_process() and self.wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())
            wandb.finish()
        
        if self.dist and dist.is_initialized():
            dist.barrier()

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss_record.update({"train_loss": loss.sum().item()}, n=x.size(0))
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record
    
    def inference(self, x, y, **kwargs):
        return self.model(x).reshape(y.shape)
    
    def evaluate(self, split="valid", **kwargs):
        if split == "valid":
            eval_loader = self.valid_loader
        elif split == "test":
            eval_loader = self.test_loader
        else:
            raise ValueError("split must be 'valid' or 'test'")
        
        loss_record = self.evaluator.init_record(["{}_loss".format(split)])
        all_y = []
        all_y_pred = []
        self.model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                y_pred = self.normalizer.decode(y_pred)
                y = self.normalizer.decode(y)
                all_y.append(y)
                all_y_pred.append(y_pred)
        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        loss = self.loss_fn(y_pred, y)
        total_samples = y.size(0)
        loss_record.update({"{}_loss".format(split): loss.item()}, n=total_samples)
        self.evaluator(y_pred, y, record=loss_record)
        if self.dist and dist.is_initialized():
            loss_record.dist_reduce()
        return loss_record
