import torch
import os.path as osp
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer
from utils.sr import make_lr_blur


class ERA5WindDataset:
    def __init__(self, data_args, **kwargs):        
        data_path = data_args['data_path']
        sample_factor = data_args.get('sample_factor', 2)
        train_batchsize = data_args.get('train_batchsize', 10)
        eval_batchsize = data_args.get('eval_batchsize', 10)
        train_ratio = data_args.get('train_ratio', 0.8)
        valid_ratio = data_args.get('valid_ratio', 0.1)
        test_ratio = data_args.get('test_ratio', 0.1)
        subset = data_args.get('subset', None)
        normalize = data_args.get('normalize', True)
        normalizer_type = data_args.get('normalizer_type', 'PGN')

        process_path = data_path.split('.')[0] + str(sample_factor) + '_sr.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer = torch.load(process_path)
        else:
            print('Processing data...')
            # Load ERA5 wind data from .npy file
            # Expected shape: [B, 512, 512, 2] where channels are u and v components
            raw_data = np.load(data_path)
            data = torch.tensor(raw_data, dtype=torch.float32)  # [B, H, W, C]
            
            data_size = data.shape[0]
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            train_x, train_y, normalizer = self.pre_process(data[:train_idx], mode='train', normalize=normalize, normalizer_type=normalizer_type, sample_factor=sample_factor)
            valid_x, valid_y = self.pre_process(data[train_idx:valid_idx], mode='valid', normalize=normalize, normalizer=normalizer, sample_factor=sample_factor)
            test_x, test_y = self.pre_process(data[valid_idx:test_idx], mode='test', normalize=normalize, normalizer=normalizer, sample_factor=sample_factor)
            print('Saving data...')
            torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer), process_path)
            print('Data processed and saved to', process_path)

        self.normalizer = normalizer
        self.train_dataset = ERA5WindBase(train_x, train_y, mode='train')
        self.valid_dataset = ERA5WindBase(valid_x, valid_y, mode='valid')
        self.test_dataset = ERA5WindBase(test_x, test_y, mode='test')

    def pre_process(self, data, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, sample_factor=2, **kwargs):
        # data shape: (B, H, W, C) where C=2 (u, v components)
        B, H, W, C = data.shape
        
        if normalize:
            # Reshape for normalization: (B, H*W, C)
            data = data.reshape(B, -1, C)
            if mode == 'train':
                if normalizer_type == 'PGN':
                    normalizer = UnitGaussianNormalizer(data)
                else:
                    normalizer = GaussianNormalizer(data)
                data = normalizer.encode(data)
            else:
                data = normalizer.encode(data)
            data = data.reshape(B, H, W, C)
        else:
            normalizer = None

        # Generate low-resolution input by downsampling
        # x: low-resolution (B, h, w, C), y: high-resolution (B, H, W, C)
        x = make_lr_blur(data.permute(0, 3, 1, 2), scale=sample_factor, ks=7, sigma=1.2, down_mode='bicubic').permute(0, 2, 3, 1)
        y = data
                
        if mode == 'train':
            return x, y, normalizer
        else:
            return x, y


class ERA5WindBase(Dataset):
    """
    A base class for the ERA5 wind dataset.

    Args:
        x (torch.Tensor): The low-resolution input data (B, h, w, 2).
        y (torch.Tensor): The high-resolution target data (B, H, W, 2).
        mode (str, optional): The mode of the dataset ('train', 'valid', or 'test'). Defaults to 'train'.
        **kwargs: Additional keyword arguments.

    Attributes:
        mode (str): The mode of the dataset.
        x (torch.Tensor): The low-resolution input data with u and v wind components.
        y (torch.Tensor): The high-resolution target data with u and v wind components.
    """

    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]