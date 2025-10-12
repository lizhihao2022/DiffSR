import torch

import os.path as osp
import scipy.io as sio
import numpy as np

from h5py import File
from torch.utils.data import Dataset, DataLoader
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer
from utils.sr import make_lr_blur


class NavierStokes2DDataset:
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
            try:
                raw_data = sio.loadmat(data_path)
                data = torch.tensor(raw_data['u'], dtype=torch.float32)
            except:
                raw_data = File(data_path, 'r')
                data = torch.tensor(np.transpose(raw_data['u'], (3, 1, 2, 0)), dtype=torch.float32)
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
        self.train_dataset = NavierStokes2DBase(train_x, train_y, mode='train')
        self.valid_dataset = NavierStokes2DBase(valid_x, valid_y, mode='valid')
        self.test_dataset = NavierStokes2DBase(test_x, test_y, mode='test')

    def pre_process(self, data, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, sample_factor=2, **kwargs):
        data = data.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(-1) # (B, H, W, C)
        B, H, W, C = data.shape
        
        if normalize:
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

        x = make_lr_blur(data.permute(0, 3, 1, 2), scale=sample_factor, ks=7, sigma=1.2, down_mode='bicubic').permute(0, 2, 3, 1) # (B, h, w, C)
        y = data
                
        if mode == 'train':
            return x, y, normalizer
        else:
            return x, y


class NavierStokes2DBase(Dataset):
    """
    A base class for the Navier-Stokes dataset.

    Args:
        x (list): The input data.
        y (list): The target data.
        mode (str, optional): The mode of the dataset. Defaults to 'train'.
        **kwargs: Additional keyword arguments.

    Attributes:
        mode (str): The mode of the dataset.
        x (list): The input data.
        y (list): The target data.
    """

    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
