import os.path as osp
import scipy.io as sio
import numpy as np

from h5py import File

import torch
from torch.utils.data import Dataset, DataLoader

from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class NavierStokes2DDataset:
    def __init__(self, data_path, sample_factor=[1, 1],
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 subset=None, normalize=True, normalizer_type='PGN', **kwargs):
        self.load_data(data_path=data_path, 
                       train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, 
                       sample_factor=sample_factor,
                       normalize=normalize, normalizer_type=normalizer_type)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path, sample_factor,
                  train_ratio, valid_ratio, test_ratio, 
                  normalize, normalizer_type):
        process_path = data_path.split('.')[0] + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_x, train_y, valid_x, valid_y, test_x, test_y, x_normalizer, y_normalizer = torch.load(process_path)
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
            
            train_x, train_y, x_normalizer, y_normalizer = self.pre_process(data[:train_idx], mode='train', normalize=normalize, normalizer_type=normalizer_type)
            valid_x, valid_y = self.pre_process(data[train_idx:valid_idx], mode='valid', normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
            test_x, test_y = self.pre_process(data[valid_idx:test_idx], mode='test', normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
            print('Saving data...')
            torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, x_normalizer, y_normalizer), process_path)
            print('Data processed and saved to', process_path)

        self.train_dataset = NavierStokes2DBase(train_x, train_y, mode='train', x_normalizer=x_normalizer, y_normalizer=y_normalizer, sample_factor=sample_factor)
        self.valid_dataset = NavierStokes2DBase(valid_x, valid_y, mode='valid', x_normalizer=x_normalizer, y_normalizer=y_normalizer, sample_factor=sample_factor)
        self.test_dataset = NavierStokes2DBase(test_x, test_y, mode='test', x_normalizer=x_normalizer, y_normalizer=y_normalizer, sample_factor=sample_factor)

    def pre_process(self, data, mode='train', normalize=False, 
                    normalizer_type='PGN', x_normalizer=None, y_normalizer=None,
                    **kwargs):
        data = data.permute(0, 3, 1, 2).unsqueeze(-1)
        x = data[:, :-1, :, :, :]
        y = data[:, 1:, :, :, :]
        
        x = x.flatten(start_dim=0, end_dim=1)
        y = y.flatten(start_dim=0, end_dim=1)

        B, H, W, C = x.shape
        
        if normalize:
            x = x.reshape(B, -1, C)
            y = y.reshape(B, -1, C)
            if mode == 'train':
                if normalizer_type == 'PGN':
                    x_normalizer = UnitGaussianNormalizer(x)
                    y_normalizer = UnitGaussianNormalizer(y)
                else:
                    x_normalizer = GaussianNormalizer(x)
                    y_normalizer = GaussianNormalizer(y)
                x = x_normalizer.encode(x)
                y = y_normalizer.encode(y)
            else:
                x = x_normalizer.encode(x)
                y = y_normalizer.encode(y)
            x = x.reshape(B, H, W, C)
            y = y.reshape(B, H, W, C)
        else:
            x_normalizer = None
            y_normalizer = None

        grid_x = torch.linspace(0, 1, H)
        grid_x = grid_x.reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        grid_y = torch.linspace(0, 1, W)
        grid_y = grid_y.reshape(1, 1, W, 1).repeat(B, W, 1, 1)
        
        x = torch.cat((grid_x, grid_y, x), dim=-1)
        
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        
        if mode == 'train':
            return x, y, x_normalizer, y_normalizer
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

    def __init__(self, x, y, mode='train', 
                 x_normalizer=None, y_normalizer=None, sample_factor=[1, 1],
                 **kwargs):
        self.mode = mode
        self.x = x[:, ::sample_factor[0], ::sample_factor[1], -1:]
        self.y = x[..., -1:]
        self.x_normalizer = x_normalizer
        self.y_normalizer = x_normalizer

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
