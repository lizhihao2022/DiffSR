import torch
from torch.utils.data import DataLoader, Dataset
import os.path as osp

from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class ERA5Dataset:
    def __init__(self, data_path, raw_resolution=[721, 1440, 24], 
                 sample_resolution=[721, 1440, 24], eval_resolution=[721, 1440, 24], 
                 in_t=1, out_t=1, duration_t=23, 
                 train_day=12, valid_day=4, test_day=4,
                 train_batchsize=10, eval_batchsize=10, 
                 normalize=True, normalizer_type='PGN', 
                 prop='temp', sub=False,
                 **kwargs):
        process_path = data_path.split('.')[0] + '_' + prop + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y), x_normalizer, y_normalizer = torch.load(process_path)
        else:
            print('Processing raw data from ', data_path)
            data = torch.load(data_path)
            
            train_x, train_y, x_normalizer, y_normalizer = self.pre_process(data[:train_day], mode='train', prop=prop,
                                                in_t=in_t, out_t=out_t, duration_t=duration_t,
                                                normalize=normalize, normalizer_type=normalizer_type)
            valid_x, valid_y = self.pre_process(data[-test_day-valid_day:-test_day], mode='valid', prop=prop,
                                                in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                                normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
            test_x, test_y = self.pre_process(data[-test_day:], mode='test', prop=prop,
                                              in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                              normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
            torch.save(((train_x, train_y), (valid_x, train_y), (test_x, test_y), x_normalizer, y_normalizer), process_path)

        if sub is not False:
            sub_index = int(len(train_x) * sub)
            train_x = train_x[:sub_index]
            train_y = train_y[:sub_index]
        
        self.train_dataset = ERA5Base(train_x, train_y, mode='train', 
                                      raw_resolution=raw_resolution, sample_resolution=sample_resolution,
                                      x_normalizer=x_normalizer, y_normalizer=y_normalizer)
        self.valid_dataset = ERA5Base(valid_x, valid_y, mode='valid', 
                                      raw_resolution=raw_resolution, sample_resolution=eval_resolution,
                                      x_normalizer=x_normalizer, y_normalizer=y_normalizer)
        self.test_dataset = ERA5Base(test_x, test_y, mode='test', 
                                     raw_resolution=raw_resolution, sample_resolution=eval_resolution,
                                     x_normalizer=x_normalizer, y_normalizer=y_normalizer)
                
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
    
    def pre_process(self, data, in_t, out_t, duration_t, mode='train', prop='temp',
                    normalize=False, normalizer_type='PGN', x_normalizer=None, y_normalizer=None,
                    **kwargs):
        if prop == 'temp':
            data = data[:, :, :, :, 0:1]
        elif prop == 'wind_u':
            data = data[:, :, :, :, 1:2]
        elif prop == 'wind_v':
            data = data[:, :, :, :, 2:3]
        elif prop == 'vel':
            data = data[:, :, :, :, 3:4]
        else:
            raise ValueError("Invalid property type.")

        if mode == 'train':
            x = data[:, :in_t, :, :, :]
            y = data[:, in_t:in_t+1, :, :, :]
            for i in range(1, duration_t):
                x = torch.cat((x, data[:, i:in_t+i, :, :, :]), dim=0)
                y = torch.cat((y, data[:, in_t+i:in_t+i+1, :, :, :]), dim=0)
        else:
            x = data[:, out_t-in_t:out_t, :, :, :]
            y = data[:, out_t:out_t+1, :, :, :]
            for i in range(1, duration_t):
                x = torch.cat((x, data[:, out_t+i-in_t:out_t+i, :, :, :]), dim=0)
                y = torch.cat((y, data[:, out_t+i:out_t+i+1, :, :, :]), dim=0)
        x = x.squeeze(1)
        y = y.squeeze(1)

        B, H, W, C = x.shape
        grid_x = torch.linspace(90, -90, H)
        grid_y = torch.linspace(0, 360, W)

        if normalize:
            x = x.view(B, -1, C)
            y = y.view(B, -1, C)
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
            grid_x = (grid_x - (-90)) / (180)
            grid_y = (grid_y - 0) / (360)
            x = x.view(B, H, W, C)
            y = y.view(B, H, W, C)
        else:
            x_normalizer = None
            y_normalizer = None
        
        grid_x = grid_x.reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        grid_y = grid_y.reshape(1, 1, W, 1).repeat(B, H, 1, 1)

        x = torch.cat([grid_x, grid_y, x], dim=-1)
        
        if mode == 'train':
            return x, y, x_normalizer, y_normalizer
        else:
            return x, y


class ERA5Base(Dataset):
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
                 raw_resolution=[512, 512, 20], sample_resolution=[512, 512, 20], 
                 x_normalizer=None, y_normalizer=None, 
                 **kwargs):
        self.mode = mode
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        sample_factor_0 = raw_resolution[0] // sample_resolution[0]
        sample_factor_1 = raw_resolution[1] // sample_resolution[1]
        
        self.x = x[:, ::sample_factor_0, ::sample_factor_1, :]
        self.y = y[:, ::sample_factor_0, ::sample_factor_1, :]
        
        self.x = self.x.view(self.x.shape[0], -1, self.x.shape[-1])
        self.y = self.y.view(self.y.shape[0], -1, self.y.shape[-1])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
