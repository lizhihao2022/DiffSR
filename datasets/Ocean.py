import torch
import os.path as osp
import numpy as np
from h5py import File
from torch.utils.data import Dataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer
from utils.sr import make_lr_blur


class OceanDataset:
    """
    Ocean surface velocity dataset for super-resolution.
    
    Args:
        data_args (dict): Dictionary containing:
            - data_path (str): Path to HDF5 file
            - data_key (str): 'uo_data' or 'vo_data'
            - sample_factor (int): Downsampling factor for LR generation
            - train_batchsize (int): Training batch size
            - eval_batchsize (int): Evaluation batch size
            - train_ratio (float): Training set ratio
            - valid_ratio (float): Validation set ratio
            - test_ratio (float): Test set ratio
            - normalize (bool): Whether to normalize data
            - normalizer_type (str): 'PGN' or 'GN'
    """
    
    def __init__(self, data_args, **kwargs):        
        data_path = data_args['data_path']
        data_key = data_args.get('data_key', 'uo_data')  # 'uo_data' or 'vo_data'
        sample_factor = data_args.get('sample_factor', 2)
        train_batchsize = data_args.get('train_batchsize', 10)
        eval_batchsize = data_args.get('eval_batchsize', 10)
        train_ratio = data_args.get('train_ratio', 0.8)
        valid_ratio = data_args.get('valid_ratio', 0.1)
        test_ratio = data_args.get('test_ratio', 0.1)
        normalize = data_args.get('normalize', True)
        normalizer_type = data_args.get('normalizer_type', 'PGN')

        # 根据data_key生成不同的缓存文件名
        process_path = data_path.split('.')[0] + f'_{data_key}_sf{sample_factor}_sr.pt'
        
        if osp.exists(process_path):
            print(f'Loading processed {data_key} data from {process_path}')
            train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer = torch.load(process_path)
        else:
            print(f'Processing {data_key} data from HDF5...')
            # 从HDF5文件读取数据
            with File(data_path, 'r') as f:
                if data_key not in f.keys():
                    raise ValueError(f"Key '{data_key}' not found in HDF5 file. Available keys: {list(f.keys())}")
                
                # 读取数据: shape为 [N, 128, 128, 1]
                raw_data = f[data_key][:]
                data = torch.tensor(raw_data, dtype=torch.float32)
                print(f'Loaded {data_key} with shape: {data.shape}')
            
            data_size = data.shape[0]
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            print(f'Dataset split: Train={train_idx}, Valid={valid_idx-train_idx}, Test={test_idx-valid_idx}')
            
            train_x, train_y, normalizer = self.pre_process(
                data[:train_idx], 
                mode='train', 
                normalize=normalize, 
                normalizer_type=normalizer_type, 
                sample_factor=sample_factor
            )
            valid_x, valid_y = self.pre_process(
                data[train_idx:valid_idx], 
                mode='valid', 
                normalize=normalize, 
                normalizer=normalizer, 
                sample_factor=sample_factor
            )
            test_x, test_y = self.pre_process(
                data[valid_idx:test_idx], 
                mode='test', 
                normalize=normalize, 
                normalizer=normalizer, 
                sample_factor=sample_factor
            )
            
            print(f'Saving processed data to {process_path}...')
            torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer), process_path)
            print(f'Data processed and saved!')

        self.normalizer = normalizer
        self.train_dataset = OceanDatasetBase(train_x, train_y, mode='train')
        self.valid_dataset = OceanDatasetBase(valid_x, valid_y, mode='valid')
        self.test_dataset = OceanDatasetBase(test_x, test_y, mode='test')

    def pre_process(self, data, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, sample_factor=2, **kwargs):
        """
        预处理数据
        
        Args:
            data: [N, H, W, C] shape的数据
        
        Returns:
            x: 低分辨率输入 [N, h, w, C]
            y: 高分辨率目标 [N, H, W, C]
            normalizer: 归一化器 (仅在mode='train'时返回)
        """
        # data已经是 [N, H, W, C] 格式
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

        # 生成低分辨率输入
        x = make_lr_blur(
            data.permute(0, 3, 1, 2),  # [B, C, H, W]
            scale=sample_factor, 
            ks=7, 
            sigma=1.2, 
            down_mode='bicubic'
        ).permute(0, 2, 3, 1)  # [B, h, w, C]
        
        y = data  # [B, H, W, C]
        
        print(f'[{mode}] Generated LR: {x.shape}, HR: {y.shape}')
                
        if mode == 'train':
            return x, y, normalizer
        else:
            return x, y


class OceanDatasetBase(Dataset):
    """
    Base class for Ocean velocity dataset.

    Args:
        x (Tensor): Low-resolution input data [N, h, w, C]
        y (Tensor): High-resolution target data [N, H, W, C]
        mode (str): 'train', 'valid', or 'test'
    """

    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]