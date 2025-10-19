import torch.nn as nn
from .basics import SpectralConv2d, SpectralUpsampleConv2d
from .utils import _get_act, add_padding2, remove_padding2
import torch
import math


class Sine(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FNO2d(nn.Module):
    def __init__(self, model_params):
        super(FNO2d, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        modes1 = model_params['modes1']
        modes2 = model_params['modes2']
        width = model_params.get('width', 64)
        fc_dim = model_params.get('fc_dim', 128)
        in_dim = model_params.get('in_dim', 3)
        out_dim = model_params.get('out_dim', 1)
        layers = model_params.get('layers', [16, 24, 24, 32, 32])
        act = model_params.get('act', 'gelu')
        pad_ratio = model_params.get('pad_ratio', [0., 0.])
        upsample_factor = model_params.get('upsample_factor', [2, 2])
        
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2
        self.upsample_factor = upsample_factor
    
        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, kernel_size=1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        
        self.up_sp_conv = SpectralUpsampleConv2d(self.layers[0], self.layers[0], self.modes1[-1], self.modes2[-1], upsample_factor=self.upsample_factor)
        # self.upsample_factor[0]//2 当前逻辑： 每次上采样2倍，做upsample_factor[0]//2次
        # self.up_ws = nn.ModuleList([nn.ConvTranspose2d(self.layers[0], self.layers[0], kernel_size=3, stride=2, padding=1, output_padding=1) for _ in range(self.upsample_factor[0]//2)])
        # 正确逻辑： 每次上采样2倍，做log2(upsample_factor[0])次，表示2的多少次方等于upsample_factor[0]
        num_upsample_layers = int(math.log2(self.upsample_factor[0]))
        self.up_ws = nn.ModuleList([nn.ConvTranspose2d(self.layers[0], self.layers[0], kernel_size=3, stride=2, padding=1, output_padding=1) for _ in range(num_upsample_layers)])
        self.up_ws = nn.Sequential(*self.up_ws)
        # self.up_ws = nn.ConvTranspose2d(self.layers[0], self.layers[0], kernel_size=3, stride=self.upsample_factor[0], padding=1, output_padding=1)

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)   # B, C, X, Y
        size_x, size_y = x.shape[-2], x.shape[-1]
        x1 = self.up_sp_conv(x)
        x2 = self.up_ws(x).view(batchsize, self.layers[0], size_x*self.upsample_factor[0], size_y*self.upsample_factor[1])
        x = self.act(x1 + x2)
        # Padding
        x = add_padding2(x, num_pad1, num_pad2)
        size_x, size_y = x.shape[-2], x.shape[-1]
        
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            x = self.act(x)
            
        x = remove_padding2(x, num_pad1, num_pad2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
