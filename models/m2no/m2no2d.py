import torch.nn as nn

from .grid_operator import LPFOperator2d, FourierOperator, VanillaOperator


class GridBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_ite, bias=True, padding_mode='zeros'):
        super(GridBlock2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.num_ite = num_ite  
        
        self.S = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode) for _ in range(num_ite)])
        # self.A = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode) for _ in range(num_ite)])

    def forward(self, A, f, u=None):           
        for i in range(self.num_ite):
            if u is None:
                u = self.S[i](f)
            else:
                u = u + self.S[i](f - A(u))
            r = f - A(u)
            
        return u, r


class MultiGrid2d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_levels, op, bias=True, 
                 padding_mode='zeros', resolutions=[64, 64], norm=False, **kwargs):
        super(MultiGrid2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_levels = grid_levels
        self.num_level = len(grid_levels)
        self.norm = norm

        self.A = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode)
        self.pre_S = GridBlock2d(in_channels, out_channels, 1, bias=bias, padding_mode=padding_mode)
        self.post_S = GridBlock2d(in_channels, out_channels, 1, bias=bias, padding_mode=padding_mode)
        self.grid_list = nn.ModuleList([GridBlock2d(in_channels, out_channels, grid_levels[i]) for i in range(self.num_level)])
        if self.norm:
            self.norm_list = nn.ModuleList([nn.LayerNorm([out_channels, resolutions[0]//2**i, resolutions[1]//2**i]) for i in range(self.num_level)])
        self.op = op
        self.resolutions = resolutions

    def forward(self, f):
        # input_resolution = f.shape[2]
        
        while f.shape[2] > self.resolutions[0]:
            f = self.op.restrict(f)
        while f.shape[2] < self.resolutions[0]:
            f = self.op.prolongate(f)
        
        u_list = [None] * (self.num_level + 1)
        r_list = [None] * (self.num_level + 1)
        
        u_n, r_n = self.pre_S(self.A, f)

        # while u_n.shape[2] > self.resolutions[0]:
        #     r_n = self.op.restrict(r_n)
        #     u_n = self.op.restrict(u_n)
        #     f = self.op.restrict(f)
        
        # while u_n.shape[2] < self.resolutions[0]:
        #     r_n = self.op.prolongate(r_n)
        #     u_n = self.op.prolongate(u_n)
        #     f = self.op.prolongate(f)
                
        r_list[0] = r_n
        u_list[0] = u_n
        
        for i in range(self.num_level):
            u = self.op.restrict(u_list[i])
            u, r = self.grid_list[i](self.A, u)
            r_list[i+1] = r
            u_list[i+1] = u

        for i in range(self.num_level, 0, -1):
            if self.norm:
                u_list[i-1] = self.norm_list[i-1](u_list[i-1] + self.op.prolongate(u_list[i]))
            else:
                u_list[i-1] = u_list[i-1] + self.op.prolongate(u_list[i])
        
        u_n = u_list[0]
        
        # while u_n.shape[2] > input_resolution:
        #     u_n = self.op.restrict(u_n)
        
        # while u_n.shape[2] < input_resolution:
        #     u_n = self.op.prolongate(u_n)
        
        u_n, r_n = self.post_S(self.A, f, u_n)
        
        return u_n

class M2NO2d(nn.Module):
    def __init__(self, model_args, **kwargs):
        super(M2NO2d, self).__init__()
        
        in_channels = model_args['in_channels']
        out_channels = model_args['out_channels']
        k = model_args['k']
        c = model_args['c']
        num_layer = model_args['num_layer']
        grid_levels = model_args['grid_levels']
        base = model_args.get('base', 'legendre')
        bias = model_args.get('bias', True)
        padding_mode = model_args.get('padding_mode', 'zeros')
        norm = model_args.get('norm', False)
        resolutions = model_args.get('resolutions', [64, 64])
        activation = model_args.get('activation', 'gelu')
        
        self.num_layer = num_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = c * (k ** 2)

        self.L_in = nn.Linear(in_channels, hidden_channels)
        self.L_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.L_out = nn.Linear(hidden_channels, out_channels)
        
        self.wavelet_operator = LPFOperator2d(k=k, c=c, base=base, bias=bias, padding_mode=padding_mode)
        # self.wavelet_operator = FourierOperator(k=k, c=c, base=base, bias=bias, padding_mode=padding_mode)
        # self.wavelet_operator = VanillaOperator(k=k, c=c, base=base, bias=bias, padding_mode=padding_mode)
        
        self.conv_list = nn.ModuleList([MultiGrid2d(hidden_channels, hidden_channels, grid_levels, self.wavelet_operator, norm=norm,
                                                  bias=bias, padding_mode=padding_mode, resolutions=resolutions) for _ in range(num_layer)])
        if activation == 'gelu':
            self.activate = nn.GELU()
        elif activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'tanh':
            self.activate = nn.Tanh()
        else:
            raise ValueError('Activation not supported')
        
    def forward(self, x):
        u = self.L_in(x)
        u = u.permute(0, 3, 1, 2)
        for i in range(self.num_layer):
            u = self.conv_list[i](u)
            if i != self.num_layer-1:
                u = self.activate(u)
        u = u.permute(0, 2, 3, 1)
        u = self.activate(self.L_hidden(u))
        u = self.L_out(u)

        return u
