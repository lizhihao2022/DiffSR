import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .utils import get_filter


class LPFOperator2d(nn.Module):
    def __init__(self, k=4, c=4, base='legendre', bias=True, padding_mode='zeros'):
        super(LPFOperator2d, self).__init__()
        self.c = c
        self.k = k
        self.hidden_channel = c * (k ** 2)
        self.fine_dim = self.hidden_channel
        self.corse_dim = self.hidden_channel // 4
        self.sub_dim = self.corse_dim * 3
        self.H, self.G = self.get_filter(base, k)
        
    def get_filter(self, base, k):
        H0, H1, G0, G1, _, _ = get_filter(base, k)
        
        H = torch.Tensor(np.concatenate((np.kron(H0, H0),
                                         np.kron(H0, H1),
                                         np.kron(H1, H0),
                                         np.kron(H1, H1),), axis=1))
        GH = torch.Tensor(np.concatenate((np.kron(G0, H0),
                                          np.kron(G0, H1),
                                          np.kron(G1, H0),
                                          np.kron(G1, H1),), axis=1))
        HG = torch.Tensor(np.concatenate((np.kron(H0, G0),
                                          np.kron(H0, G1),
                                          np.kron(H1, G0),
                                          np.kron(H1, G1),), axis=1))
        GG = torch.Tensor(np.concatenate((np.kron(G0, G0),
                                          np.kron(G0, G1),
                                          np.kron(G1, G0),
                                          np.kron(G1, G1),), axis=1))
        G = torch.cat((GH, HG, GG), dim=0)
        
        return H, G
    
    def restrict(self, x):
        if x.device.type == 'cuda':
            self.H = self.H.cuda()
            self.G = self.G.cuda()
        else:
            self.H = self.H.cpu()
            self.G = self.G.cpu()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1]//2, x.shape[2]//2, self.c, -1)
        x = torch.matmul(x, self.H.T)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        x = x.permute(0, 3, 1, 2)
        
        return x

    def prolongate(self, x):
        if x.device.type == 'cuda':
            self.H = self.H.cuda()
            self.G = self.G.cuda()
        else:
            self.H = self.H.cpu()
            self.G = self.G.cpu()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.c, -1)
        x = torch.matmul(x, self.H)
        x = x.reshape(x.shape[0], x.shape[1]*2, x.shape[2]*2, -1)
        x = x.permute(0, 3, 1, 2)
        
        return x


class FourierOperator(nn.Module):
    def __init__(self, c=4, k=4, **kwargs):
        super(FourierOperator, self).__init__()
        
    def restrict(self, x):
        size_1, size_2 = x.shape[2], x.shape[3]
        x_ft = torch.fft.rfftn(x, dim=(2, 3))
        
        x = torch.fft.irfftn(x_ft, s=(size_1//2, size_2//2), dim=(2, 3))
        return x
    
    def prolongate(self, x):
        size_1, size_2 = x.shape[2], x.shape[3]
        x_ft = torch.fft.rfftn(x, dim=(2, 3))
        x = torch.fft.irfftn(x_ft, s=(size_1*2, size_2*2), dim=(2, 3))
        return x


class VanillaOperator(nn.Module):
    def __init__(self, c=4, k=4, **kwargs):
        super(VanillaOperator, self).__init__()
        
    def restrict(self, x):
        return x[:, :, ::2, ::2]
    
    def prolongate(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x
