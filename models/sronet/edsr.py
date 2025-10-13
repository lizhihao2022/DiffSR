# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, res_scale=1,
                 scale=2, no_upsampling=True, 
                 input_channels=3, output_channels=3,
                 conv=default_conv):
        """
        EDSR 编码器
        
        Args:
            n_resblocks: 残差块数量
            n_feats: 特征通道数
            res_scale: 残差缩放因子
            scale: 上采样倍数 (当 no_upsampling=False 时使用)
            no_upsampling: 是否不使用上采样 (在 SRNO 中设为 True)
            input_channels: 输入通道数
            output_channels: 输出通道数
        """
        super(EDSR, self).__init__()
        
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.no_upsampling = no_upsampling
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        # Head: 初始卷积
        m_head = [conv(input_channels, n_feats, kernel_size)]

        # Body: 残差块堆叠
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if no_upsampling:
            self.out_dim = n_feats
        else:
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, output_channels, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)
            self.out_dim = output_channels

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W] 输入
        Returns:
            x: [B, n_feats, H, W] 特征 (no_upsampling=True)
               或 [B, output_channels, H', W'] (no_upsampling=False)
        """
        x = self.head(x)
        res = self.body(x)
        res += x

        if self.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x


# @register('edsr-baseline')
# def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
#                        scale=2, no_upsampling=False, rgb_range=1):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 3
#     return EDSR(args)


# @register('edsr')
# def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
#               scale=2, no_upsampling=False, rgb_range=1):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 3
#     return EDSR(args)
