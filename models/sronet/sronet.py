import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from .galerkin import simple_attn
from .edsr import EDSR
from .utils import make_coord


class SRNO(nn.Module):
    """
    Super-Resolution Neural Operator for Navier-Stokes equations
    
    Input: [B, H, W, C]
    Output: [B, H*scale, W*scale, out_dim]
    """

    def __init__(self, model_params):  # 【修改】统一使用 model_params
        super().__init__()
        """
        Args:
            model_params: dict containing:
                - input_channels: int, number of input channels (physics field only), default: 1
                - output_channels: int, number of output channels, default: 1
                - use_coord_input: bool, whether input contains coordinates, default: True
                - width: int, Galerkin Transformer width, default: 256
                - blocks: int, number of Galerkin blocks, default: 16
                - upsample_factor: list of int or int, upsampling scale, default: [2, 2]
                - encoder_config: dict, EDSR encoder configuration
        """
        # 【修改】从 model_params 读取所有参数
        self.input_channels = model_params.get('input_channels', 1)
        self.output_channels = model_params.get('output_channels', 1)
        self.use_coord_input = model_params.get('use_coord_input', True)
        self.width = model_params.get('width', 256)
        blocks = model_params.get('blocks', 16)
        
        # 【修改】上采样倍数支持动态配置
        upsample_factor = model_params.get('upsample_factor', [2, 2])
        if isinstance(upsample_factor, int):
            upsample_factor = [upsample_factor, upsample_factor]
        self.upsample_factor = upsample_factor
        
        # 编码器配置
        encoder_config = model_params.get('encoder_config', {})
        self.encoder = EDSR(**encoder_config)
        
        encoder_feat_dim = encoder_config.get('n_feats', 64)
        
        # 特征融合层，输入通道: (encoder_feat + 2相对坐标) * 4个邻域 + 2个cell尺寸
        fusion_input_channels = (encoder_feat_dim + 2) * 4 + 2
        self.conv00 = nn.Conv2d(fusion_input_channels, self.width, 1)

        # Galerkin Transformer
        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, self.output_channels, 1)
        
    def gen_feat(self, inp):
        """
        生成特征图
        Args:
            inp: [B, H, W, C] 其中 C 只包含物理场特征通道
        Returns:
            feat: [B, encoder_feat_dim, H, W]
        """
        if inp.dim() == 4 and inp.shape[-1] <= 4:
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B,H,W,C] -> [B,C,H,W]
        
        inp_physics = inp
        self.inp = inp_physics
        self.feat = self.encoder(inp_physics)
        return self.feat
        
    def query_field(self, coord, cell):
        """
        查询任意坐标的物理场值
        Args:
            coord: [B, H', W', 2] 查询坐标 (归一化到[-1,1])
            cell: [B, 2] cell 尺寸
        Returns:
            ret: [B, output_channels, H', W']
        """
        feat = self.feat  # [B, feat_dim, H, W]
        
        # 生成低分辨率特征图的坐标网格
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # 计算邻域采样的偏移量
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        
        # 对4个邻域进行采样
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), 
                                     mode='nearest', align_corners=False)
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), 
                                         mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # 计算面积权重 (双线性插值的思想)
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
        
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        # 拼接所有特征: [相对坐标×4, 加权特征×4, cell尺寸]
        grid = torch.cat([
            *rel_coords, 
            *feat_s,
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])
        ], dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)
        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        
        # 残差连接
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                                  padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp):
        """
        前向传播
        Args:
            inp: [B, H, W, C] 输入数据 (包含坐标和物理场)
        Returns:
            output: [B, output_channels, H*scale_h, W*scale_w] 预测的物理场
        """
        # 【修改】动态计算输出分辨率
        B, H, W, C = inp.shape
        H_out = H * self.upsample_factor[0]
        W_out = W * self.upsample_factor[1]
        
        # 生成查询坐标
        coord = make_coord((H_out, W_out)).to(inp.device)
        coord = coord.unsqueeze(0).expand(B, -1, -1)
        coord = coord.view(B, H_out, W_out, 2)
        
        # 计算 cell 尺寸
        cell = torch.ones(B, 2).to(inp.device)
        cell[:, 0] *= 2 / H_out
        cell[:, 1] *= 2 / W_out
        
        self.gen_feat(inp)
        out = self.query_field(coord, cell)
        
        # 【修改】输出格式改为 [B, H, W, C] 以保持一致性
        out = out.permute(0, 2, 3, 1)
        
        return out