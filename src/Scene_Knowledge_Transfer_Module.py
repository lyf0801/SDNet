import os
import copy
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

"""
Multiscale Pooling
"""
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self):
        super().__init__()
        pool_size = 1
        self.pool1 = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        pool_size = 3
        self.pool3 = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        pool_size = 5
        self.pool5 = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        pool_size = 7
        self.pool7 = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x):
        return torch.cat(((self.pool1(x) - x), (self.pool3(x) - x), (self.pool5(x) - x), (self.pool7(x) - x)), dim=1)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim,  mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                  layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(1)##scene vector is one channel
        self.token_mixer = Pooling()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features = 1, #one channel scene context vector
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
       
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((1)), requires_grad=True)##torch.ones(1) guarantee scene vector as one channel

    def forward(self, x):
        x_repeat = x.repeat(1, 4, 1, 1) #torch.Size([B, 1, H, W]) --> torch.Size([B, 4, H, W])

        out = x_repeat + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
            * self.token_mixer(self.norm1(x))) #--> torch.Size([B, 4, H, W])
        
        out = torch.mean(out, dim=1, keepdim = True) + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
            * self.mlp(self.norm2(out)))  #torch.Size([B, 4, H, W]) --> torch.Size([B, 1, H, W])
        return out

if __name__ == "__main__":
    module = PoolFormerBlock(dim= 4).cuda()
    y = torch.Tensor(2, 1, 14, 14).cuda() ##Scene context vector
    print(module(y).size())