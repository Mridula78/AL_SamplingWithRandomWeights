import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import math
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.init as init

from math import sqrt
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
from einops import rearrange
import torchvision.transforms as T
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import json  # Added missing import

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2 Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_out = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    @staticmethod
    def compute_macs(module, input, output):
        B, N, C = input[0].shape

        module.__flops__ += module.flops(N) * B


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = input_resolution[0]
        self.W = input_resolution[1]

        self.attn_mask_dict = {} # {self.H: self.create_attn_mask(self.H, self.W)}


    def create_attn_mask(self, H, W):
        # calculate attention mask for SW-MSA

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


    def forward(self, x):

        B, L, C = x.shape
        H = int(sqrt(L))
        W = H

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            if H in self.attn_mask_dict.keys():  # Fixed: "is" -> "in"
                attn_mask = self.attn_mask_dict[H]
            else:
                self.attn_mask_dict[H] = self.create_attn_mask(H, W).to(x.device)
                attn_mask = self.attn_mask_dict[H]

        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size} mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H = int(sqrt(L))
        W = H

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x, _ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward_with_features(self, x):
        fea = []
        for blk in self.blocks:
            x, _ = blk(x)
            fea.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, fea

    def forward_with_attention(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, attns


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input channels.
        num_classes (int): Number of classes for classification head.
        embed_dim (int): Embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layers.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), ape=False, patch_norm=True, 
                 return_all_tokens=False, use_mean_pooling=True, masked_im_modeling=False):

        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x_region = self.norm(x)  # B L C
        x = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return self.head(x)

    def get_selfattention(self, x, n=1):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if n==1:
            return self.get_last_selfattention(x)
        else:
            return self.get_all_selfattention(x)

    def get_last_selfattention(self, x):

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
            else:
                x, attns = layer.forward_with_attention(x)
                return attns[-1]

    def get_all_selfattention(self, x):
        attn_out = []

        for layer in self.layers:
            x, attns = layer.forward_with_attention(x)
            attn_out += attns

        return attn_out

    def get_intermediate_layers(self, x, n=1, return_patch_avgpool=False):

        num_blks = sum(self.depths)
        start_idx = num_blks - n

        sum_cur = 0
        for i, d in enumerate(self.depths):
            sum_cur_new = sum_cur + d
            if start_idx >= sum_cur and start_idx < sum_cur_new:
                start_stage = i
                start_blk = start_idx - sum_cur
            sum_cur = sum_cur_new


        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        output = []
        s = 0
        for i, layer in enumerate(self.layers):
            x, fea = layer.forward_with_features(x)

            if i >= start_stage:
                for x_ in fea[start_blk:]:

                    if i == len(self.layers)-1: 
                        x_ = self.norm(x_)

                    x_avg = torch.flatten(self.avgpool(x_.transpose(1, 2)), 1)  # B C 
                    if return_patch_avgpool:
                        x_o = x_avg
                    else:
                        x_o = torch.cat((x_avg.unsqueeze(1), x_), dim=1)         
                    output.append(x_o)

                start_blk = 0

        return output

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            if dist.get_rank() == 0:
                print(f"GFLOPs layer_{i}: {layer.flops() / 1e9}")
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                        or 'relative_position_index' not in k
                        or 'attn_mask' not in k
                )

                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')

                    if 'relative_position_bias_table' in k and v.size() != model_dict[k].size():
                        relative_position_bias_table_pretrained = v
                        relative_position_bias_table_current = model_dict[k]
                        L1, nH1 = relative_position_bias_table_pretrained.size()
                        L2, nH2 = relative_position_bias_table_current.size()
                        if nH1 != nH2:
                            logging.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logging.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((L1, nH1), (L2, nH2))
                                )
                                S1 = int(L1 ** 0.5)
                                S2 = int(L2 ** 0.5)
                                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                                    size=(S2, S2),
                                    mode='bicubic')
                                v = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

                    if 'absolute_pos_embed' in k and v.size() != model_dict[k].size():
                        absolute_pos_embed_pretrained = v
                        absolute_pos_embed_current = model_dict[k]
                        _, L1, C1 = absolute_pos_embed_pretrained.size()
                        _, L2, C2 = absolute_pos_embed_current.size()
                        if C1 != C1:
                            logging.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logging.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((1, L1, C1), (1, L2, C2))
                                )
                                S1 = int(L1 ** 0.5)
                                S2 = int(L2 ** 0.5)
                                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                                v = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1).flatten(1, 2)

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def freeze_pretrained_layers(self, frozen_layers=[]):
        for name, module in self.named_modules():
            if (
                    name.split('.')[0] in frozen_layers
                    or '.'.join(name.split('.')[0:2]) in frozen_layers
                    or (len(frozen_layers) > 0 and frozen_layers[0] is '*')
            ):
                for _name, param in module.named_parameters():
                    param.requires_grad = False
                logging.info(
                    '=> set param {} requires grad to False'
                        .format(name)
                )
        for name, param in self.named_parameters():
            if (
                    name.split('.')[0] in frozen_layers
                    or (len(frozen_layers) > 0 and frozen_layers[0] is '*')
                    and param.requires_grad is True
            ):
                param.requires_grad = False
                logging.info(
                    '=> set param {} requires grad to False'
                        .format(name)
                )
        return self

    def get_num_layers(self):
     
        return sum(self.depths)
    

# Ensure reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device setup - added missing device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
print('==> Preparing data..')
train_transform = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


# Load CIFAR-10 dataset
def load_data():
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    print(f"Size of train_set: {len(train_set)}")
    print(f"Size of test_set: {len(test_set)}")
    return train_set, test_set

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct * 100 / total
    return accuracy

# Function to train the model with timing
def train_model(model, train_loader, test_loader, epochs, learning_rate, path, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    total_train_time = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        scheduler.step()
        new_accuracy = test_model(model, test_loader)
        print(f"epoch: {epoch}, new accuracy: {new_accuracy:.2f}, loss: {loss.item():.4f}, epoch time: {epoch_time:.2f}s")
        
        # Check if the current accuracy is higher than the best
        if new_accuracy > best_accuracy:
            model_path = f"{path}/model_{best_accuracy:.2f}.pt"
            if os.path.exists(model_path):
                os.remove(model_path)
            best_accuracy = new_accuracy
            torch.save(model.state_dict(), f"{path}/model_{best_accuracy:.2f}.pt")
    
    print(f"Total training time for this phase: {total_train_time:.2f}s")
    return best_accuracy, total_train_time

def train_test_save(train_dataset, test_set, n, epochs, path, lr=0.01):
    best_accuracy = 0.0
    total_time = 0.0
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = SwinTransformer(img_size=32,
                        num_classes=10,
                        window_size=4, 
                        patch_size=2, 
                        embed_dim=96, 
                        depths=[2, 6, 4], 
                        num_heads=[3, 6, 12],
                        mlp_ratio=2, 
                        qkv_bias=True, 
                        drop_path_rate=0.1).to(device)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        new_accuracy, iter_time = train_model(model, train_loader, test_loader, epochs, lr, path)
        total_time += iter_time

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    
    print(f"\nTotal training time for train_test_save: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.2f}")
    return best_accuracy, total_time

def get_samples_by_confidence(model, dataset, k, selection_type='low'):
    """
    Get samples based on confidence level
    
    Args:
        model: PyTorch model
        dataset: PyTorch dataset
        k: number of samples to select
        selection_type: 'low' for low confidence, 'high' for high confidence
    """
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = torch.nn.functional.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    if selection_type == 'low':
        # Select k samples with LOWEST confidence
        selected_indices = sorted(range(len(confidence_scores)), 
                                  key=lambda i: confidence_scores[i], 
                                  reverse=False)[:k]
    else:  # 'high'
        # Select k samples with HIGHEST confidence
        selected_indices = sorted(range(len(confidence_scores)), 
                                  key=lambda i: confidence_scores[i], 
                                  reverse=True)[:k]
    
    selected_samples = torch.utils.data.Subset(dataset, selected_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in selected_indices]
    remainder_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    
    print(f"Selected {k} {selection_type}-confidence samples from {len(dataset)} total samples")
    
    return selected_samples, remainder_dataset

def load_best_model_from_folder(folder_path):
    def get_accuracy_from_filename(filename):
        try:
            return float(filename.split("_")[1][:-3])
        except:
            return 0.0

    model_files = [file for file in os.listdir(folder_path) if file.startswith("model_") and file.endswith(".pt")]

    if not model_files:
        print("No model files found in the folder.")
        return None
    else:
        try:
            best_model_filename = max(model_files, key=get_accuracy_from_filename)
            best_model_path = os.path.join(folder_path, best_model_filename)
            
            best_model = SwinTransformer(img_size=32,
                        num_classes=10,
                        window_size=4, 
                        patch_size=2, 
                        embed_dim=96, 
                        depths=[2, 6, 4], 
                        num_heads=[3, 6, 12],
                        mlp_ratio=2, 
                        qkv_bias=True, 
                        drop_path_rate=0.1).to(device)
            
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, test_set, path, 
                           batch_size=5000, selection_type='low'):
    """
    Train model until all data is consumed
    
    Args:
        batch_size: number of samples to add each iteration (k=5000)
        selection_type: 'low' for low confidence, 'high' for high confidence
    """
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    best_accuracy = test_model(model, test_loader)
    current_accuracy = 0.0
    accuracies = []
    iteration_times = []
    iteration = 1
    total_iteration_time = 0.0

    print(f"\nStarting degradation training with {len(remaining_set)} remaining samples")
    print(f"Selection type: {selection_type}-confidence, Batch size: {batch_size}")
    
    while len(remaining_set) > 0:
        print(f"\n--- Degradation Iteration {iteration} ---")
        iteration_start_time = time.time()
        
        # Get samples based on confidence level
        if len(remaining_set) < batch_size:
            batch_size = len(remaining_set)  # Use remaining samples if less than batch_size
        
        new_images, remaining_set = get_samples_by_confidence(model, remaining_set, 
                                                             k=batch_size, 
                                                             selection_type=selection_type)
        
        # Add new images to training dataset
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, new_images])   
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)      
        
        # Train for 200 epochs
        current_accuracy, iter_time = train_model(model, train_loader, test_loader, 100, 0.01, path)
        
        iteration_time = time.time() - iteration_start_time
        iteration_times.append(iteration_time)
        total_iteration_time += iteration_time
        
        accuracies.append(current_accuracy)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        
        print(f"Iteration {iteration} complete:")
        print(f"  - Current accuracy: {current_accuracy:.2f}")
        print(f"  - Best accuracy: {best_accuracy:.2f}")
        print(f"  - Iteration time: {iteration_time:.2f}s")
        print(f"  - Remaining samples: {len(remaining_set)}")
        print(f"  - Training set size: {len(train_dataset)}")
        
        iteration += 1
        
    print(f"\nTraining dataset exhausted. Stopping degradation training.")
    print(f"Total degradation training time: {total_iteration_time:.2f}s")
    print(f"Final best accuracy: {best_accuracy:.2f}")

    # Save accuracies and times
    with open(f"{path}/accuracies.txt", "w") as file:
        file.write("Iteration,Accuracy,Time(s)\n")
        for i, (acc, t) in enumerate(zip(accuracies, iteration_times), start=1):
            file.write(f"{i},{acc:.2f},{t:.2f}\n")
        file.write(f"\nTotal degradation time: {total_iteration_time:.2f}s\n")
        file.write(f"Final best accuracy: {best_accuracy:.2f}\n")

    torch.save(model.state_dict(), f"{path}/finalmodel_{best_accuracy:.2f}.pt")
    return model, best_accuracy, total_iteration_time

def run_experiment(seed_idx, seed, experiment_config):
    """
    Run a complete experiment with a given seed and configuration
    
    Args:
        experiment_config: dictionary with keys:
            - name: experiment name
            - initial_selection: 'low' or 'high' for initial 10k
            - degradation_selection: 'low' or 'high' for 5k batches
    """
    print(f"\n{'='*80}")
    print(f"Running {experiment_config['name']}")
    print(f"Experiment {seed_idx+1} with seed: {seed}")
    print(f"Initial 10k: {experiment_config['initial_selection']}-confidence")
    print(f"Degradation 5k: {experiment_config['degradation_selection']}-confidence")
    print(f"{'='*80}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directory for this experiment
    exp_name = experiment_config['name'].replace(" ", "_").lower()
    path = f"./results/{exp_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{path}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    
    # Load data
    train_set, test_set = load_data()
    
    # Initialize model
    model = SwinTransformer(img_size=32,
                        num_classes=10,
                        window_size=4, 
                        patch_size=2, 
                        embed_dim=96, 
                        depths=[2, 6, 4], 
                        num_heads=[3, 6, 12],
                        mlp_ratio=2, 
                        qkv_bias=True, 
                        drop_path_rate=0.1).to(device)
    
    # Track total experiment time
    experiment_start_time = time.time()
    
    # Step 1: Get initial dataset (10k samples)
    print(f"\nStep 1: Getting initial {experiment_config['initial_selection']}-confidence dataset...")
    initial_trainset, remainder = get_samples_by_confidence(model, train_set, 
                                                           k=10000, 
                                                           selection_type=experiment_config['initial_selection'])
    print(f"Initial training set size: {len(initial_trainset)}")
    print(f"Remaining set size: {len(remainder)}")
    
    # Step 2: Train initial model
    print("\nStep 2: Training initial model...")
    initial_accuracy, initial_time = train_test_save(initial_trainset, test_set, 1, 100, path, lr=0.01)
    
    # Step 3: Load best model
    print("\nStep 3: Loading best model...")
    model = load_best_model_from_folder(path)
    if model is None:
        print("Failed to load model. Skipping this run.")
        return 0.0, 0.0
    
    # Step 4: Train until degradation
    print(f"\nStep 4: Training until degradation ({experiment_config['degradation_selection']}-confidence)...")
    final_model, final_accuracy, degradation_time = train_until_degradation(
        model, initial_trainset, remainder, test_set, path, 
        batch_size=5000, selection_type=experiment_config['degradation_selection']
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    print(f"\n{experiment_config['name']} (Seed: {seed}) Summary:")
    print(f"  - Initial training time: {initial_time:.2f}s")
    print(f"  - Degradation training time: {degradation_time:.2f}s")
    print(f"  - Total experiment time: {total_experiment_time:.2f}s")
    print(f"  - Initial accuracy: {initial_accuracy:.2f}%")
    print(f"  - Final accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy, total_experiment_time

def run_all_experiments():
    """Run all 4 experiments with"""
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    # Define 5 different seeds
    seeds = [42, 789, 101112]
    
    # Define all 4 experiments
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 10k + Low Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 10k + Low Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 10k + High Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 10k + High Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'high'
        }
    ]
    
    # Store results from all runs
    all_results = {}
    
    
    # Run each experiment
    for exp_idx, experiment in enumerate(experiments):
        print(f"\n{'*'*100}")
        print(f"STARTING: {experiment['name']}")
        print(f"{'*'*100}")
        
        exp_accuracies = []
        exp_times = []
        exp_summary = {}
        
        # Run with 5 different seeds
        for seed_idx, seed in enumerate(seeds):
            accuracy, exp_time = run_experiment(seed_idx, seed, experiment)
            exp_accuracies.append(accuracy)
            exp_times.append(exp_time)
            exp_summary[f"seed_{seed}"] = {
                "accuracy": accuracy,
                "time_hours": exp_time / 3600
            }
            
            # Save intermediate results for this experiment
            exp_dir = f"./results/{experiment['name'].replace(' ', '_').lower()}"
            os.makedirs(exp_dir, exist_ok=True)
            with open(f"{exp_dir}/results_summary.json", "w") as f:
                json.dump(exp_summary, f, indent=4)
        
        # Calculate statistics for this experiment
        if exp_accuracies and exp_times:
            mean_accuracy = np.mean(exp_accuracies)
            std_accuracy = np.std(exp_accuracies)
            mean_time = np.mean(exp_times)
            std_time = np.std(exp_times)
            
            exp_results = {
                'name': experiment['name'],
                'seeds': seeds,
                'accuracy_format': f"{mean_accuracy:.2f} ± {std_accuracy:.2f}%",
                'time_format_hours': f"{mean_time/3600:.2f} ± {std_time/3600:.2f} hours"
            }
            
            all_results[experiment['name']] = exp_results
            
            # Save results for this experiment
            with open(f"{exp_dir}/final_results.json", "w") as f:
                json.dump(exp_results, f, indent=4)
            
            # Save a simple text summary for this experiment
            with open(f"{exp_dir}/summary.txt", "w") as f:
                f.write(f"{experiment['name']} - FINAL RESULTS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Configuration:\n")
                f.write(f"  Initial 10k: {experiment['initial_selection']}-confidence\n")
                f.write(f"  Degradation batches (5k): {experiment['degradation_selection']}-confidence\n\n")
                f.write("Accuracy Results:\n")
                f.write(f"  Mean ± Std: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n\n")
                f.write("Training Time Results:\n")
                f.write(f"  Mean ± Std: {mean_time/3600:.2f} ± {std_time/3600:.2f} hours\n\n")
                
                f.write("Detailed Results:\n")
                for i, (seed, acc, t) in enumerate(zip(seeds, exp_accuracies, exp_times)):
                    f.write(f"  Seed {seed}:\n")
                    f.write(f"    Accuracy: {acc:.2f}%\n")
                    f.write(f"    Time: {t:.2f}s ({t/60:.2f} minutes)\n")
        
        print(f"\n{'*'*100}")
        print(f"COMPLETED: {experiment['name']}")
        print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
        print(f"{'*'*100}\n")
    
    # Save comprehensive results across all experiments
    with open("./results/all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print final comparison
    print(f"\n{'='*100}")
    print("ALL EXPERIMENTS COMPLETE - FINAL COMPARISON")
    print(f"{'='*100}\n")
    
    for exp_name, results in all_results.items():
        print(f"{exp_name}:")
        print(f"  Accuracy: {results['accuracy_format']}")
        print(f"  Time: {results['time_format_hours']}")
        print()

def run_specific_experiment(exp_number):
    """Run a specific experiment (1, 2, 3, or 4)"""
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 10k + Low Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 10k + Low Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 10k + High Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 10k + High Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'high'
        }
    ]
    
    if exp_number < 1 or exp_number > 4:
        print("Invalid experiment number. Choose 1, 2, 3, or 4.")
        return
    
    experiment = experiments[exp_number - 1]
    seeds = [42, 789, 101112]
    
    print(f"\n{'='*100}")
    print(f"RUNNING SPECIFIC EXPERIMENT: {experiment['name']}")
    print(f"{'='*100}")
    
    exp_accuracies = []
    exp_times = []
    
    for seed_idx, seed in enumerate(seeds):
        accuracy, exp_time = run_experiment(seed_idx, seed, experiment)
        exp_accuracies.append(accuracy)
        exp_times.append(exp_time)
    
    # Calculate statistics
    if exp_accuracies and exp_times:
        mean_accuracy = np.mean(exp_accuracies)
        std_accuracy = np.std(exp_accuracies)
        mean_time = np.mean(exp_times)
        std_time = np.std(exp_times)
        
        print(f"\n{'='*100}")
        print(f"{experiment['name']} - FINAL RESULTS")
        print(f"{'='*100}")
        print(f"\nConfiguration:")
        print(f"  Initial 10k: {experiment['initial_selection']}-confidence")
        print(f"  Degradation batches (5k): {experiment['degradation_selection']}-confidence")
        print(f"\nResults:")
        print(f"  Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"  Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
        print(f"\nDetailed Results:")
        for i, (seed, acc, t) in enumerate(zip(seeds, exp_accuracies, exp_times)):
            print(f"  Seed {seed}: Accuracy = {acc:.2f}%, Time = {t/60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run swin experiments on CIFAR-10')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4], 
                       help='Run specific experiment (1, 2, 3, or 4)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all 4 experiments (default)')
    
    args = parser.parse_args()
    
    if args.exp:
        run_specific_experiment(args.exp)
    else:
        # Default: run all experiments
        run_all_experiments()