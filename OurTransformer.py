import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
from einops.layers.torch import Rearrange
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows
def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkvx = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropx = nn.Dropout(attn_drop)
        self.projx = nn.Linear(dim, dim)
        self.proj_dropx = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmaxx = nn.Softmax(dim=-1)

    def forward(self, x, maskx=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkvx = self.qkvx(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkvx[0], qkvx[1], qkvx[2]  # make torchscript happy (cannot use tensor as tuple)
        qx = qx * self.scale
        attnx = (qx @ kx.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attnx = attnx + relative_position_bias.unsqueeze(0)

        if maskx is not None:
            nW = maskx.shape[0]
            attnx = attnx.view(B_ // nW, nW, self.num_heads, N, N) + maskx.unsqueeze(1).unsqueeze(0)
            attnx = attnx.view(-1, self.num_heads, N, N)
            attnx = self.softmaxx(attnx)
        else:
            attnx = self.softmaxx(attnx)

        attnx = self.attn_dropx(attnx)
        x = (attnx @ vx).transpose(1, 2).reshape(B_, N, C)

        x = self.projx(x)
        x = self.proj_dropx(x)
        return x
class CrossWindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkvx = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkvy = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropx = nn.Dropout(attn_drop)
        self.attn_dropy = nn.Dropout(attn_drop)
        self.projx = nn.Linear(dim, dim)
        self.projy = nn.Linear(dim, dim)
        self.proj_dropx = nn.Dropout(proj_drop)
        self.proj_dropy = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmaxx = nn.Softmax(dim=-1)
        self.softmaxy = nn.Softmax(dim=-1)

    def forward(self, x, y, maskx=None, masky=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkvx = self.qkvx(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkvx[0], qkvx[1], qkvx[2]  # make torchscript happy (cannot use tensor as tuple)
        qkvy = self.qkvy(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qy, ky, vy = qkvy[0], qkvy[1], qkvy[2]  # make torchscript happy (cannot use tensor as tuple)
        qx = qx * self.scale
        qy = qy * self.scale
        attnx = (qx @ ky.transpose(-2, -1))
        attny = (qy @ kx.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attnx = attnx + relative_position_bias.unsqueeze(0)
            attny = attny + relative_position_bias.unsqueeze(0)

        if maskx is not None:
            nW = maskx.shape[0]
            attnx = attnx.view(B_ // nW, nW, self.num_heads, N, N) + maskx.unsqueeze(1).unsqueeze(0)
            attnx = attnx.view(-1, self.num_heads, N, N)
            attnx = self.softmaxx(attnx)
        else:
            attnx = self.softmaxx(attnx)
        if masky is not None:
            nW = masky.shape[0]
            attny = attny.view(B_ // nW, nW, self.num_heads, N, N) + masky.unsqueeze(1).unsqueeze(0)
            attny = attny.view(-1, self.num_heads, N, N)
            attny = self.softmaxy(attny)
        else:
            attny = self.softmaxy(attny)

        attnx = self.attn_dropx(attnx)
        attny = self.attn_dropy(attny)
        x = (attnx @ vy).transpose(1, 2).reshape(B_, N, C)
        y = (attny @ vx).transpose(1, 2).reshape(B_, N, C)

        x = self.projx(x)
        x = self.proj_dropx(x)
        y = self.projy(y)
        y = self.proj_dropy(y)
        return x,y
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, h, w ,t) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, h, w, t)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, h, w, t):
        b, n, c = x.size()
        x = x.view(b, h, w, t, c).permute(0,4,1,2,3)
        y = self.avg_pool(x)
        y = y.permute(0,2,3,4,1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        x = x * y.expand_as(x)
        out = x.view(b, c, n).permute(0, 2, 1)
        return out
class LeFF(nn.Module):
    def __init__(self, dim, hidden_dim, h, w, t):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.se = SELayer(hidden_dim)
        self.gule1 = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.hidden_dim = hidden_dim
        self.h, self.w, self.t = h, w, t
    def forward(self, x):
        x = self.linear1(x)
        rx = x
        x = self.dwconv(x, self.h, self.w, self.t)
        x = self.norm1(x)
        x = self.se(x, self.h, self.w, self.t)
        x = x + rx
        x = self.gule1(x)
        x = self.linear2(x)
        return x
class TripleWindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_selfx = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.qkv_crossx = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.qkv_selfy = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.qkv_crossy = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.attn_drop_selfx = nn.Dropout(attn_drop)
        self.attn_drop_crossx = nn.Dropout(attn_drop)
        self.attn_drop_selfy = nn.Dropout(attn_drop)
        self.attn_drop_crossy = nn.Dropout(attn_drop)
        self.proj_selfx = nn.Linear(dim // 2, dim // 2)
        self.proj_crossx = nn.Linear(dim // 2, dim // 2)
        self.proj_selfy = nn.Linear(dim // 2, dim // 2)
        self.proj_crossy = nn.Linear(dim // 2, dim // 2)
        self.proj_drop_selfx = nn.Dropout(proj_drop)
        self.proj_drop_crossx = nn.Dropout(proj_drop)
        self.proj_drop_selfy = nn.Dropout(proj_drop)
        self.proj_drop_crossy = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax_selfx = nn.Softmax(dim=-1)
        self.softmax_crossx = nn.Softmax(dim=-1)
        self.softmax_selfy = nn.Softmax(dim=-1)
        self.softmax_crossy = nn.Softmax(dim=-1)

    def forward(self, self_x, cross_x, self_y, cross_y, mask_selfx=None, mask_crossx=None, mask_selfy=None, mask_crossy=None):

        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = self_x.shape #(num_windows*B, Wh*Ww*Wt, C)

        qkv_selfx = self.qkv_selfx(self_x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self_qx, self_kx, self_vx = qkv_selfx[0], qkv_selfx[1], qkv_selfx[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_crossx = self.qkv_crossx(cross_x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cross_qx, cross_kx, cross_vx = qkv_crossx[0], qkv_crossx[1], qkv_crossx[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_selfy = self.qkv_selfy(self_y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self_qy, self_ky, self_vy = qkv_selfy[0], qkv_selfy[1], qkv_selfy[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_crossy = self.qkv_crossy(cross_y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cross_qy, cross_ky, cross_vy = qkv_crossy[0], qkv_crossy[1], qkv_crossy[2]  # make torchscript happy (cannot use tensor as tuple)

        self_qx = self_qx * self.scale
        cross_qx = cross_qx * self.scale
        self_qy = self_qy * self.scale
        cross_qy = cross_qy * self.scale

        self_attnx = (self_qx @ self_kx.transpose(-2, -1))
        self_attny = (self_qy @ self_ky.transpose(-2, -1))
        cross_attnx = (cross_qx @ cross_ky.transpose(-2, -1))
        cross_attny = (cross_qy @ cross_kx.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            self_attnx = self_attnx + relative_position_bias.unsqueeze(0)
            cross_attnx = cross_attnx + relative_position_bias.unsqueeze(0)
            self_attny = self_attny + relative_position_bias.unsqueeze(0)
            cross_attny = cross_attny + relative_position_bias.unsqueeze(0)

        if mask_selfx is not None:
            nW = mask_selfx.shape[0]
            self_attnx = self_attnx.view(B_ // nW, nW, self.num_heads, N, N) + mask_selfx.unsqueeze(1).unsqueeze(0)
            self_attnx = self_attnx.view(-1, self.num_heads, N, N)
            self_attnx = self.softmax_selfx(self_attnx)
        else:
            self_attnx = self.softmax_selfx(self_attnx)
        if mask_crossx is not None:
            nW = mask_crossx.shape[0]
            cross_attnx = cross_attnx.view(B_ // nW, nW, self.num_heads, N, N) + mask_crossx.unsqueeze(1).unsqueeze(0)
            cross_attnx = cross_attnx.view(-1, self.num_heads, N, N)
            cross_attnx = self.softmax_crossx(cross_attnx)
        else:
            cross_attnx = self.softmax_crossx(cross_attnx)
        if mask_selfy is not None:
            nW = mask_selfy.shape[0]
            self_attny = self_attny.view(B_ // nW, nW, self.num_heads, N, N) + mask_selfy.unsqueeze(1).unsqueeze(0)
            self_attny = self_attny.view(-1, self.num_heads, N, N)
            self_attny = self.softmax_selfy(self_attny)
        else:
            self_attny = self.softmax_selfy(self_attny)
        if mask_crossy is not None:
            nW = mask_crossy.shape[0]
            cross_attny = cross_attny.view(B_ // nW, nW, self.num_heads, N, N) + mask_crossy.unsqueeze(1).unsqueeze(0)
            cross_attny = cross_attny.view(-1, self.num_heads, N, N)
            cross_attny = self.softmax_crossy(cross_attny)
        else:
            cross_attny = self.softmax_crossy(cross_attny)

        self_attnx = self.attn_drop_selfx(self_attnx)
        self_x = (self_attnx @ self_vx).transpose(1, 2).reshape(B_, N, C)
        self_x = self.proj_selfx(self_x)
        self_x = self.proj_drop_selfx(self_x)

        self_attny = self.attn_drop_selfy(self_attny)
        self_y = (self_attny @ self_vy).transpose(1, 2).reshape(B_, N, C)
        self_y = self.proj_selfy(self_y)
        self_y = self.proj_drop_selfy(self_y)

        cross_attnx = self.attn_drop_crossx(cross_attnx)
        cross_x = (cross_attnx @ cross_vy).transpose(1, 2).reshape(B_, N, C)
        cross_x = self.proj_crossx(cross_x)
        cross_x = self.proj_drop_crossx(cross_x)

        cross_attny = self.attn_drop_crossy(cross_attny)
        cross_y = (cross_attny @ cross_vx).transpose(1, 2).reshape(B_, N, C)
        cross_y = self.proj_crossy(cross_y)
        cross_y = self.proj_drop_crossy(cross_y)

        return self_x, cross_x, self_y, cross_y
class TripleAttentionBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
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

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=10, w=12, t=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size,
                                                                                                     self.window_size)

        self.norm1selfx = norm_layer(dim // 2)
        self.norm1crossx = norm_layer(dim // 2)
        self.norm1selfy = norm_layer(dim // 2)
        self.norm1crossy = norm_layer(dim // 2)
        self.attn = TripleWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2x = norm_layer(dim)
        self.norm2y = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlpx = LeFF(dim, mlp_hidden_dim, h=h, w=w, t=t)
        self.mlpy = LeFF(dim, mlp_hidden_dim, h=h, w=w, t=t)

        self.H = None
        self.W = None
        self.T = None

    def forward(self, x, y, mask_matrix_selfx, mask_matrix_crossx, mask_matrix_selfy, mask_matrix_crossy):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcutx = x
        self_x, cross_x = torch.split(x, C // 2, dim=-1)
        self_x = self.norm1selfx(self_x)
        self_x = self_x.view(B, H, W, T, C // 2)
        cross_x = self.norm1crossx(cross_x)
        cross_x = cross_x.view(B, H, W, T, C // 2)

        shortcuty = y
        self_y, cross_y = torch.split(y, C // 2, dim=-1)
        self_y = self.norm1selfy(self_y)
        self_y = self_y.view(B, H, W, T, C // 2)
        cross_y = self.norm1crossy(cross_y)
        cross_y = cross_y.view(B, H, W, T, C // 2)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        self_x = nnf.pad(self_x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        cross_x = nnf.pad(cross_x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        self_y = nnf.pad(self_y, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        cross_y = nnf.pad(cross_y, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = self_x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_self_x = torch.roll(self_x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask_selfx = mask_matrix_selfx
        else:
            shifted_self_x = self_x
            attn_mask_selfx = None
        if min(self.shift_size) > 0:
            shifted_cross_x = torch.roll(cross_x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask_crossx = mask_matrix_crossx
        else:
            shifted_cross_x = cross_x
            attn_mask_crossx = None
        if min(self.shift_size) > 0:
            shifted_self_y = torch.roll(self_y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask_selfy = mask_matrix_selfy
        else:
            shifted_self_y = self_y
            attn_mask_selfy = None
        if min(self.shift_size) > 0:
            shifted_cross_y = torch.roll(cross_y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask_crossy = mask_matrix_crossy
        else:
            shifted_cross_y = cross_y
            attn_mask_crossy = None
        # partition windows
        self_x_windows = window_partition(shifted_self_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        self_x_windows = self_x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C // 2)  # nW*B, window_size*window_size*window_size, C
        cross_x_windows = window_partition(shifted_cross_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        cross_x_windows = cross_x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C // 2)  # nW*B, window_size*window_size*window_size, C
        self_y_windows = window_partition(shifted_self_y, self.window_size)  # nW*B, window_size, window_size, window_size, C
        self_y_windows = self_y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C // 2)  # nW*B, window_size*window_size*window_size, C
        cross_y_windows = window_partition(shifted_cross_y, self.window_size)  # nW*B, window_size, window_size, window_size, C
        cross_y_windows = cross_y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C // 2)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows_self_x, attn_windows_cross_x, attn_windows_self_y, attn_windows_cross_y \
            = self.attn(self_x_windows, cross_x_windows, self_y_windows, cross_y_windows, mask_selfx=attn_mask_selfx, mask_crossx=attn_mask_crossx, mask_selfy=attn_mask_selfy, mask_crossy=attn_mask_crossy)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows_self_x = attn_windows_self_x.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        shifted_self_x = window_reverse(attn_windows_self_x, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        attn_windows_cross_x = attn_windows_cross_x.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        shifted_cross_x = window_reverse(attn_windows_cross_x, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        attn_windows_self_y = attn_windows_self_y.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        shifted_self_y = window_reverse(attn_windows_self_y, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        attn_windows_cross_y = attn_windows_cross_y.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        shifted_cross_y = window_reverse(attn_windows_cross_y, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            self_x = torch.roll(shifted_self_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            self_x = shifted_self_x
        if min(self.shift_size) > 0:
            cross_x = torch.roll(shifted_cross_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            cross_x = shifted_cross_x
        if min(self.shift_size) > 0:
            self_y = torch.roll(shifted_self_y, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            self_y = shifted_self_y
        if min(self.shift_size) > 0:
            cross_y = torch.roll(shifted_cross_y, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            cross_y = shifted_cross_y

        if pad_r > 0 or pad_b > 0:
            self_x = self_x[:, :H, :W, :T, :].contiguous()
        if pad_r > 0 or pad_b > 0:
            cross_x = cross_x[:, :H, :W, :T, :].contiguous()
        if pad_r > 0 or pad_b > 0:
            self_y = self_y[:, :H, :W, :T, :].contiguous()
        if pad_r > 0 or pad_b > 0:
            cross_y = cross_y[:, :H, :W, :T, :].contiguous()

        self_x = self_x.view(B, H * W * T, C // 2)
        cross_x = cross_x.view(B, H * W * T, C // 2)
        x = torch.cat([self_x, cross_x], dim=-1)
        self_y = self_y.view(B, H * W * T, C // 2)
        cross_y = cross_y.view(B, H * W * T, C // 2)
        y = torch.cat([self_y, cross_y], dim=-1)

        # FFN
        x = shortcutx + self.drop_pathx(x)
        x = x + self.drop_pathx(self.mlpx(self.norm2x(x)))
        y = shortcuty + self.drop_pathy(y)
        y = y + self.drop_pathy(self.mlpy(self.norm2y(y)))

        return x, y
class TripleAttentionBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 h=10,
                 w=12,
                 t=14):
        super().__init__()
        self.dim = dim
        self.norm_layer = nn.LayerNorm(self.dim)
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            TripleAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                h=h,
                w=w,
                t=t,
            )
            for i in range(depth)])

        # patch merging layer
        #self.downsample = None

    def forward(self, x, y, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        self_img_maskx = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        cross_img_maskx = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        self_img_masky = torch.zeros((1, Hp, Wp, Tp, 1), device=y.device)  # 1 Hp Wp 1
        cross_img_masky = torch.zeros((1, Hp, Wp, Tp, 1), device=y.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))

        self_cntx = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    self_img_maskx[:, h, w, t, :] = self_cntx
                    self_cntx += 1

        cross_cntx = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    cross_img_maskx[:, h, w, t, :] = cross_cntx
                    cross_cntx += 1

        self_cnty = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    self_img_masky[:, h, w, t, :] = self_cnty
                    self_cnty += 1

        cross_cnty = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    cross_img_masky[:, h, w, t, :] = cross_cnty
                    cross_cnty += 1

        self_mask_windowsx = window_partition(self_img_maskx, self.window_size)  # nW, window_size, window_size, 1
        self_mask_windowsx = self_mask_windowsx.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        self_attn_maskx = self_mask_windowsx.unsqueeze(1) - self_mask_windowsx.unsqueeze(2)
        self_attn_maskx = self_attn_maskx.masked_fill(self_attn_maskx != 0, float(-100.0)).masked_fill(self_attn_maskx == 0, float(0.0))

        cross_mask_windowsx = window_partition(cross_img_maskx, self.window_size)  # nW, window_size, window_size, 1
        cross_mask_windowsx = cross_mask_windowsx.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        cross_attn_maskx = cross_mask_windowsx.unsqueeze(1) - cross_mask_windowsx.unsqueeze(2)
        cross_attn_maskx = cross_attn_maskx.masked_fill(cross_attn_maskx != 0, float(-100.0)).masked_fill(cross_attn_maskx == 0, float(0.0))

        self_mask_windowsy = window_partition(self_img_masky, self.window_size)  # nW, window_size, window_size, 1
        self_mask_windowsy = self_mask_windowsy.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        self_attn_masky = self_mask_windowsy.unsqueeze(1) - self_mask_windowsy.unsqueeze(2)
        self_attn_masky = self_attn_masky.masked_fill(self_attn_masky != 0, float(-100.0)).masked_fill(self_attn_masky == 0, float(0.0))

        cross_mask_windowsy = window_partition(cross_img_masky, self.window_size)  # nW, window_size, window_size, 1
        cross_mask_windowsy = cross_mask_windowsy.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        cross_attn_masky = cross_mask_windowsy.unsqueeze(1) - cross_mask_windowsy.unsqueeze(2)
        cross_attn_masky = cross_attn_masky.masked_fill(cross_attn_masky != 0, float(-100.0)).masked_fill(cross_attn_masky == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            x, y = blk(x, y, self_attn_maskx, cross_attn_maskx, self_attn_masky, cross_attn_masky)
        x = self.norm_layer(x)
        y = self.norm_layer(y)
        return x, H, W, T, x, H, W, T, y, H, W, T, y, H, W, T
class TripleAttention(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 embed_dim=96,
                 depth=2,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 rpe=True,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 h=10,
                 w=12,
                 t=14):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        '''self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)'''

        self.pos_dropx = nn.Dropout(p=drop_rate)
        self.pos_dropy = nn.Dropout(p=drop_rate)

        # stochastic depth
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]  # stochastic depth decay rule

        # build layer
        self.layer = TripleAttentionBasicLayer(dim=int(embed_dim),
                                depth=depth,
                                num_heads=8,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                rpe=rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=0,
                                norm_layer=norm_layer,
                                downsample=None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                h=h,
                                w=w,
                                t=t, )
        num_features = int(embed_dim)
        self.num_features = num_features

        # add a norm layer for each output
        '''for i_layer in out_indices:
            layer = norm_layer(num_features)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)'''
        #self.layer.append(self.norm_layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, y):
        """Forward function."""
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_dropx(x)
        y = y.flatten(2).transpose(1, 2)
        y = self.pos_dropy(y)

        x_out, H, W, T, x, Wh, Ww, Wt, y_out, H, W, T, y, Wh, Ww, Wt = self.layer(x, y, Wh, Ww, Wt)
        xout = x_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()
        yout = y_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()

        return xout, yout

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(TripleAttention, self).train(mode)
        self._freeze_stages()
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class GatedAdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super(GatedAdaptiveFusion, self).__init__()

        self.dim = dim
        self.dwconvx = nn.Conv3d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2)
        self.dwconvy = nn.Conv3d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2)
        self.dwconvz = nn.Conv3d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2)
        self.gulez = nn.GELU()

    def forward(self, x, y, z):
        rx = x
        dwconvx = self.dwconvx(x)
        ry = y
        dwconvy = self.dwconvy(y)

        zx = z[:,0:self.dim // 2,:,:]
        zy = z[:,self.dim // 2:,:,:]

        dwconvzx = self.dwconvz(zx)
        gulezx = self.gulez(dwconvzx)
        dwconvzy = self.dwconvz(zy)
        gulezy = self.gulez(dwconvzy)

        gatezx = gulezx * dwconvx
        attx = rx + gatezx
        gatezy = gulezy * dwconvy
        atty = ry + gatezy

        return attx, atty
class FusionDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.fusion = GatedAdaptiveFusion(in_channels)

    def forward(self, z, x, y):
        z = self.up(z)
        x, y = self.fusion(x, y, z)
        z = torch.cat([x, y, z], dim=1)
        z = self.conv1(z)
        z = self.conv2(z)
        return z
class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

#   CC  CC  CT  CT  CT
class AEMorph(nn.Module):
    def __init__(self):
        super(AEMorph, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec2 = self.encoder(32, 64, stride=2)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.tatransformer2 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)

        ex2 = self.ec2(ex1)
        ey2 = self.ec2(ey1)

        ex2, ey2 = self.tatransformer2(ex2, ey2)

        ex3 = self.ec3(ex2)
        ey3 = self.ec3(ey2)

        ex3, ey3 = self.tatransformer3(ex3, ey3)

        ex4 = self.ec4(ex3)
        ey4 = self.ec4(ey3)

        ex4, ey4 = self.tatransformer4(ex4, ey4)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
#   ...PE...CT  CT  CT
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, embed_dim=96, patch_size=4, in_chans=1, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x
class AEMorph_PE(nn.Module):
    def __init__(self):
        super(AEMorph_PE, self).__init__()

        self.eninput = self.encoder(1, 16)
        self.ec1 = self.encoder(16, 16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec3 = self.encoder(32, 32, kernel_size=3, stride=1, padding=1)
        self.ec4 = self.encoder(32, 64, stride=2)
        self.ec6 = self.encoder(64, 128, stride=2)
        self.ec8 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.patch_embedx = PatchEmbed(embed_dim=64) # PatchMerging + downsample
        self.patch_embedy = PatchEmbed(embed_dim=64) # PatchMerging + downsample

        self.tatransformer2 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.eninput(x)
        ex0 = self.ec1(ex0)

        ex1 = self.ec2(ex0)
        ex1 = self.ec3(ex1)

        ey0 = self.eninput(y)
        ey0 = self.ec1(ey0)

        ey1 = self.ec2(ey0)
        ey1 = self.ec3(ey1)

        ex1_pe = self.patch_embedx(x)
        ey1_pe = self.patch_embedy(y)

        ex2, ey2 = self.tatransformer2(ex1_pe, ey1_pe)
        ex2_down = self.ec6(ex2)
        ey2_down = self.ec6(ey2)

        ex3, ey3 = self.tatransformer3(ex2_down, ey2_down)
        ex3_down = self.ec8(ex3)
        ey3_down = self.ec8(ey3)

        ex4, ey4 = self.tatransformer4(ex3_down, ey3_down)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
#   CC  CC  CCT  CCT  CCT
class Morph_TAT_GAF(nn.Module):
    def __init__(self):
        super(Morph_TAT_GAF, self).__init__()

        self.eninput = self.encoder(1, 16)
        self.ec1 = self.encoder(16, 16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec3 = self.encoder(32, 32, kernel_size=3, stride=1, padding=1)
        self.ec4 = self.encoder(32, 64, stride=2)
        self.ec5 = self.encoder(64, 64, kernel_size=3, stride=1, padding=1)
        self.ec6 = self.encoder(64, 128, stride=2)
        self.ec7 = self.encoder(128, 128, kernel_size=3, stride=1, padding=1)
        self.ec8 = self.encoder(128, 256, stride=2)
        self.ec9 = self.encoder(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.tatransformer2 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.eninput(x)
        ex0 = self.ec1(ex0)

        ex1 = self.ec2(ex0)
        ex1 = self.ec3(ex1)

        ex2 = self.ec4(ex1)
        ex2 = self.ec5(ex2)

        ey0 = self.eninput(y)
        ey0 = self.ec1(ey0)

        ey1 = self.ec2(ey0)
        ey1 = self.ec3(ey1)

        ey2 = self.ec4(ey1)
        ey2 = self.ec5(ey2)

        ex2, ey2 = self.tatransformer2(ex2, ey2)

        ex3 = self.ec6(ex2)
        ex3 = self.ec7(ex3)

        ey3 = self.ec6(ey2)
        ey3 = self.ec7(ey3)

        ex3, ey3 = self.tatransformer3(ex3, ey3)

        ex4 = self.ec8(ex3)
        ex4 = self.ec9(ex4)

        ey4 = self.ec8(ey3)
        ey4 = self.ec9(ey4)

        ex4, ey4 = self.tatransformer4(ex4, ey4)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow

class AEMorph_TAT(nn.Module):
    def __init__(self):
        super(AEMorph_TAT, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec2 = self.encoder(32, 64, stride=2)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.tatransformer2 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              depth=2,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = DecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = DecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = DecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = DecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)

        ex2 = self.ec2(ex1)
        ey2 = self.ec2(ey1)

        ex2, ey2 = self.tatransformer2(ex2, ey2)

        ex3 = self.ec3(ex2)
        ey3 = self.ec3(ey2)

        ex3, ey3 = self.tatransformer3(ex3, ey3)

        ex4 = self.ec4(ex3)
        ey4 = self.ec4(ey3)

        ex4, ey4 = self.tatransformer4(ex4, ey4)

        de4 = torch.cat([ex4, ey4], dim=1)
        de3 = torch.cat([ex3, ey3], dim=1)
        de2 = torch.cat([ex2, ey2], dim=1)
        de1 = torch.cat([ex1, ey1], dim=1)
        de0 = torch.cat([ex0, ey0], dim=1)

        e4 = self.conv(de4)

        d1 = self.up1(e4, de3)
        d2 = self.up2(d1, de2)
        d3 = self.up3(d2, de1)
        d4 = self.up4(d3, de0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
class AEMorph_GAF(nn.Module):
    def __init__(self):
        super(AEMorph_GAF, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec2 = self.encoder(32, 64, stride=2)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)
        ex2 = self.ec2(ex1)
        ex3 = self.ec3(ex2)
        ex4 = self.ec4(ex3)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)
        ey2 = self.ec2(ey1)
        ey3 = self.ec3(ey2)
        ey4 = self.ec4(ey3)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow

class AEMorph_window(nn.Module):
    def __init__(self):
        super(AEMorph_window, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec2 = self.encoder(32, 64, stride=2)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.tatransformer21 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )
        self.tatransformer22 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer31 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )
        self.tatransformer32 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer41 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )
        self.tatransformer42 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)

        ex2 = self.ec2(ex1)
        ey2 = self.ec2(ey1)

        ex2, ey2 = self.tatransformer21(ex2, ey2)
        ex2, ey2 = self.tatransformer22(ex2, ey2)

        ex3 = self.ec3(ex2)
        ey3 = self.ec3(ey2)

        ex3, ey3 = self.tatransformer31(ex3, ey3)
        ex3, ey3 = self.tatransformer32(ex3, ey3)

        ex4 = self.ec4(ex3)
        ey4 = self.ec4(ey3)

        ex4, ey4 = self.tatransformer41(ex4, ey4)
        ex4, ey4 = self.tatransformer42(ex4, ey4)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
class AEMorph_singleT_window(nn.Module):
    def __init__(self):
        super(AEMorph_singleT_window, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec2 = self.encoder(32, 64, stride=2)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.tatransformer2 = TripleAttention(embed_dim=64,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=40,
                                              w=48,
                                              t=56,
                                              )

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(3, 3, 3),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)

        ex2 = self.ec2(ex1)
        ey2 = self.ec2(ey1)

        ex2, ey2 = self.tatransformer2(ex2, ey2)

        ex3 = self.ec3(ex2)
        ey3 = self.ec3(ey2)

        ex3, ey3 = self.tatransformer3(ex3, ey3)

        ex4 = self.ec4(ex3)
        ey4 = self.ec4(ey3)

        ex4, ey4 = self.tatransformer4(ex4, ey4)

        de4 = torch.cat([ex4, ey4], dim=1)
        e4 = self.conv(de4)

        d1 = self.up1(e4, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
class AEMorph_win(nn.Module):
    def __init__(self):
        super(AEMorph_win, self).__init__()

        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(16, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(32, 32)
        self.ec21 = self.encoder(32, 64, stride=2)
        self.ec22 = self.encoder(64, 64)
        self.ec3 = self.encoder(64, 128, stride=2)
        self.ec4 = self.encoder(128, 256, stride=2)
        self.ec5 = self.encoder(256, 512, stride=2)
        self.conv = self.encoder(512 * 2, 512, kernel_size=1, stride=1, padding=0)

        self.tatransformer3 = TripleAttention(embed_dim=128,
                                              window_size=(5, 6, 7),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=20,
                                              w=24,
                                              t=28,
                                              )

        self.tatransformer4 = TripleAttention(embed_dim=256,
                                              window_size=(5, 6, 7),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=10,
                                              w=12,
                                              t=14,
                                              )

        self.tatransformer5 = TripleAttention(embed_dim=512,
                                              window_size=(5, 6, 7),
                                              depth=1,
                                              mlp_ratio=4,
                                              qkv_bias=False,
                                              drop_rate=0,
                                              drop_path_rate=0.3,
                                              rpe=True,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              pat_merg_rf=4,
                                              h=5,
                                              w=6,
                                              t=7,
                                              )

        self.up0 = FusionDecoderBlock(512, 256, skip_channels=512, use_batchnorm=False)
        self.up1 = FusionDecoderBlock(256, 128, skip_channels=256, use_batchnorm=False)
        self.up2 = FusionDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = FusionDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = FusionDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec02(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec12(ex1)
        ex2 = self.ec21(ex1)
        ex2 = self.ec22(ex2)

        ey0 = self.ec01(y)
        ey0 = self.ec02(ey0)
        ey1 = self.ec11(ey0)
        ey1 = self.ec12(ey1)
        ey2 = self.ec21(ey1)
        ey2 = self.ec22(ey2)

        ex3 = self.ec3(ex2)
        ey3 = self.ec3(ey2)

        ex3, ey3 = self.tatransformer3(ex3, ey3)

        ex4 = self.ec4(ex3)
        ey4 = self.ec4(ey3)

        ex4, ey4 = self.tatransformer4(ex4, ey4)

        ex5 = self.ec5(ex4)
        ey5 = self.ec5(ey4)

        ex5, ey5 = self.tatransformer5(ex5, ey5)

        de5 = torch.cat([ex5, ey5], dim=1)
        e5 = self.conv(de5)

        d0 = self.up0(e5, ex4, ey4)
        d1 = self.up1(d0, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow

'''class DualNet(nn.Module):
    def __init__(self):
        super(DualNet, self).__init__()
        # OASIS分数合适，IXI太高
        self.ec01 = self.encoder(1, 16)
        self.ec02 = self.encoder(1, 16)
        self.ec11 = self.encoder(16, 32, stride=2)
        self.ec12 = self.encoder(16, 32, stride=2)
        self.ec21 = self.encoder(32, 64, stride=2)
        self.ec22 = self.encoder(32, 64, stride=2)
        self.ec31 = self.encoder(64, 128, stride=2)
        self.ec32 = self.encoder(64, 128, stride=2)
        self.ec41 = self.encoder(128, 256, stride=2)
        self.ec42 = self.encoder(128, 256, stride=2)
        #self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.up11 = DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)
        self.up12 = DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)
        #self.conv1 = self.encoder(128 + 128, 128, kernel_size=1, stride=1, padding=0)
        self.up21 = DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)
        self.up22 = DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)
        #self.conv2 = self.encoder(64 + 64, 64, kernel_size=1, stride=1, padding=0)
        self.up31 = DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)
        self.up32 = DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)
        #self.conv3 = self.encoder(32 + 32, 32, kernel_size=1, stride=1, padding=0)
        self.up41 = DecoderBlock(32, 16, skip_channels=16, use_batchnorm=False)
        self.up42 = DecoderBlock(32, 16, skip_channels=16, use_batchnorm=False)
        #self.conv4 = self.encoder(16 + 16, 16, kernel_size=1, stride=1, padding=0)

        self.pred0 = nn.Conv3d(256 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(128 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(64 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex1 = self.ec11(ex0)
        ex2 = self.ec21(ex1)
        ex3 = self.ec31(ex2)
        ex4 = self.ec41(ex3)

        ey0 = self.ec02(y)
        ey1 = self.ec12(ey0)
        ey2 = self.ec22(ey1)
        ey3 = self.ec32(ey2)
        ey4 = self.ec42(ey3)

        e4 = torch.cat([ex4, ey4], dim=1)
        #e4 = self.conv(e4)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up11(ex4, ex3)
        dy1 = self.up12(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        #d1 = self.conv1(d1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up21(dx1, ex2)
        dy2 = self.up22(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        #d2 = self.conv2(d2)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up31(dx2, ex1)
        dy3 = self.up32(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        #d3 = self.conv3(d3)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up41(dx3, ex0)
        dy4 = self.up42(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        #d4 = self.conv4(d4)
        pred4 = self.pred4(d4)
        #pred4_up = nnf.interpolate(pred4, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow
class UpDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x

class DualPRNet(nn.Module):
    def __init__(self):
        super(DualPRNet, self).__init__()

        self.ec01 = self.encoder(1, 8)
        self.ec02 = self.encoder(1, 8)
        self.ec11 = self.encoder(8, 16, stride=2)
        self.ec12 = self.encoder(8, 16, stride=2)
        self.ec21 = self.encoder(16, 16, stride=2)
        self.ec22 = self.encoder(16, 16, stride=2)
        self.ec31 = self.encoder(16, 32, stride=2)
        self.ec32 = self.encoder(16, 32, stride=2)
        self.ec41 = self.encoder(32, 32, stride=2)
        self.ec42 = self.encoder(32, 32, stride=2)
        #self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        self.up11 = UpDecoderBlock(32, 32, skip_channels=32, use_batchnorm=False)
        self.up12 = UpDecoderBlock(32, 32, skip_channels=32, use_batchnorm=False)
        #self.conv1 = self.encoder(128 + 128, 128, kernel_size=1, stride=1, padding=0)
        self.up21 = UpDecoderBlock(32, 16, skip_channels=16, use_batchnorm=False)
        self.up22 = UpDecoderBlock(32, 16, skip_channels=16, use_batchnorm=False)
        #self.conv2 = self.encoder(64 + 64, 64, kernel_size=1, stride=1, padding=0)
        self.up31 = UpDecoderBlock(16, 16, skip_channels=16, use_batchnorm=False)
        self.up32 = UpDecoderBlock(16, 16, skip_channels=16, use_batchnorm=False)
        #self.conv3 = self.encoder(32 + 32, 32, kernel_size=1, stride=1, padding=0)
        self.up41 = UpDecoderBlock(16, 8, skip_channels=8, use_batchnorm=False)
        self.up42 = UpDecoderBlock(16, 8, skip_channels=8, use_batchnorm=False)
        #self.conv4 = self.encoder(16 + 16, 16, kernel_size=1, stride=1, padding=0)

        self.pred0 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex1 = self.ec11(ex0)
        ex2 = self.ec21(ex1)
        ex3 = self.ec31(ex2)
        ex4 = self.ec41(ex3)

        ey0 = self.ec02(y)
        ey1 = self.ec12(ey0)
        ey2 = self.ec22(ey1)
        ey3 = self.ec32(ey2)
        ey4 = self.ec42(ey3)

        e4 = torch.cat([ex4, ey4], dim=1)
        #e4 = self.conv(e4)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up11(ex4, ex3)
        dy1 = self.up12(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        #d1 = self.conv1(d1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up21(dx1, ex2)
        dy2 = self.up22(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        #d2 = self.conv2(d2)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up31(dx2, ex1)
        dy3 = self.up32(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        #d3 = self.conv3(d3)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up41(dx3, ex0)
        dy4 = self.up42(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        #d4 = self.conv4(d4)
        pred4 = self.pred4(d4)
        #pred4_up = nnf.interpolate(pred4, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow'''
# OASIS分数太低
'''class DSPRNet(nn.Module):
    def __init__(self):
        super(DSPRNet, self).__init__()

        self.ec01 = self.encoder(1, 8)
        self.ec01res = ResNet(8,8)
        self.ec02 = self.encoder(1, 8)
        self.ec02res = ResNet(8, 8)
        self.ec11 = self.encoder(8, 16, stride=2)
        self.ec11res = ResNet(16,16)
        self.ec12 = self.encoder(8, 16, stride=2)
        self.ec12res = ResNet(16,16)
        self.ec21 = self.encoder(16, 16, stride=2)
        self.ec21res = ResNet(16,16)
        self.ec22 = self.encoder(16, 16, stride=2)
        self.ec22res = ResNet(16,16)
        self.ec31 = self.encoder(16, 32, stride=2)
        self.ec31res = ResNet(32,32)
        self.ec32 = self.encoder(16, 32, stride=2)
        self.ec32res = ResNet(32,32)
        self.ec41 = self.encoder(32, 32, stride=2)
        self.ec41res = ResNet(32,32)
        self.ec42 = self.encoder(32, 32, stride=2)
        self.ec42res = ResNet(32,32)

        self.up11 = DSDecoderBlock(32, 32, skip_channels=32, use_batchnorm=True)
        self.up12 = DSDecoderBlock(32, 32, skip_channels=32, use_batchnorm=True)
        self.up21 = DSDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up22 = DSDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up31 = DSDecoderBlock(16, 16, skip_channels=16, use_batchnorm=True)
        self.up32 = DSDecoderBlock(16, 16, skip_channels=16, use_batchnorm=True)
        self.up41 = DSDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)
        self.up42 = DSDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)

        self.pred0 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec01(x)
        ex0 = self.ec01res(ex0)
        ex1 = self.ec11(ex0)
        ex1 = self.ec11res(ex1)
        ex2 = self.ec21(ex1)
        ex2 = self.ec21res(ex2)
        ex3 = self.ec31(ex2)
        ex3 = self.ec31res(ex3)
        ex4 = self.ec41(ex3)
        ex4 = self.ec41res(ex4)

        ey0 = self.ec02(y)
        ey0 = self.ec02res(ey0)
        ey1 = self.ec12(ey0)
        ey1 = self.ec12res(ey1)
        ey2 = self.ec22(ey1)
        ey2 = self.ec22res(ey2)
        ey3 = self.ec32(ey2)
        ey3 = self.ec32res(ey3)
        ey4 = self.ec42(ey3)
        ey4 = self.ec42res(ey4)

        e4 = torch.cat([ex4, ey4], dim=1)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up11(ex4, ex3)
        dy1 = self.up12(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up21(dx1, ex2)
        dy2 = self.up22(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up31(dx2, ex1)
        dy3 = self.up32(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up41(dx3, ex0)
        dy4 = self.up42(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        pred4 = self.pred4(d4)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow'''
# OASIS分数太低
'''class DualResPRNet(nn.Module):
    def __init__(self):
        super(DualResPRNet, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec0res = ResNet(8,8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec1res = ResNet(16,16)
        self.ec2 = self.encoder(16, 16, stride=2)
        self.ec2res = ResNet(16,16)
        self.ec3 = self.encoder(16, 32, stride=2)
        self.ec3res = ResNet(32,32)
        self.ec4 = self.encoder(32, 32, stride=2)
        self.ec4res = ResNet(32,32)

        self.up1 = DSDecoderBlock(32, 32, skip_channels=32, use_batchnorm=True)
        self.up2 = DSDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up3 = DSDecoderBlock(16, 16, skip_channels=16, use_batchnorm=True)
        self.up4 = DSDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)

        self.pred0 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex0 = self.ec0res(ex0)
        ex1 = self.ec1(ex0)
        ex1 = self.ec1res(ex1)
        ex2 = self.ec2(ex1)
        ex2 = self.ec2res(ex2)
        ex3 = self.ec3(ex2)
        ex3 = self.ec3res(ex3)
        ex4 = self.ec4(ex3)
        ex4 = self.ec4res(ex4)

        ey0 = self.ec0(y)
        ey0 = self.ec0res(ey0)
        ey1 = self.ec1(ey0)
        ey1 = self.ec1res(ey1)
        ey2 = self.ec2(ey1)
        ey2 = self.ec2res(ey2)
        ey3 = self.ec3(ey2)
        ey3 = self.ec3res(ey3)
        ey4 = self.ec4(ey3)
        ey4 = self.ec4res(ey4)

        e4 = torch.cat([ex4, ey4], dim=1)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up1(ex4, ex3)
        dy1 = self.up1(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up2(dx1, ex2)
        dy2 = self.up2(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up3(dx2, ex1)
        dy3 = self.up3(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up4(dx3, ex0)
        dy4 = self.up4(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        pred4 = self.pred4(d4)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow'''
# OASIS分数合适
'''class Dual128(nn.Module):
    def __init__(self):
        super(Dual128, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec0res = ResNet(8,8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec1res = ResNet(16,16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec2res = ResNet(32,32)
        self.ec3 = self.encoder(32, 64, stride=2)
        self.ec3res = ResNet(64,64)
        self.ec4 = self.encoder(64, 128, stride=2)
        self.ec4res = ResNet(128,128)

        self.up1 = DSDecoderBlock(128, 64, skip_channels=64, use_batchnorm=True)
        self.up2 = DSDecoderBlock(64, 32, skip_channels=32, use_batchnorm=True)
        self.up3 = DSDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up4 = DSDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)

        self.pred0 = nn.Conv3d(128 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(64 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex0 = self.ec0res(ex0)
        ex1 = self.ec1(ex0)
        ex1 = self.ec1res(ex1)
        ex2 = self.ec2(ex1)
        ex2 = self.ec2res(ex2)
        ex3 = self.ec3(ex2)
        ex3 = self.ec3res(ex3)
        ex4 = self.ec4(ex3)
        ex4 = self.ec4res(ex4)

        ey0 = self.ec0(y)
        ey0 = self.ec0res(ey0)
        ey1 = self.ec1(ey0)
        ey1 = self.ec1res(ey1)
        ey2 = self.ec2(ey1)
        ey2 = self.ec2res(ey2)
        ey3 = self.ec3(ey2)
        ey3 = self.ec3res(ey3)
        ey4 = self.ec4(ey3)
        ey4 = self.ec4res(ey4)

        e4 = torch.cat([ex4, ey4], dim=1)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up1(ex4, ex3)
        dy1 = self.up1(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up2(dx1, ex2)
        dy2 = self.up2(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up3(dx2, ex1)
        dy3 = self.up3(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up4(dx3, ex0)
        dy4 = self.up4(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        pred4 = self.pred4(d4)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow
class Dual64(nn.Module):
    def __init__(self):
        super(Dual64, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec0res = ResNet(8,8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec1res = ResNet(16,16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec2res = ResNet(32,32)
        self.ec3 = self.encoder(32, 64, stride=2)
        self.ec3res = ResNet(64,64)
        self.ec4 = self.encoder(64, 64, stride=2)
        self.ec4res = ResNet(64,64)

        self.up1 = DSDecoderBlock(64, 64, skip_channels=64, use_batchnorm=True)
        self.up2 = DSDecoderBlock(64, 32, skip_channels=32, use_batchnorm=True)
        self.up3 = DSDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up4 = DSDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)

        self.pred0 = nn.Conv3d(64 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(64 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex0 = self.ec0res(ex0)
        ex1 = self.ec1(ex0)
        ex1 = self.ec1res(ex1)
        ex2 = self.ec2(ex1)
        ex2 = self.ec2res(ex2)
        ex3 = self.ec3(ex2)
        ex3 = self.ec3res(ex3)
        ex4 = self.ec4(ex3)
        ex4 = self.ec4res(ex4)

        ey0 = self.ec0(y)
        ey0 = self.ec0res(ey0)
        ey1 = self.ec1(ey0)
        ey1 = self.ec1res(ey1)
        ey2 = self.ec2(ey1)
        ey2 = self.ec2res(ey2)
        ey3 = self.ec3(ey2)
        ey3 = self.ec3res(ey3)
        ey4 = self.ec4(ey3)
        ey4 = self.ec4res(ey4)

        e4 = torch.cat([ex4, ey4], dim=1)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up1(ex4, ex3)
        dy1 = self.up1(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up2(dx1, ex2)
        dy2 = self.up2(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up3(dx2, ex1)
        dy3 = self.up3(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up4(dx3, ex0)
        dy4 = self.up4(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        pred4 = self.pred4(d4)

        #flow = self.reg_head(d4)
        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow'''

class DualConv3d(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)
        nm = nn.InstanceNorm3d(out_channels)
        super(DualConv3d, self).__init__(conv, nm, relu)
class ResNet(nn.Module):
    def __init__(self,in_channels,out_channels,use_batchnorm=True):
        super().__init__()
        self.conv1 = DualConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = DualConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)
        self.conv3 = DualConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)
        self.conv4 = DualConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x):
        rx1 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + rx1
        rx2 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + rx2
        return x
class DualDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = DualConv3d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x
class DualPRNet(nn.Module):
    def __init__(self):
        super(DualPRNet, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec0res = ResNet(8, 8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec1res = ResNet(16, 16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec2res = ResNet(32, 32)
        self.ec3 = self.encoder(32, 64, stride=2)
        self.ec3res = ResNet(64, 64)
        self.ec4 = self.encoder(64, 128, stride=2)
        self.ec4res = ResNet(128, 128)

        self.up1 = DualDecoderBlock(128, 64, skip_channels=64, use_batchnorm=True)
        self.up2 = DualDecoderBlock(64, 32, skip_channels=32, use_batchnorm=True)
        self.up3 = DualDecoderBlock(32, 16, skip_channels=16, use_batchnorm=True)
        self.up4 = DualDecoderBlock(16, 8, skip_channels=8, use_batchnorm=True)

        self.pred0 = nn.Conv3d(128 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred1 = nn.Conv3d(64 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv3d(32 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv3d(16 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv3d(8 * 2, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans1 = SpatialTransformer((20,24,28))
        self.spatial_trans2 = SpatialTransformer((40,48,56))
        self.spatial_trans3 = SpatialTransformer((80,96,112))
        self.spatial_trans4 = SpatialTransformer((160,192,224))
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex0 = self.ec0res(ex0)
        ex1 = self.ec1(ex0)
        ex1 = self.ec1res(ex1)
        ex2 = self.ec2(ex1)
        ex2 = self.ec2res(ex2)
        ex3 = self.ec3(ex2)
        ex3 = self.ec3res(ex3)
        ex4 = self.ec4(ex3)
        ex4 = self.ec4res(ex4)

        ey0 = self.ec0(y)
        ey0 = self.ec0res(ey0)
        ey1 = self.ec1(ey0)
        ey1 = self.ec1res(ey1)
        ey2 = self.ec2(ey1)
        ey2 = self.ec2res(ey2)
        ey3 = self.ec3(ey2)
        ey3 = self.ec3res(ey3)
        ey4 = self.ec4(ey3)
        ey4 = self.ec4res(ey4)

        e4 = torch.cat([ex4, ey4], dim=1)
        pred0 = self.pred0(e4)
        pred0_up = nnf.interpolate(pred0, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx1 = self.up1(ex4, ex3)
        dy1 = self.up1(ey4, ey3)
        out1 = self.spatial_trans1(dx1, pred0_up)
        d1 = torch.cat([out1, dy1], dim=1)
        pred1 = self.pred1(d1)
        pred1_up = nnf.interpolate(pred1, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx2 = self.up2(dx1, ex2)
        dy2 = self.up2(dy1, ey2)
        out2 = self.spatial_trans2(dx2, pred1_up)
        d2 = torch.cat([out2, dy2], dim=1)
        pred2 = self.pred2(d2)
        pred2_up = nnf.interpolate(pred2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx3 = self.up3(dx2, ex1)
        dy3 = self.up3(dy2, ey1)
        out3 = self.spatial_trans3(dx3, pred2_up)
        d3 = torch.cat([out3, dy3], dim=1)
        pred3 = self.pred3(d3)
        pred3_up = nnf.interpolate(pred3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        dx4 = self.up4(dx3, ex0)
        dy4 = self.up4(dy3, ey0)
        out4 = self.spatial_trans4(dx4, pred3_up)
        d4 = torch.cat([out4, dy4], dim=1)
        pred4 = self.pred4(d4)

        flow = pred4
        out = self.spatial_trans(x, flow)
        return out, flow

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3), x.size(4))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3), kernel.size(4))
    out = torch.nn.functional.conv3d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3), out.size(4))
    return out
class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        # in_channels:256, hidden:256, out_channels:2*K(K is number of anchors)
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feat_corr = xcorr_depthwise(search, kernel)
        feat_corr = torch.nn.functional.interpolate(feat_corr,size=(10,12,14),mode='nearest')# (batch,64,16,16)
        out = self.head(feat_corr)
        return out
class SiamDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, z, x, y):
        z = self.up(z)
        z = torch.cat([x, y, z], dim=1)
        z = self.conv1(z)
        z = self.conv2(z)
        return z
class SUNet(nn.Module):
    def __init__(self):
        super(SUNet, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec3 = self.encoder(32, 64, stride=2)
        self.ec4 = self.encoder(64, 128, stride=2)
        #self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)
        self.corr = DepthwiseXCorr(128,128,128)

        self.up1 = SiamDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up2 = SiamDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up3 = SiamDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)
        self.up4 = SiamDecoderBlock(16, 8, skip_channels=16, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex1 = self.ec1(ex0)
        ex2 = self.ec2(ex1)
        ex3 = self.ec3(ex2)
        ex4 = self.ec4(ex3)

        ey0 = self.ec0(y)
        ey1 = self.ec1(ey0)
        ey2 = self.ec2(ey1)
        ey3 = self.ec3(ey2)
        ey4 = self.ec4(ey3)

        out = self.corr(ex4, ey4)
        #e4 = torch.cat([ex4, ey4], dim=1)

        d1 = self.up1(out, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
class SiamUNet(nn.Module):
    def __init__(self):
        super(SiamUNet, self).__init__()

        self.ec0 = self.encoder(1, 8)
        self.ec1 = self.encoder(8, 16, stride=2)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec3 = self.encoder(32, 64, stride=2)
        self.ec4 = self.encoder(64, 128, stride=2)
        #self.conv = self.encoder(256 * 2, 256, kernel_size=1, stride=1, padding=0)
        self.corr = DepthwiseXCorr(128,128,128)

        self.up1 = SiamDecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up2 = SiamDecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up3 = SiamDecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)
        self.up4 = SiamDecoderBlock(16, 8, skip_channels=16, use_batchnorm=False)

        self.reg_head = RegistrationHead(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.ec0(x)
        ex1 = self.ec1(ex0)
        ex2 = self.ec2(ex1)
        ex3 = self.ec3(ex2)
        ex4 = self.ec4(ex3)

        ey0 = self.ec0(y)
        ey1 = self.ec1(ey0)
        ey2 = self.ec2(ey1)
        ey3 = self.ec3(ey2)
        ey4 = self.ec4(ey3)
        print(ex4.shape)
        print(ey4.shape)
        out = self.corr(ex4, ey4)
        #e4 = torch.cat([ex4, ey4], dim=1)
        print(out.shape)

        d1 = self.up1(out, ex3, ey3)
        d2 = self.up2(d1, ex2, ey2)
        d3 = self.up3(d2, ex1, ey1)
        d4 = self.up4(d3, ex0, ey0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
