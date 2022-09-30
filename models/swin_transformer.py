# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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
        window_size (int): window size ， default  7

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # E.g. ( B , 56 , 56 , 96 )
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    '''
            B , 
            num patch of a raw within a window  , 
            window_size, 
            num patch of a raw within a window   , 
            window size , 
            C
    '''
    # (B , num patch of a raw within a window  , window_size, num patch of a raw within a window , window size , C )
    # ( B , 8 , 7 , 8 , 7 , 96)
    # ( 0 , 1 , 2 ,  3 ,  4 , 5 )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # ( B * total num of window , window_size , window_size ,C)
    # ( B * 8 * 8 , 7 , 7 , 96 )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)  # that is ( B * 8 * 8 , 7 , 7 , 96 )
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
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window. default 7
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww  default  7
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # sqrt(dim K = Q)

        # define a parameter table of relative position bias
        # see https://svainzhu.com/2022/02/Swin-T.html
        # 把每个patch看成一个像素 , M = 7
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        '''
            >>> coords_h = torch.arange(3)
            >>> coords_w = torch.arange(3)
            >>> coords_w
            out[0] : 
                    tensor([0, 1, 2])
            >>> torch.meshgrid([coords_h, coords_w])
            out[1] : 
                     (tensor([[0, 0, 0],
                              [1, 1, 1],
                              [2, 2, 2]]),
                      tensor([[0, 1, 2],
                              [0, 1, 2],
                              [0, 1, 2]]))
            >>> coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            out[2] : 
                    tensor([[[0, 0, 0],
                             [1, 1, 1],
                             [2, 2, 2]],
                            [[0, 1, 2],
                             [0, 1, 2],
                             [0, 1, 2]]])
        
        '''
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        '''
            >>> coords_flatten
            out[0]:
                    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        '''

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        '''
            # here coords_flatten[:, :, None] <=> coords_flatten.unsqueeze(-1)
            # so the " coords_flatten[:, :, None] "  is like :
                                tensor([[[0],
                                         [0],
                                         [0],
                                         [1],
                                         [1],
                                         [1],
                                         [2],
                                         [2],
                                         [2]],
                                        [[0],
                                         [1],
                                         [2],
                                         [0],
                                         [1],
                                         [2],
                                         [0],
                                         [1],
                                         [2]]])
            # so the " coords_flatten[:, None, : ] "  is like : 
                    tensor([[[0, 0, 0, 1, 1, 1, 2, 2, 2]],
                            [[0, 1, 2, 0, 1, 2, 0, 1, 2]]])
        
        '''
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 行标

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 将得到的相对索引(x,y)合并变为一个新的索引 : x + y , 同时这个索引表不需要变动,注册为 buffer
        self.register_buffer("relative_position_index", relative_position_index)
        #  =========================================================================================


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # for q , k , v
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) # Dropout ratio of output. Default: 0.0

        trunc_normal_(self.relative_position_bias_table, std=.02) # 将bias控制在0附近
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # shape like ( batch  , seq_len , dimension )
        # e.g ( B * 8 * 8  , 7 * 7 , 96 ) for window size = 7 ,patch size = 4
        # then it have 56 * 56 patch , and a window contains 7*7 patch , a patch have 4 * 4 pixel
        # so have 8 * 8 window


        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        # self.qkv(x) shape with ( B * 8 * 8  , 7 * 7 , 96 * 3 )
        # lets say head = 3 , then reshape to ( B * 8 * 8 , 7 * 7 , 3,  3 , 96/3 )
        # because of num_heads = [3, 6, 12, 24]
        #  qkv with （if num_head = 3）
        #  ( 3,        B_,   self.num_heads,  N,    C // self.num_heads)
        #  ( 3 ,  B * 8 * 8 ,     3  ,      7 * 7  ,     32 )
        #  Q K V , 操作所有batch , 然后再拆分为具体几个head,每个head都要把seq
        #  过一遍, 但是seq的每一行 feature num 变为 C // head (只负责当前head这一部分)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # shape with ( B * 8 * 8 ,  3  , 7 * 7  ,   32 )

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # ( B_ ,  num_heads , N  , N)
        # ( B * 8 * 8 ,  3  , 7 * 7  , 7 * 7)
        # where N_ij is weights of patch_i and patch_j

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # Q*K^T + B
        # ( B_ ,  num_heads , N , N )
        # ( B * 8 * 8 ,  3  , 7 * 7  , 7 * 7)
        # where N_ij is weights of patch_i and patch_j

        if mask is not None:
            nW = mask.shape[0]
            # input shape with # ( 8 * 8 , 7 * 7 , 7 * 7 ）
            # (B_ // nW, nW, self.num_heads, N, N)  ： （ B , 8 * 8 , 3 , 7 * 7 , 7 * 7 )
            # mask.unsqueeze(1).unsqueeze(0) : torch.Size([1, 64, 1, 49, 49])
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # mask
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # (attn @ v) =
        #       ( B_ ,  3 , 56*56 , 56*56 ) * ( B_, 3 , 56 * 56 , 32 )
        #           = ( B_, 3 , 56 * 56 , 32 )
        # 再reshape , ( B_ , 56 * 56 , C )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # input_resolution = [ 56 , 28 , 14 , 7 ]
        # Corresponding [ H/4 , H/8 , H/16 , H/32 ] ,H = 224
        self.num_heads = num_heads
        # num_heads = [3, 6, 12, 24]
        self.window_size = window_size # 7
        self.shift_size = shift_size # 0 if (i % 2 == 0) else window_size // 2
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
        '''
            attention 操作是一样的 , 只不过输入不一样,当shift_size > 0 时,会对输入的x进行roll操作
            具体实现见forward函数,另外,当shift_size > 0时,还要有相应的mask矩阵.
            在得到 Q@K^T 的矩阵之后 再把 mask 加到上边
        '''

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # Need calculate attention mask for SW-MSA
            # see detail https://github.com/microsoft/Swin-Transformer/issues/38
            H, W = self.input_resolution # default path size = 4 , image size = 224 ,then H = 56

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            # e.g. shape with ( 1 , 56 , 56 ,1 )
            # default window_size = 7 , total 8*8 window
            # shift_size = window_size//2

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            # Is equality : (slice(0, -7, None), slice(-7, -3, None), slice(-3, None, None))
            # slice(start, stop[, step])
            '''
            a = list(range(5))
             out : [0, 1, 2, 3, 4]
            b = slice(1,4,1)
             out : slice(1, 4, 1)
            a[b]
             out : [1, 2, 3]
            '''

            # 这里在给几个区域划分标号
            # 标号为 0,1,2,3,4,5,6,7,8 ,从左到右,从上到下
            cnt = 0
            for h in h_slices:
                '''
                    slice(0, -7, None)
                    slice(-7, -3, None)
                    slice(-3, None, None)
                '''
                for w in w_slices:
                    '''
                        slice(0, -7, None)
                        slice(-7, -3, None)
                        slice(-3, None, None)
                    '''
                    img_mask[:, h, w, :] = cnt # 标号
                    # e.g. shape with ( 1 , 56 , 56 ,1 )
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            # nW, window_size, window_size, 1
            # ( total num of window , window_size , window_size ,1) , B = 1
            # ( 8 * 8 , 7 , 7 , 1 )
            # 这个地方就是 ‘1张’ image
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # ( 8 * 8 , 7 * 7 ）
            # 总共 8 * 8 个窗口， 每个窗口内 7 * 7 个 patch相互之间计算attention

            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 广播减法
            # # ( 8 * 8 , 7 * 7 , 7 * 7）

            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            '''
            # 拉直 , 做广播减法 , 以右下角4个区域为例, 4 5 7 8
            >>> a = torch.Tensor([4,5,7,8])
            >>> b = a.reshape(-1,1)
            >>> a - b
            >>>           4    5    7    8
                tensor([[ 0.,  1.,  3.,  4.],   4
                        [-1.,  0.,  2.,  3.],   5
                        [-3., -2.,  0.,  1.],   7
                        [-4., -3., -1.,  0.]])  8
            # 可以看到,对角线为0,正好表示 区域4 和 区域4 要做attention ,
            # 而4 与其他位置不为0 ,不需要丛attention , 同理

            # 这里分了9个区域,实际上4个区域即可
            # See https://github.com/microsoft/Swin-Transformer/issues/194
            '''
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # eg. shape( B , 56*56 , 96) ,56 =  224/4 means that "input_resolution"
        # 4 means patch size ,which pixel num of a row of a patch

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) # eg , shape( B , 56 , 56 , 96)

        # cyclic shift
        if self.shift_size > 0:
            # in a block  , i for layer index , the shift_size =  0 if (i % 2 == 0) else window_size // 2
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                '''
                    torch.roll 
                    https://blog.csdn.net/weixin_42899627/article/details/116095067
                    >>> x  
                        tensor([[0, 1, 2],
                                [3, 4, 5],
                                [6, 7, 8]])
                    >>> shifted_x = torch.roll(x, shifts=(-1,-1),dims=(0,1))
                        tensor([[4, 5, 3],
                                [7, 8, 6],
                                [1, 2, 0]])   
                        
                    the shifted attention visualization 
                        https://github.com/microsoft/Swin-Transformer/issues/38 
                '''
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)
                # input shape( B , 56 , 56 , 96)
                # # ( B * total num of window , window_size , window_size ,C)
                # e.g ( B * 8 * 8  , 7 , 7 , 96 ) for window size = 7 ,patch size = 4
                # then it have 56 * 56 patch , and a window contains 7*7patch

            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            # Do not slide
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)
            # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # nW*B, window_size*window_size, C
        # # ( B * total num of window , window_size * window_size ,C)
        # input ( B * 8 * 8  , 7 , 7 , 96 ) - > ( B * 8 * 8  , 7 * 7 , 96 )
        # then it have 56 * 56 patch , and a window contains 7*7 patch
        # shape like ( batch  , seq_len , dimension )


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # nW*B, window_size*window_size, C
        # here nW = 8 * 8

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        # 因为前边为了计算attention方便,对shift之后的map进行了局部移位,现在移回去
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

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
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature. eg 56
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
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution # eg 56
        B, L, C = x.shape # ( B , 56 * 56 , C )
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) # eg ( B , 56 , 56 , C )
        '''
        assume x : 6 * 6 
                        *  *  *  *  *  * 
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *
                        *  *  *  *  *  *  
        '''

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        '''
            x0 
                        +  *  +  *  +  * 
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *   
        '''
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        '''
            x1 
                        *  *  *  *  *  * 
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *
                        *  *  *  *  *  *
                        +  *  +  *  +  *  
        '''

        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        '''
            x2 
                         *  +  *  +  *  + 
                         *  *  *  *  *  *
                         *  +  *  +  *  +
                         *  *  *  *  *  *
                         *  +  *  +  *  +
                         *  *  *  *  *  *         
        '''
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        '''
            x3 
                        *  *  *  *  *  * 
                        *  +  *  +  *  +
                        *  *  *  *  *  *
                        *  +  *  +  *  +
                        *  *  *  *  *  *
                        *  +  *  +  *  +  
        '''
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
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
                                        # patch num in a row or column ,eg. 224/4 = 56
                                        # that's number of patches contained in each line
        depth (int): Number of layer wwithin a block.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        #  dim =  [ 96 , 96*2 ,96*4 ,96*8]
        self.input_resolution = input_resolution
        # input_resolution = [ 56 , 28 , 14 , 7 ]
        # Corresponding [ H/4 , H/8 , H/16 , H/32 ] ,H = 224
        self.depth = depth

        # # [2,2,6,2]
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size= 0 if (i % 2 == 0) else window_size // 2,
                                 # odd layer original attention and even layer shifted attention
                                 # in one block
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path= drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])
        # for first block , depth = 2 , second is 2 ,third is 6 , last is 2
        # which means first block have 2 layers , second blocks have 2 layers

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
                # https://blog.csdn.net/ONE_SIX_MIX/article/details/93937091
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

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
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # (224, 224)
        patch_size = to_2tuple(patch_size)
        # ( 4, 4 ) , means that a patch within (4 pixel * 4 pixel )

        # patch num in a row or column ,eg. 224/4 = 56
        # that's number of patches contained in each line
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # （56 ，56）
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.num_patches = patches_resolution[0] * patches_resolution[1] # 56 * 56
        # total path in a channel

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可以看到，这里使用的是patch_size的核 和 patch_size的步长来实现几个像素作为一个patch
        # 和VIT里边的不一样，VIT是直接物理上把图片按8*8划分，然后拉直进行后续操作，所以这里和VIT还不一样
        # 后来的ConvNext也是和这个操作一样。
        # (B , 3 , 224 ,224) - > (B , 96 , 56 ,56 )
        if norm_layer is not None:
            # default nn.LayerNorm
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C

        # ** flatten(2) , 2 means that start flatten dim is 2 , which is w and h **
        # input ( B , 3 , 224 , 224) -> ( B , 96 , 56 , 56) -> ( B , 56*56 , 96)

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
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size,the num of patch within this window. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1 ,see https://paperswithcode.com/method/stochastic-depth
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths) # 4 for [2, 2, 6, 2] in [ stage 1,stage 2,stage 3,stage 4]
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # size half and channel twice
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer= norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches # default 56*56
        patches_resolution = self.patch_embed.patches_resolution # default 56
        self.patches_resolution = patches_resolution # default 56

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # Regardless of batch, each block is the same
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # stochastic depth decay rule
        # see https://paperswithcode.com/method/stochastic-depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # (1 - dpr(i))*f_i(x) + x , dpr(i) from 0 to drop_path_rate

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # default 4
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               # [ 96 , 96*2 ,96*4 ,96*8]
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               # input_resolution = [ 56 , 28 , 14 , 7 ]
                               # Corresponding [ H/4 , H/8 , H/16 , H/32 ] ,H = 224

                               depth=depths[i_layer],
                               # [ 2, 2, 6, 2]
                               num_heads=num_heads[i_layer],
                               # [3, 6, 12, 24]
                               window_size=window_size, # 7

                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               # Res drop rate of layers belonging to this block
                               # eg. block 2 ,have 2 layer ,it's drop_path_rate should be [0.02 0.03]
                               norm_layer=norm_layer,
                               downsample= PatchMerging if (i_layer < self.num_layers - 1) else None,
                               # merging the patch Neighbor

                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
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

    def forward_features(self, x):
        x = self.patch_embed(x) # ( B , 56*56 , 96)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x) # classification
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
