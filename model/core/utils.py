
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        # uses upsample then conv to avoid checkerboard artifacts
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)
    
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)

class ResBlock(TimestepBlock):
    def __init__(self,in_channels,time_embed_dim,dropout,out_channels=None,use_conv=False,up=False,down=False):
        super(ResBlock,self).__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
            GroupNorm32(32,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,3,padding=1)
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels,False)
            self.x_upd = Upsample(in_channels,False)
        elif down:
            self.h_upd = Downsample(in_channels,False)
            self.x_upd = Downsample(in_channels,False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim,out_channels)
        )
        self.out_layers  = nn.Sequential(
            GroupNorm32(32,out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class QKVAttention(nn.Module):
    def __init__(self,n_heads):
        super().__init__()
        self.n_heads = n_heads
    def forward(self,qkv,time=None):
        bs,width,length = qkv.shape
        assert width % (3*self.n_heads)==0
        ch = width // (3*self.n_heads)
        q,k,v = qkv.reshape(bs * self.n_heads,ch*3,length).split(ch,dim=1)
        scale = 1/math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
                )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(),dim=1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
    
class AttentionBlock(nn.Module):
    def __init__(self,in_channels,n_heads=1,n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32,self.in_channels)
        if n_head_channels==-1:
            self.num_heads = n_heads
        else:
            assert(in_channels%n_head_channels==0), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels
        self.to_qkv = nn.Conv1d(in_channels,in_channels*3,1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels,in_channels,1))
    def forward(self,x,time=None):
        b,c, *spatial = x.shape
        x = x.reshape(b,c,-1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x+h).reshape(b,c,*spatial)
    
