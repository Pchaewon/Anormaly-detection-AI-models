# UNet.py -> AnoDDPM
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.core.utils import *

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        num_res_blocks=2
        in_channels=3
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        biggan_updown=True
        conv_resample=True
        dropout= 0
        channel_mults = (1,1,2,2,4,4)
        attentions_ds = []
        attention_resolutions="32,16,8"
        img_size = 256
        base_channels = 128
        time_embed_dim = base_channels * 4
        n_heads = 4
        n_head_channels = 64
        
        for res in attention_resolutions.split(","):
            attentions_ds.append(img_size//int(res))

        # embedding
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels,1),
            nn.Linear(base_channels,time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim,time_embed_dim)
        )

        ch  = int(channel_mults[0]*base_channels)
        self.down = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))])
        channels = [ch]
        ds = 1

        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,time_embed_dim=time_embed_dim,out_channels=base_channels*mult,dropout=dropout)]
                ch = base_channels * mult

                if ds in attentions_ds:
                    layers.append(AttentionBlock(ch,n_heads=n_heads,n_head_channels=n_head_channels))
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(ch,time_embed_dim=time_embed_dim,out_channels=out_channels,dropout=dropout,down=True)
                                if biggan_updown
                                else
                                Downsample(ch, conv_resample, out_channels=out_channels)
                                ))
                ds *= 2
                ch = out_channels
                channels.append(ch)

        self.middle = TimestepEmbedSequential(ResBlock(ch,time_embed_dim=time_embed_dim,dropout=dropout),
                AttentionBlock(ch,n_heads=n_heads,n_head_channels=n_head_channels),
                ResBlock(ch,time_embed_dim=time_embed_dim,dropout=dropout)
                )
        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [ResBlock(ch + inp_chs,time_embed_dim=time_embed_dim,out_channels=base_channels * mult,dropout=dropout)]
                ch = base_channels * mult
                if ds in attentions_ds:
                    layers.append(AttentionBlock(ch,n_heads=n_heads,n_head_channels=n_head_channels),)

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(ResBlock(ch,time_embed_dim=time_embed_dim,out_channels=out_channels,dropout=dropout,up=True)
                            if biggan_updown
                            else
                            Upsample(ch, conv_resample, out_channels=out_channels)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))
        self.dtype = torch.float32
        self.out = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
                zero_module(nn.Conv2d(base_channels * channel_mults[0], out_channels, 3, padding=1))
                )

    def forward(self, x, time):
        time_embed = self.time_embedding(time)
        
        skips = []
        
        h = x.type(self.dtype)
        for i, module in enumerate(self.down):
            h = module(h, time_embed)
            skips.append(h)
        h = self.middle(h, time_embed)
        for i, module in enumerate(self.up):
            h = torch.cat([h, skips.pop], dim=1)
            h = module(h, time_embed)
        h = h.type(x.dtype)
        h = self.out(h)
        return h

if __name__ == "__main__":
    print("UNet Success!")
