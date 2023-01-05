import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.channel = 3
        self.latent_z_size = 100
        self.g_maps = 64
        self.d_maps = 64
        self.g = nn.Sequential(
            #input is Z (1,1)
            nn.ConvTranspose2d(in_channels=self.latent_z_size,out_channels=self.g_maps*8,kernel_size=8,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(num_features=self.g_maps*8),
            nn.ReLU(True),
            #(1,1)->(8,8)

            nn.ConvTranspose2d(in_channels=self.g_maps*8, out_channels=self.g_maps*4, kernel_size=4,stride=4,padding=0,bias=False),
            nn.BatchNorm2d(self.g_maps*4),
            nn.ReLU(True),
            #(8,8)->(32,32)

            nn.ConvTranspose2d(in_channels=self.g_maps*4,out_channels=self.g_maps*2,kernel_size=2,stride=2,padding=0,bias=False),
            nn.BatchNorm2d(num_features=self.g_maps*2),
            nn.ReLU(True),
            #(32,32)->(64,64)

            nn.ConvTranspose2d(in_channels=self.g_maps*2, out_channels=self.g_maps,kernel_size=2,stride=2,padding=0,bias=False),
            nn.BatchNorm2d(self.g_maps),
            nn.ReLU(True),
            #(64,64)->(128,128)

            nn.ConvTranspose2d(in_channels=self.g_maps,out_channels=self.channel,kernel_size=2,stride=2,padding=0,bias=False),
            nn.Tanh()
            #(128,128)->(256,256)
        )
    
    def forward(self, x):
        netG=self.g(x)   
        return netG

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channel = 3
        self.latent_z_size = 100
        self.g_maps = 64
        self.d_maps = 64
        self.D = nn.Sequential(
            #input : (256,256)
            nn.Conv2d(in_channels=self.channel,out_channels=self.d_maps,kernel_size=6,stride=4,padding=1,bias=False), #(256,256)->(64,64)
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=self.d_maps,out_channels=self.d_maps*2,kernel_size=4,stride=4,padding=0,bias=False),#(64,64)->(16,16)
            nn.BatchNorm2d(num_features=self.d_maps*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=self.d_maps*2,out_channels=self.d_maps*4,kernel_size=4,stride=2,padding=1,bias=False),#(16,16)->(8,8)
            nn.BatchNorm2d(num_features=self.d_maps*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=self.d_maps*4,out_channels=self.d_maps*8,kernel_size=4,stride=2,padding=1,bias=False),#(8,8)->(4,4)
            nn.BatchNorm2d(num_features=self.d_maps*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(self.d_maps*8,out_channels=1,kernel_size=4,stride=1,padding=0,bias=False),#(4,4)->(1,1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        netD=self.D(x)
        return netD