import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=21, initial_filter_size=64, kernel_size=3, do_instancenorm=True, dropout=0.0):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, stride=(1,3), instancenorm=do_instancenorm, dropout=dropout)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, stride=(1,1), instancenorm=do_instancenorm, dropout=dropout)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm, dropout=dropout)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm, dropout=dropout)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm, dropout=dropout)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm, dropout=dropout)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm, dropout=dropout)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm, dropout=dropout)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**2, initial_filter_size*2**2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**2, initial_filter_size*2**2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2**2, 2, stride=2, output_padding=(0,1)),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3, dropout=dropout)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3, dropout=dropout)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2, dropout=dropout)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2, dropout=dropout)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2, output_padding=(1,0))

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2, dropout=dropout)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2, dropout=dropout)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=(2, 2), output_padding=(1,1))

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size, dropout=dropout)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size, dropout=dropout)
        
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(initial_filter_size, initial_filter_size, 3, stride=(1,3), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        
        # Output layer for target
        self.final = nn.Conv2d(initial_filter_size, 1, kernel_size=1)


    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, stride=1, instancenorm=True, dropout=0.0):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.Dropout2d(dropout, inplace=True),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3, dropout=0.0):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.Dropout2d(dropout, inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            )
        return layer

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        center = self.center(pool) # actually there's an upsampling atrous conv here, watch out

        concat = torch.cat([center, contr_3*concat_weight], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        concat = torch.cat([upscale, contr_2*concat_weight], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        concat = torch.cat([upscale, contr_1*concat_weight], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))
        upscale = self.upscale1(expand)

        output = self.final(upscale)

        return output