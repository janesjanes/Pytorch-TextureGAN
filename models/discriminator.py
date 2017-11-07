import torch
import torch.nn as nn
import torch.legacy as legacy
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf, use_sigmoid):
        super(Discriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.res_block = ResidualBlock

        self.model = self.create_discriminator(use_sigmoid)

    def create_discriminator(self, use_sigmoid):
        norm_layer = self.batch_norm
        ndf = self.ndf  # 32
        self.res_block = ResidualBlock
        
        sequence = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 2, self.ndf * 8, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            
            self.res_block(self.ndf * 8, self.ndf * 8),
            self.res_block(self.ndf * 8, self.ndf * 8),

            nn.Conv2d(self.ndf * 8, self.ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 4, 1, kernel_size=4, stride=2, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class LocalDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, use_sigmoid):
        super(LocalDiscriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.res_block = ResidualBlock

        self.model = self.create_discriminator(use_sigmoid)

    def create_discriminator(self, use_sigmoid):
        norm_layer = self.batch_norm
        ndf = self.ndf  # 32
        self.res_block = ResidualBlock
        
        sequence = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=3, stride=2, padding=1),nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 4, kernel_size=3, stride=2, padding=1),nn.InstanceNorm2d(ndf* 4),
            nn.LeakyReLU(0.2, True),

            #nn.Conv2d(self.ndf * 2, self.ndf * 8, kernel_size=5, stride=2, padding=1),
            #nn.LeakyReLU(0.2, True),
            #nn.Dropout(0.2),
            
            self.res_block(self.ndf * 4, self.ndf * 4),
            self.res_block(self.ndf * 4, self.ndf * 4),

            nn.Conv2d(self.ndf * 4, self.ndf * 2, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(ndf* 2),
            #nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 2, 1, kernel_size=3, stride=2, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), norm_layer(ndf * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                      padding=padw), norm_layer(ndf * nf_mult,
                                                affine=True), nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, dilation=dilation)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride, 
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

