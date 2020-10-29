import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from cnn.resnet import resnet18


class ResNet_VAE(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet_VAE, self).__init__()

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        self.resnet = resnet18(pretrained=pretrained)
        self.mu_encoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.logvar_encoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),    # y = (y1, y2, y3) \in [0 ,1]^3
        )

        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),  # y = (y1, y2, y3) \in [0 ,1]^3
        )

        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),  # y = (y1, y2, y3) \in [0 ,1]^3

        )
        #
        # self.convTrans11 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=self.k2, stride=self.s2,
        #                        padding=self.pd2),
        #     nn.BatchNorm2d(3, momentum=0.01),
        #     nn.Sigmoid(),  # y = (y1, y2, y3) \in [0 ,1]^3
        # )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, out_shape):
        x = self.convTrans6(z)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = self.convTrans9(x)
        x = self.convTrans10(x)
        x = F.interpolate(x, size=out_shape, mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z, x.shape[-2:])
        return z, x_reconst, mu, logvar


def resnet18_vae(pretrained=False, **kwargs):
    return ResNet_VAE(pretrained=pretrained)


if __name__ == '__main__':
    x = torch.randn(5, 3, 416, 416)

    model = ResNet_VAE()

    z, x_reconst, mu, logvar = model(x)
    print(x_reconst.shape)
