import torch
from torch import nn
from cnn import resnet18
import torch.nn.functional as F

class ResNet_VAE(nn.Module):

    def __init__(self, pretrained):
        super(ResNet_VAE, self).__init__()

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        self.resnet = resnet18(pretrained=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1,
                               padding=self.pd4),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.out_mu = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.out_logvar = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        # (256, 8, 8) -> (3, 256, 256)
        self.decoder = nn.Sequential(
            #  (256, 8, 8) -> (256, 16, 16)
            UpsamplingLayer(512, 256, activation="ReLU"),
            # -> (128, 32, 32)
            UpsamplingLayer(256, 128, activation="ReLU"),
            # -> (64, 64, 64)
            UpsamplingLayer(128, 64, activation="ReLU", type="upsample"),
            # -> (32, 128, 128)
            UpsamplingLayer(64, 32, activation="ReLU", bn=False),
            # -> (3, 256, 256)
            # UpsamplingLayer(32, 3, activation="none", bn=False, type="upsample"),
            # nn.Tanh()
            # nn.Hardtanh(-1.0, 1.0),
        )
        self.conv_last = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Hardtanh(-1.0, 1.0)

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = self.conv1(x)
        mu = self.out_mu(x)
        logvar = self.out_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, out_shape):
        x = self.decoder(z)
        x = F.interpolate(x, size=out_shape[-2:], mode='bilinear', align_corners=True)
        x = self.conv_last(x)
        x = self.tanh(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z, x.shape)
        return z, x_reconst, mu, logvar

    # def loss_function(self, x, x_hat, mu, logvar):
    #     recon_loss = torch.nn.MSELoss(x, x_hat, self.recon_loss_type)
    #     kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
    #     return kl_loss + recon_loss

class UpsamplingLayer(nn.Module):
    def __init__(self, in_channel, out_channel, activation="none", bn=True, type="transpose"):
        super(UpsamplingLayer, self).__init__()
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
        if activation == "ReLU":
            self.activaton = nn.ReLU(True)
        elif activation == "none":
            self.activaton = None
        else:
            assert()
        if type == "transpose":
            self.upsampler = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),
            )
        elif type == "upsample":
            self.upsampler = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1), 
            )
        else:
            assert()

    def forward(self, x):
        x = self.upsampler(x)
        if self.activaton:
            x = self.activaton(x)
        if self.bn:
            x = self.bn(x)
        return x

def resnet18_vae(pretrained=False, **kwargs):
    return ResNet_VAE(pretrained=pretrained)

if __name__ == '__main__':
    x = torch.randn(5, 3, 544, 775)

    model = resnet18_vae(pretrained=True)

    z, x_reconst, mu, logvar = model(x)
    print(x_reconst.shape, z.shape)