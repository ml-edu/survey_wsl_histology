import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, in_channels, classes):
        super(Classification, self).__init__()
        self.classes = classes
        self.conv = nn.Conv2d(in_channels, out_channels=classes, kernel_size=1)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)

class AblationCam(Classification):

    def __init__(self, in_channels, classes):
        super(AblationCam, self).__init__(in_channels, classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eval_cams = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)

        logits = self.pool(out).flatten(1)

        if self.eval_cams:
            class_maps = []
            for c in range(self.classes):
                score = logits[:, c].squeeze()

                cams = out.detach()
                b, m, h, w = cams.size()
                weighted_maps = torch.zeros([b, m, h, w])
                for k in range(m):
                    abl_cams = cams.detach().clone()
                    abl_cams[0][k] = 0

                    abl_logits = self.pool(abl_cams).flatten(1)
                    abl_score = abl_logits[:, c].squeeze()
                    weight = (score - abl_score) / score

                    w_map = weight * cams[0][k]
                    weighted_maps[0][k] = w_map
                saliency_map = F.relu(weighted_maps.sum(1, keepdim=True))
                class_maps.append(saliency_map)

            class_maps = torch.cat(class_maps, 1).cpu()
            self.cam = class_maps


        return logits


class Average(Classification):
    def __init__(self, in_channels, classes):
        super(Average, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        return self.pool(out).flatten(1)


class GAP(Classification):
    def __init__(self, in_channels, classes):
        super(GAP, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pool(out)
        return self.flatten(out)
        # return self.pool(out).flatten(1)

class DrnPool(Classification):

    def __init__(self, in_channels, classes):
        super(DrnPool, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # out = self.softmax(out)
        self.cam = out.detach()
        return self.pool(out).flatten(1)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class Max(Classification):
    def __init__(self, in_channels, classes):
        super(Max, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        return self.pool(out).flatten(1)


class LogSumExp(Classification):
    def __init__(self, in_channels, classes, r=10):
        super(LogSumExp, self).__init__(in_channels, classes)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        m = self.maxpool(out)
        out = self.avgpool((self.r * (out - m)).exp()).log().mul(1 / self.r) + m

        return out.flatten(1)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4)

    modules = [
        Average(in_channels=3, classes=2),
        Max(in_channels=3, classes=2),
        LogSumExp(in_channels=3, classes=2),
    ]

    for m in modules:
        print(m, '\n', x.shape, ' -> ', m(x).shape, '\n', sep='')
