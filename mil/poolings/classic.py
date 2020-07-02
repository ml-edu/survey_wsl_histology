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

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class GradCamPP(Classification):
    def __init__(self, in_channels, classes):
        super(GradCamPP, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        last_conv = self.conv

        last_conv.register_forward_hook(forward_hook)
        last_conv.register_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        logits = self.pool(x).flatten(1)

        score = logits[:, logits.max(1)[-1]].squeeze()

        if not self.training:
            self.zero_grad()
            score.backward(retain_graph=False)
            gradients = self.gradients['value']  # dS/dA
            activations = self.activations['value']  # A
            b, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                          activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom + 1e-7)
            positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            # saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

            self.cam = saliency_map

        return logits


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
