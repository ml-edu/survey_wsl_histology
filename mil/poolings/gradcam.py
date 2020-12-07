import torch
import torch.nn as nn
import torch.nn.functional as F
from .classic import Classification

# Implementations adapted from https://github.com/1Konny/gradcam_plus_plus-pytorch

class GradCamPooling(Classification):

    def __init__(self, in_channels, classes):
        super(GradCamPooling, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eval_cams = False

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
        b, c, h, w = x.shape

        out = self.conv(x)
        logits = self.pool(out).flatten(1)

        if self.eval_cams:
            class_maps = []
            for i, c in enumerate(range(self.classes)):
                if i > 0:
                    out = self.conv(x)
                    logits = self.pool(out).flatten(1)


                score = logits[:, c].squeeze()

                self.zero_grad()

                if i == 0:
                    score.backward(retain_graph=True)
                else:
                    score.backward(retain_graph=False)

                gradients = self.gradients['value']  # dS/dA
                activations = self.activations['value']  # A
                b, k, u, v = gradients.size()

                alpha = gradients.view(b, k, -1).mean(2)
                weights = alpha.view(b, k, 1, 1)

                saliency_map = (weights * activations).sum(1, keepdim=True)
                saliency_map = F.relu(saliency_map)

                class_maps.append(saliency_map)

            class_maps = torch.cat(class_maps, 1).cpu()
            self.cam = class_maps

        return logits

class GradCamPlusPooling(GradCamPooling):

    def __init__(self, in_channels, classes):
        super(GradCamPlusPooling, self).__init__(in_channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        out = self.conv(x)
        logits = self.pool(out).flatten(1)

        if self.eval_cams:
            class_maps = []
            for i, c in enumerate(range(self.classes)):

                if i > 0:
                    out = self.conv(x)
                    logits = self.pool(out).flatten(1)

                score = logits[:, c].squeeze()

                self.zero_grad()

                if i == 0:
                    score.backward(retain_graph=True)
                else:
                    score.backward(retain_graph=False)

                gradients = self.gradients['value']  # dS/dA
                activations = self.activations['value']  # A
                b, k, u, v = gradients.size()

                alpha_num = gradients.pow(2)
                global_sum = activations.view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
                alpha_denom = gradients.pow(2).mul(2) + global_sum.mul(gradients.pow(3))
                alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

                alpha = alpha_num.div(alpha_denom + 1e-7)
                positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
                weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

                saliency_map = (weights * activations).sum(1, keepdim=True)
                saliency_map = F.relu(saliency_map)

                class_maps.append(saliency_map)

            class_maps = torch.cat(class_maps, 1).cpu()

            self.cam = class_maps

        return logits