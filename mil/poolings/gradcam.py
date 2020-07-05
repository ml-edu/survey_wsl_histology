import torch
import torch.nn as nn
import torch.nn.functional as F
from .classic import Classification

class GradCAM(nn.Module):

    def __init__(self, backbone, pooling):
        super(GradCAM, self).__init__()

        self.backbone = backbone
        self.pooling = pooling

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        last_conv = backbone.layer4

        last_conv.register_forward_hook(forward_hook)
        last_conv.register_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _, _, H, W = x.shape

        out = self.backbone(x)
        logits = self.pooling(out)

        if not self.training:
            class_maps = []
            for c in range(self.pooling.classes):

                if c > 0:
                    out = self.backbone(x)
                    logits = self.pooling(out)

                score = logits[:, c].squeeze()

                self.backbone.zero_grad()
                self.pooling.zero_grad()
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
                # saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                # saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
                class_maps.append(saliency_map)

            class_maps = torch.cat(class_maps, 1).cpu()

            saliency_map_min, saliency_map_max = class_maps.min(), class_maps.max()
            saliency_map = (class_maps - saliency_map_min).div(saliency_map_max - saliency_map_min).data

            self.pooling.cam = saliency_map

        return logits