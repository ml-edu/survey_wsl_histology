import torch.nn as nn
from collections import OrderedDict
from sacred import Ingredient

import cnn
from .configs import poolings

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    # arch = 'drn_d_22'
    # arch = 'resnet18'
    arch = 'resnet18_vae'
    pretrained = True
    num_classes = 2


@model_ingredient.named_config
def average():
    pooling = 'average'


@model_ingredient.named_config
def max():
    pooling = 'max'


@model_ingredient.named_config
def lse():
    pooling = 'lse'
    r = 10


@model_ingredient.named_config
def wildcat():
    pooling = 'wildcat'
    modalities = 4
    kmax = 0.1
    kmin = kmax
    alpha = 0.6


@model_ingredient.named_config
def deepmil():
    pooling = 'deepmil'
    mid_channels = 128
    gated = False

@model_ingredient.named_config
def drn_pool():
    pooling = "drn_pool"

@model_ingredient.named_config
def deepmil_multi():
    pooling = 'deepmil_multi'
    mid_channels = 128
    gated = False


@model_ingredient.named_config
def gradcampp():
    pooling = 'gradcampp'


@model_ingredient.named_config
def gradcam():
    pooling = 'gradcam'

@model_ingredient.named_config
def ablation():
    pooling = 'ablation'

@model_ingredient.capture
def load_backbone(arch, pretrained):
    cnn_names = sorted(name for name in cnn.__dict__ if name.islower() and not name.startswith("__")
                       and callable(cnn.__dict__[name]))
    if arch not in cnn_names:
        raise ValueError('Invalid choice for architecture - choices: {}'.format(' | '.join(cnn_names)))
    backbone = cnn.__dict__[arch](pretrained=pretrained)

    return backbone


@model_ingredient.capture
def load_average(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module


@model_ingredient.capture
def load_max(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module


@model_ingredient.capture
def load_lse(pooling, in_channels, num_classes, r):
    pooling_module = poolings[pooling](in_channels, num_classes, r=r)

    return pooling_module


@model_ingredient.capture
def load_wildcat(pooling, in_channels, num_classes, modalities, kmax, kmin, alpha):
    pooling_module = poolings[pooling](in_channels, num_classes, modalities, kmax, kmin, alpha)

    return pooling_module


@model_ingredient.capture
def load_deepmil(pooling, in_channels, mid_channels, gated):
    pooling_module = poolings[pooling](in_channels, mid_channels, gated)

    return pooling_module


@model_ingredient.capture
def load_deepmil_multi(pooling, in_channels, mid_channels, num_classes, gated):
    pooling_module = poolings[pooling](in_channels, mid_channels, num_classes, gated)

    return pooling_module

@model_ingredient.capture
def load_gradcam_pp(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module

@model_ingredient.capture
def load_gradcam(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module

@model_ingredient.capture
def load_ablation(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module

@model_ingredient.capture
def load_drn_pool(pooling, in_channels, num_classes):
    pooling_module = poolings[pooling](in_channels, num_classes)

    return pooling_module

_pooling_loaders = {
    'average': load_average,
    'max': load_max,
    'lse': load_lse,
    'wildcat': load_wildcat,
    'deepmil': load_deepmil,
    'deepmil_multi': load_deepmil_multi,
    'gradcam': load_gradcam,
    'gradcampp': load_gradcam_pp,
    'ablation': load_ablation,
    'drn_pool': load_drn_pool
}


@model_ingredient.capture
def load_model(pooling):
    pooling_names = list(poolings.keys())
    if pooling not in pooling_names:
        raise ValueError('Invalid choice for pooling - choices: {}'.format(' | '.join(pooling_names)))

    backbone = load_backbone()
    if isinstance(backbone, cnn.ResNet_VAE):
        # out_channels = backbone.resnet.inplanes
        out_channels = 512
    else:
        out_channels = backbone.inplanes
    pooling_module = _pooling_loaders[pooling](in_channels=out_channels)

    # if pooling == 'gradcampp':
    #     model = GradCAMpp(backbone=backbone, pooling=pooling_module)
    # # elif pooling == 'gradcam':
    # #     model = GradCAM(backbone=backbone, pooling=pooling_module)
    # else:
    #     model = nn.Sequential(OrderedDict([
    #         ('backbone', backbone),
    #         ('pooling', pooling_module)
    #     ]))

    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('pooling', pooling_module)
    ]))

    return model
