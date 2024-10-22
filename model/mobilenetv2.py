import torch

from timm.models import register_model

@register_model
def mobilenetv2_x1_0(num_classes, **kwargs):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=True)
    return model