"""Supervised restoration losses (Eq. 8)."""

import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """0.8 * L1 + 0.2 * L2."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, prediction, target):
        return 0.8 * self.l1(prediction, target) + 0.2 * self.l2(prediction, target)


class VGGPerceptualLoss(nn.Module):
    """Mean MSE over the relu1_2, relu2_2, and relu3_3 features of VGG16."""

    def __init__(self, vgg_model):
        super().__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, prediction, target):
        losses = [F.mse_loss(p, t) for p, t in zip(self.output_features(prediction), self.output_features(target))]
        return sum(losses) / len(losses)
