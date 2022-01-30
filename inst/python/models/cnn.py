#!/usr/bin/env python
import torch
from torch import nn

class CBRNet(nn.Module):
    def __init__(self, p_in=3, nf=32, trunc_output=False):
        super(CBRNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Three layer fully convolutional net
            nn.Conv2d(p_in, nf, kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf * 2, kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf * 2, nf * 4, kernel_size=(6,5), stride=1, padding=2),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        final_layer = [nn.Linear(in_features=nf * 4, out_features=1, bias=True)]
        self.linear_layers = nn.Sequential(*final_layer)

    def forward_(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)

    # Defining the forward pass
    def forward(self, x):
        return {"y_hat": self.forward_(x)}


def cnn_loss(x, y, output):
    l2 = torch.nn.MSELoss()
    return l2(output["y_hat"], y)
