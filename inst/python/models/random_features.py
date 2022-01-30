#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class ConvSubset(nn.Module):
    def __init__(self, patches, device=None):
        super(ConvSubset, self).__init__()
        self.conv = nn.Conv2d(3, len(patches), patches.shape[2], bias=False)
        self.conv.weight = nn.Parameter(patches.to(device))

    def forward(self, x):
        return self.conv(x)


class WideNet(nn.Module):
    def __init__(self, patches):
        super(WideNet, self).__init__()
        self.patches = patches
        self.filter_subset = 1024
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h = []

        for i in range(0, len(self.patches), self.filter_subset):
            h_cur = self.forward_partial(x, i, i + self.filter_subset)
            h.append(h_cur)

        return torch.cat(h, dim=1)

    def forward_partial(self, x, start, end):
        conv = ConvSubset(self.patches[start:end], self.device)
        pre_h = conv(x)
        return F.max_pool2d(F.relu(pre_h - 1), pre_h.shape[2])


def random_patches(paths, k=2048, patch_size=12, im_size=64):
    selected_ = np.random.choice(paths, k)
    p, n = np.unique(selected_, return_counts=True)
    selected = dict(zip(p, n))

    patches = []
    for p, n in selected.items():
        im = np.load(p)
        for i in range(n):
            ix = np.random.randint(0, im_size - patch_size, 2)
            patches.append(im[ix[0]:(ix[0] + patch_size), ix[1]:(ix[1] + patch_size)])

    patches = np.stack(patches, axis=0)
    patches = np.transpose(patches, (0, 3, 1, 2))
    return torch.Tensor(patches)
