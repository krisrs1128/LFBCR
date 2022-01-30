"""
Save features from trained model

Usage:
python3 -m features -c ../conf/train.yaml
"""
import numpy as np
import os
from addict import Dict
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import pandas as pd
import torch
import yaml


def load_checkpoint(model, path, load_keys=None):
    """
    Load from a subset of keys
    """
    if torch.cuda.is_available():
        pretrained = torch.load(path)
    else:
        pretrained = torch.load(path, map_location=torch.device("cpu"))

    state = model.state_dict()
    if load_keys is None:
      load_keys = state.keys()

    state_subset = {k: v for k, v in prtrained.items() if k in load_keys}
    state.update(state_subset)
    self.load_state_dict(state)


def loader_activations(loader, prefixes, device):
    h = {}
    for k in prefixes.keys():
        h[k] = []

    # loop over layers and then over samples
    for k in prefixes.keys():
        prefixes[k].to(device)

        for x, _ in loader:
            x = x.to(device)

            with torch.no_grad():
                h[k].append(prefixes[k](x).cpu())

        h[k] = torch.cat(h[k])

    return h


def vae_prefixes(model):
    return {
        #"layer_1": model.encoder[0],
        #"layer_2": model.encoder[:2], # remove to reduce space
        #"layer_3": model.encoder[:4],
        #"layer_4": model.encoder[:6],
        "mu": model.encoder_avg
    }


def cbr_prefixes(model):
    return {
        #"layer_1": model.cnn_layers[:3],
        #"layer_2": model.cnn_layers[:7],
        #"layer_3": model.cnn_layers[:11],
        #"layer_4": model.cnn_layers[:15],
        "linear": model.cnn_layers
    }


def save_features(loader, model, epoch, out_paths, device):
    if "VAE" in str(model.__class__):
        prefixes = vae_prefixes(model)
    else:
        prefixes = cbr_prefixes(model)

    # save these activations
    h = loader_activations(loader, prefixes, device)
    metadata = []
    for k in h.keys():
        k_path = Path(out_paths[0]) / f"{k}_{str(epoch)}.npy"
        if not k_path.parent.exists():
            k_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(k_path, h[k].detach().cpu().numpy())
        metadata.append({"epoch": epoch, "layer": k, "out_path": k_path})

    # save relevant metadata
    metadata = pd.DataFrame(metadata)
    if Path(out_paths[1]).exists():
        metadata.to_csv(out_paths[1], mode="a", header=False)
    else:
        metadata.to_csv(out_paths[1])

    return h
