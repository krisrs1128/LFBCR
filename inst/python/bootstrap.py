"""
Organize and Launch many VAE runs

python3 -m learning.bootstrap -c conf/train.yaml
"""
import pandas as pd
import numpy as np
import argparse
import pathlib
import os
from addict import Dict
import yaml

def bootstrap_indices(N, B=30, out_path="./bootstraps.csv"):
    result = np.zeros((B, N))
    for b in range(B):
        result[b, :] = np.random.choice(range(N), N)

    os.makedirs(out_path.parent, exist_ok=True)
    result = result.astype(int)
    pd.DataFrame(result).to_csv(out_path, index=False)
    return result
