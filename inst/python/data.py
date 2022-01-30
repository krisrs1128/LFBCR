"""
Utilities for working with multichannel Tiffs

"""
from pathlib import Path
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import re
import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd


def convert_dir_numpy(input_dir, output_dir):
    """
    Wrap tiff_to_numpy over an entire directory
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = list(Path(input_dir).glob("*.tif"))
    for i, path in enumerate(paths):
        out_name = Path(path).stem + ".npy"
        tiff_to_numpy(path, Path(output_dir, out_name))


def save_pngs(input_dir, output_dir):
    """
    Save arrays as pngs, for easier viewing
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = list(Path(input_dir).glob("*.npy"))
    for i, path in enumerate(paths):
        out_name = Path(path).stem + ".png"
        im = Image.fromarray((255 * np.load(path)).astype(np.uint8))
        im.save(Path(output_dir, out_name))


def random_split(ids, split_ratio):
    """
    Randomly split a list of paths into train / dev / test
    """
    random.shuffle(ids)
    sizes = len(ids) * np.array(split_ratio)
    ix = [int(s) for s in np.cumsum(sizes)]
    splits = {
        "train": ids[: ix[0]],
        "dev": ids[ix[0] : ix[1]],
        "test": ids[ix[1] : ix[2]],
    }

    splits_df = []
    for k in splits.keys():
        for v in splits[k]:
            splits_df.append({"path": v, "split": k})

    return pd.DataFrame(splits_df)


def initialize_loader(paths, data_dir, opts, num_samples=None, **kwargs):
    cell_data = CellDataset(paths, data_dir / opts.organization.xy, data_dir)
    if num_samples is not None:
        cell_data = Subset(cell_data, np.arange(num_samples))

    return DataLoader(cell_data, batch_size=opts.train.batch_size, **kwargs)


class CellDataset(Dataset):
    """
    Dataset for working with simulated cell images
    """
    def __init__(self, img_paths, xy_path=None, root=Path(".")):
        """Initialize dataset."""
        self.img_paths = img_paths
        self.root = root

        # default xy values
        if xy_path:
            self.xy = pd.read_csv(root / xy_path, index_col="path")
        else:
            self.xy = {"y": np.zeros(len(img_paths))}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = np.load(self.root / self.img_paths[i])
        y = self.xy.loc[self.img_paths[i], "y"]

        # random reflections
        for j in range(2):
            if np.random.random() < 0.5:
                img = np.flip(img, axis=j)

        # random 0 / 90 / 180 / 270 rotation
        k = np.random.randint(4)
        for j in range(k):
            img = np.rot90(img, j)

        img = img.transpose(2, 0, 1).copy()
        img = 2 * img - 1
        return torch.Tensor(img), torch.Tensor([y])
