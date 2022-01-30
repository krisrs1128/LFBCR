#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import torch

def train_elnet(model, loader, **kwargs):
    D, y = random_features(model, loader)
    elnet_model = lm.ElasticNet(**kwargs)
    elnet_model.fit(X = D, y = y)
    y_hat = elnet_model.predict(X = D)
    return elnet_model, (D, y), y_hat


def random_features(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D, y = [], []
    for x, yi in loader:
        x = x.to(device)
        with torch.no_grad():
            D.append(model(x).cpu())
            y.append(yi)

    D = torch.cat(D).squeeze()
    y = torch.cat(y).squeeze()
    return D, y


def predict_rcf(model, elnet_model, loader):
    D, y = random_features(model, loader)
    y_hat = elnet_model.predict(X = D)
    return D, y_hat, y


def train_rcf(model, loaders, out_paths, **kwargs):
    metadata, errors = [], {}
    elnet_model, Dy, y_hat = train_elnet(model, loaders["train"], **kwargs)
    torch.save(model.state_dict(), out_paths[2])
    np.save(f"{(str(out_paths[2])).replace('pt', '')}-patches.npy", elnet_model.coef_)

    # get errors
    errors["train"] = np.mean((Dy[1].cpu().numpy() - y_hat) ** 2)
    for split in ["dev", "test"]:
        _, y_hat, y = predict_rcf(model, elnet_model, loaders[split])
        errors[split] = np.mean((y.cpu().numpy() - y_hat) ** 2)
    json.dump(errors, open(out_paths[0] / "logs" / "errors.json", "w"))

    # save full feature set
    D, _ = random_features(model, loaders["features"])
    k_path = Path(out_paths[0]) / "full_best.npy"
    np.save(k_path, D.cpu().numpy())
    metadata.append({"epoch": "best", "layer": "full", "out_path": k_path})

    # save features reweighted by coefficient
    k_path = Path(out_paths[0]) / "selected_best.npy"
    pos_ix = np.where(np.abs(elnet_model.coef_) > 0)[0]
    np.save(k_path, D[:, pos_ix].cpu().numpy())
    metadata.append({"epoch": "best", "layer": "selected", "out_path": k_path})
    pd.DataFrame(metadata).to_csv(out_paths[1])
