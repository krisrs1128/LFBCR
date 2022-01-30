#!/usr/bin/env python

def model_outputs(model, loader, output_fn=output_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    derived_data = []
    for x, y in loader:
        # use GPU
        x = x.to(device)
        y = y.to(device)

        # setup model
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            raw_output = model(x)
            o = output_fn(x, y, raw_output)
            derived_data.append(o)

    return derived_data


def cnn_preds(x, y, output):
    return {"y": y, "y_hat": output["y_hat"]}

def vae_x_hat(x, y, output):
    return {"x": x, "x_hat": output["x_hat"]}

def vae_loss:
    pass
