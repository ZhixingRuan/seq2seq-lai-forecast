"""Customized loss -- compute loss where LAI_flag == 1 (real valid obs)

Ignore:
├── flag=2  made-up daily timesteps (upsampled)
├── flag=0  invalid pixels (cloud/water)

"""

import torch
import torch.nn as nn


def masked_mse_loss(y_pred, y_true, flag_y, valid_flag=1):
    """
    Compute MSE loss only on valid observation pixels.

    Args:
        y_pred:     (batch, seq_len, lat, lon)  model predictions
        y_true:     (batch, seq_len, lat, lon)  target LAI values
        flag_y:     (batch, seq_len, lat, lon)  target sequence flags: 1=valid, 2=carried, 0=invalid
        valid_flag: which flag value to supervise on (default=1)

    Returns:
        scalar loss
    """
    # Build mask: unsqueeze channel dim to match (batch, seq, 1, H, W)
    mask = (flag_y == valid_flag).float().unsqueeze(2)

    # Check we have valid pixels to compute loss on
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Squared error, zeroed out where mask=0
    squared_error = (y_pred - y_true) ** 2

    # Mean only over valid pixels
    denom = mask.sum().clamp(min=1)
    loss = (squared_error * mask).sum() / denom

    return loss


def masked_metrics(y_pred, y_true, flag_y, valid_flag=1):
    """
    Returns MSE, MAE, and RMSE on valid pixels only.
    Useful for tracking during training and evaluation.
    """
    mask   = (flag_y == valid_flag).float().unsqueeze(2)
    n_valid = mask.sum()

    if n_valid == 0:
        zero = torch.tensor(0.0)
        return {"mse": zero, "mae": zero, "rmse": zero, "n_valid": 0}

    diff    = (y_pred - y_true) * mask

    mse  = (diff ** 2).sum() / n_valid
    mae  = diff.abs().sum()  / n_valid
    rmse = mse.sqrt()

    return {
        "mse"    : mse,
        "mae"    : mae,
        "rmse"   : rmse,
        "n_valid": int(n_valid.item())
    }