from collections import defaultdict
import torch


# ── Simple masked RMSE tracker ─────────────────────────────────────────────
class MaskedRMSETracker:
    """
    Accumulates squared errors and counts over valid pixels only.
    Resets after each epoch.
    """
    def __init__(self, device):
        self.device      = device
        self.sum_sq_err  = torch.tensor(0.0).to(device)
        self.sum_abs_err = torch.tensor(0.0).to(device)
        self.n_valid     = torch.tensor(0.0).to(device)

    def update(self, y_pred, y_true, flag, valid_flag=1):
        mask    = (flag == valid_flag).float()
        n       = mask.sum()

        if n == 0:
            return

        diff             = (y_pred - y_true) * mask
        self.sum_sq_err  += (diff ** 2).sum()
        self.sum_abs_err += diff.abs().sum()
        self.n_valid     += n

    def compute(self):
        if self.n_valid == 0:
            return {"rmse": 0.0, "mae": 0.0, "n_valid": 0}
        return {
            "rmse"   : (self.sum_sq_err / self.n_valid).sqrt().item(),
            "mae"    : (self.sum_abs_err / self.n_valid).item(),
            "n_valid": int(self.n_valid.item())
        }

    def reset(self):
        self.sum_sq_err  = torch.tensor(0.0).to(self.device)
        self.sum_abs_err = torch.tensor(0.0).to(self.device)
        self.n_valid     = torch.tensor(0.0).to(self.device)
