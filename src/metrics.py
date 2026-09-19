"""Shared regression metrics and the per-target weighted loss.

Both the teacher and the student paths report MAE / R^2 on a validation split, and
the DRO objective weights its three targets unequally. Those two pieces live here so
the teacher, the from-scratch student and the distilled student all compute them the
same way -- a difference in either would make their numbers incomparable.

Pure ASCII.
"""

import numpy as np
import torch
import torch.nn as nn


def regression_metrics(pred, true):
    """MAE and R^2 over a flat array of predictions."""
    pred, true = np.asarray(pred), np.asarray(true)
    mae = float(np.mean(np.abs(pred - true)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    return {"mae": mae, "r2": 1.0 - ss_res / ss_tot}


def make_loss(target_weights, device):
    """MSE, optionally weighted per target.

    DRO uses [4.0, 1.0, 0.3] over [c, s, t0] so the low-signal timing target does not
    dominate the energy fit. `None` gives plain unweighted MSE, which is the DCH case
    and any other single-target head.

    The weight and the target divisor are one decision, not two. Targets are
    divided by `target_divisors` at load time and the divisor enters this loss squared,
    so the effective weight on target i is w_i * (mean_i / D_i)^2. Changing a divisor
    without re-deriving the weights silently reweights the objective, and err68 is
    divisor-invariant, so every reported number still looks correct.
    """
    if target_weights is None:
        return nn.MSELoss()
    w = torch.as_tensor(target_weights, dtype=torch.float32, device=device).reshape(1, -1)

    def weighted_mse(pred, true):
        return (w * (pred - true) ** 2).mean()
    return weighted_mse
