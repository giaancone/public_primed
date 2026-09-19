"""Kaon/pion separation power -- the drift-chamber evaluation metric.

    S = |mean(counts_pi) - mean(counts_K)| / ((std_pi + std_K) / 2)

`S` is computed from the predicted primary-cluster counts of two samples, then multiplied
by `length_scale` to express it over a full particle track rather than a single drift cell.
That scaling is a plain multiply, not a Poisson rescaling of the means and deviations.

Two properties worth knowing when calling this:

  * `S` is invariant to any common scaling of the counts, since numerator and denominator
    scale together. Predictions may be passed in raw counts or divided by a target divisor.
  * `length_scale` depends on the geometry being quoted -- sqrt(track length / cell size).
    Pass the value from the config's `eval.length_scale`; the module default below is only
    a fallback and will not match a different detector geometry.

Pure ASCII.
"""

import numpy as np

# Fallback scaler only. Callers should pass `length_scale` explicitly from their config.
LEN_SCALER = 10.0 * np.sqrt(2.0)   # ~14.142


def separation_power(counts_pi, counts_k, length_scale=LEN_SCALER):
    """Separation power in sigma.

    Returns a dict with the scaled separation under "separation", the unscaled per-cell
    value, and the per-sample means, deviations and counts that produced them. Returns
    NaN for the separation if both samples have zero spread.
    """
    a = np.asarray(counts_pi, dtype=np.float64).ravel()
    b = np.asarray(counts_k, dtype=np.float64).ravel()

    denom = (a.std() + b.std()) / 2.0
    s = float("nan") if denom <= 0.0 else abs(a.mean() - b.mean()) / denom
    scaled = s * float(length_scale)

    return {
        "separation_unscaled": float(s),       # per-cell, before the track scaler
        "separation": float(scaled),           # the reported value
        "length_scale": float(length_scale),
        "mean_pi": float(a.mean()), "mean_k": float(b.mean()),
        "std_pi": float(a.std()), "std_k": float(b.std()),
        "n_pi": int(a.size), "n_k": int(b.size),
    }
