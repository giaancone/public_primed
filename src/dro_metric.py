"""Dual-readout evaluation metrics: err68 and a timing resolution.

    keep  = target > 0
    rel   = |pred - target| / target * 100        # per-event relative error, %
    err68 = percentile(rel, 68)

Reported per regression component (c, s) and, when both are present, on the combined
C/S ratio R = c / s. Lower is better.

err68 is invariant to the target normalization divisor, since the divisor cancels in a
relative error. A number computed on divisor-normalized targets is therefore directly
comparable to one computed in physical units. The timing target t0 is the exception and
is handled separately -- see `resolution68`.

Pure ASCII.
"""

import numpy as np

# Published reference values for this detector configuration, used only to annotate console
# output during training. Not used in any computation. t0 has no entry: it is scored as a
# timing resolution rather than a relative error.
BENCHMARK = {"c": 1.65, "s": 2.5, "ratio": 4.18}


def fmt(name, value):
    """Format a metric with its unit: t0 is a resolution in ns, the rest are percentages."""
    return "t0=%.3fns" % value if name == "t0" else "%s=%.2f%%" % (name, value)


def err68(pred, target):
    """68th-percentile relative error (%), over events with target > 0."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    keep = t > 0
    if not np.any(keep):
        return float("nan")
    rel = np.abs(p[keep] - t[keep]) / t[keep] * 100.0
    return float(np.percentile(rel, 68))


def err68_ratio(pred_c, pred_s, target_c, target_s):
    """68th-percentile relative error (%) of the ratio R = c / s.

    Events are kept only where `target_c > 0`, `target_s > 0` and `pred_s != 0`. Zeros are
    dropped rather than guarded with an epsilon, so a near-zero denominator cannot inflate
    the percentile. Any residual inf/nan is removed before the percentile is taken. Returns
    NaN if no events survive.
    """
    pc = np.asarray(pred_c, dtype=np.float64).ravel()
    ps = np.asarray(pred_s, dtype=np.float64).ravel()
    tc = np.asarray(target_c, dtype=np.float64).ravel()
    ts = np.asarray(target_s, dtype=np.float64).ravel()
    keep = (tc > 0.0) & (ts > 0.0) & (ps != 0.0)
    if not np.any(keep):
        return float("nan")
    r_pred = pc[keep] / ps[keep]
    r_true = tc[keep] / ts[keep]
    rel = np.abs(r_pred - r_true) / r_true * 100.0
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return float("nan")
    return float(np.percentile(rel, 68))


def resolution68(pred, target):
    """68th-percentile absolute error -- the timing analogue of err68.

    Used for t0, whose value passes through zero, making a relative error meaningless.
    Returned in the same units as the inputs; multiply by the target divisor to get ns.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    keep = np.isfinite(p) & np.isfinite(t)
    if not np.any(keep):
        return float("nan")
    return float(np.percentile(np.abs(p[keep] - t[keep]), 68))


def dro_metrics(pred, target, target_names, divisors=None):
    """All metrics for one set of predictions.

    c and s are scored with `err68`, plus the C/S ratio when both are present. t0 is scored
    with `resolution68` and converted to ns if `divisors` is supplied.

    `pred` and `target` are `(N, n_targets)` in the same units. Returns a flat dict, e.g.
    `{"c": 1.7, "s": 2.6, "ratio": 4.3, "t0": 0.21}`.

    One common event mask is applied to every component. When both c and s are
    present, events with a non-finite prediction or a zero true c or s are dropped once, and
    every component including t0 is then scored on that same surviving set. Masking each
    component separately would score them on different event sets, so the numbers in the
    returned dict would not refer to the same events.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if "c" in target_names and "s" in target_names:
        ci, si = target_names.index("c"), target_names.index("s")
        keep = (np.isfinite(pred[:, ci]) & np.isfinite(pred[:, si])
                & (target[:, ci] > 0.0) & (target[:, si] > 0.0))
        pred, target = pred[keep], target[keep]
    out = {}
    for i, name in enumerate(target_names):
        if name == "t0":
            res = resolution68(pred[:, i], target[:, i])
            if divisors is not None:                 # normalized -> physical ns
                res *= float(divisors[i])
            out["t0"] = res
        else:
            out[name] = err68(pred[:, i], target[:, i])
    if "c" in target_names and "s" in target_names:
        ci, si = target_names.index("c"), target_names.index("s")
        out["ratio"] = err68_ratio(pred[:, ci], pred[:, si],
                                    target[:, ci], target[:, si])
    return out
