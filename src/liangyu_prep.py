"""Input preprocessing for the small student: decimation and min-max scaling.

The student consumes a downsampled, [0, 1]-scaled waveform rather than the raw full-rate
trace the teacher sees. This module provides those two steps as plain numpy, with no torch
and no network, so preprocessing can be applied independently of training.

The scaler is fit on the Training split only and applied unchanged to validation and test.
Refitting per split would leak the evaluation range into the input transform.

Pure ASCII.
"""
import numpy as np

N_PRETRIGGER = 200          # length of the pre-trigger region, in samples


def downsample(X, factor, mode="stride"):
    """Decimate a batch of waveforms `(n, L)` by `factor`.

    mode="stride" (default)
        `X[:, ::f]` -- keeps one sample in every `f` and discards the rest.

    mode="max"
        Maximum over each window of `f`. Same output width and therefore the same
        downstream parameter count, and it needs no multiplications (a comparator, a
        register and a mod-`f` counter), but it preserves a narrow peak located anywhere
        in the window. Stride can drop such a peak entirely, which matters when the target
        depends on peak structure rather than on an integral.

    Output width is `ceil(L / f)` in both modes, so they are drop-in swappable. When
    `L % f != 0` the final partial window is edge-padded with its last sample.
    """
    f = int(factor)
    X = np.asarray(X, dtype=np.float32)
    if f <= 1:
        return X
    if str(mode).lower() != "max":
        return X[:, ::f]
    n, L = X.shape
    n_out = -(-L // f)                       # ceil, to match stride's width exactly
    pad = n_out * f - L
    if pad:
        X = np.concatenate([X, np.repeat(X[:, -1:], pad, axis=1)], axis=1)
    return X.reshape(n, n_out, f).max(axis=2).astype(np.float32)


def fit_minmax(X, per_feature=False):
    """Fit a min-max scaler on the training split. Returns a scaler dict.

    `per_feature=False` (default) takes one global range over all samples and time bins.
    `per_feature=True` takes a separate range per time bin.
    """
    X = np.asarray(X, dtype=np.float32)
    ax = 0 if per_feature else None
    lo = X.min(axis=ax); hi = X.max(axis=ax)
    return {"lo": np.atleast_1d(lo).astype(np.float32),
            "hi": np.atleast_1d(hi).astype(np.float32),
            "per_feature": bool(per_feature)}


def apply_minmax(X, sc):
    """Map to [0, 1] using a previously fit scaler.

    Values outside the training range are not clipped: validation and test data may
    legitimately exceed it, and clipping would mask distribution shift. Constant features
    (hi == lo) map to 0 instead of dividing by zero.
    """
    X = np.asarray(X, dtype=np.float32)
    lo, hi = sc["lo"], sc["hi"]
    rng = np.where((hi - lo) == 0, 1.0, (hi - lo))
    return ((X - lo) / rng).astype(np.float32)
