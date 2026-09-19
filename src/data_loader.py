"""Load raw detector waveforms according to a dataset config.

Both detectors are read as raw sampled waveforms rather than any pre-reduced feature map,
because the local peak structure the counting task depends on does not survive reduction.

  DCH   .npz holding `wf_i` of shape (N, 3000). The target is the number of primary
        ionization clusters inside the analysis window:
        `((tag_values == 1) & (tag_times < truncate)).sum(axis=1)`.
  DRO   .h5 holding a summed waveform per event plus a labels array, from which the
        configured `target_indices` are taken and divided by `target_divisors`.

do not AMPLITUDE-NORMALIZE here. The foundation-model backbone applies reversible
instance normalization internally, so the teacher must receive raw samples. The student's
normalization is a separate, train-fitted scaler applied later in the student path.

Pure ASCII.
"""

import glob
import os

import numpy as np


def _resolve(root, pattern):
    """Join a config glob to the data root and expand it (sorted, stable).

    recursive=True so DRO's `**` globs (per-crystal subtrees) expand; harmless
    for DCH's flat globs. If `pattern` is absolute, os.path.join ignores `root`,
    so a config may give either a root-relative or an absolute glob."""
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))


def _dch_window(wf, tt, tv, d):
    """Apply the DCH input window + primary-cluster count. Two modes:
      truncate_samples = <int>  -> truncate to the first N samples (old footing);
                                   count over [0, N) exactly as before (unchanged).
      truncate_samples = null   -> full waveform (1 m footing). Count all primary
                                   clusters (tv==1 & 0<=tt<L, matching the ftpc
                                   peak-count loader), then pad the waveform to a
                                   multiple of pad_multiple (32) so the ftpc teacher's
                                   token guard (tokens*patch == length) is satisfied.
    Returns (processed_wf, count[N] float32)."""
    trunc = d.get("truncate_samples")
    if trunc is not None:
        t = int(trunc)
        wf = wf[:, :t]
        count = ((tv == 1) & (tt < t)).sum(axis=1).astype(np.float32)
    else:
        L = wf.shape[1]
        count = ((tv == 1) & (tt >= 0) & (tt < L)).sum(axis=1).astype(np.float32)
        pm = int(d.get("pad_multiple", 32))
        if L % pm:
            newL = ((L // pm) + 1) * pm
            wf = np.pad(wf, ((0, 0), (0, newL - L)), mode="edge")
    # STUDENT-SIDE downsample (DCH). Off by default (ds=1): the teacher must receive the
    # full-rate waveform, so only the student path sets this.
    #
    # The student's input width is set here and nowhere else, so a run that omits it trains a
    # different architecture than the one being reported. `pool` defaults to "stride"; "max"
    # is the peak-preserving alternative -- see student_prep.downsample for the trade.
    _ds = int(d.get("student_downsample", 1) or 1)
    if _ds > 1:
        from src import student_prep as _sprep
        wf = _sprep.downsample(wf, _ds, mode=str(d.get("student_pool", "stride")))
        wf = np.ascontiguousarray(wf, dtype=np.float32)
    return wf, count


def load_dch(config, root, split="train", max_files=None, max_events=None):
    """Load DCH waveforms + cluster-count targets.

    Returns
    -------
    X : float32 (N, truncate_samples)   truncated raw waveforms
    y : float32 (N, 1)                  primary-cluster counts
    meta : dict                         momenta, file list, truncate window
    """
    d = config["data"]
    pattern = d["train_glob"] if split == "train" else d["test_glob"]
    files = _resolve(root, pattern)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            "No DCH .npz found for split=%s under %s (pattern %r). "
            "Pass --root to point at the directory holding the waveform files."
            % (split, root, pattern)
        )

    wf_key, tt_key, tv_key, mom_key = (
        d["wf_key"], d["tag_times_key"], d["tag_values_key"], d["mom_key"],
    )
    xs, ys, moms = [], [], []
    for f in files:
        with np.load(f) as npz:
            wf = np.asarray(npz[wf_key], dtype=np.float32)      # (n, 3000)
            tt = np.asarray(npz[tt_key])                        # (n, 300)
            tv = np.asarray(npz[tv_key])                        # (n, 300)
            mom = np.asarray(npz[mom_key], dtype=np.float32) if mom_key in npz else None
        wf, count = _dch_window(wf, tt, tv, d)                  # truncate or full+pad
        xs.append(wf)
        ys.append(count)
        moms.append(mom if mom is not None else np.full(len(wf), np.nan, np.float32))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).reshape(-1, 1)
    mom = np.concatenate(moms, axis=0)
    if max_events:
        X, y, mom = X[:max_events], y[:max_events], mom[:max_events]
    return X, y, {"momentum": mom, "files": files, "truncate": d.get("truncate_samples")}


def resolve_dch_files(config, root, split="train", eval_set=None):
    """List raw .npz files for either a data split (train/test) or an eval set
    (pion/kaon, from the config `eval:` globs). Used by the resumable extractor."""
    if eval_set:
        pattern = config["eval"]["%s_glob" % eval_set]
    else:
        pattern = config["data"]["train_glob" if split == "train" else "test_glob"]
    return _resolve(root, pattern)


# backwards-compatible alias
def dch_files(config, root, split="train"):
    return resolve_dch_files(config, root, split=split)


def load_dch_one(config, path, mom_window=None):
    """Load a single .npz -> (X, y, mom). Loading per file allows sharding, so an
    interrupted job loses at most the current file's work.

    mom_window=(momentum, tol) keeps only events with |mom - momentum| <= tol. The
    evaluation productions span a range of momenta while the metric is quoted at one
    momentum, so this window selects the slice being reported."""
    d = config["data"]
    with np.load(path) as npz:
        wf = np.asarray(npz[d["wf_key"]], dtype=np.float32)
        tt = np.asarray(npz[d["tag_times_key"]])
        tv = np.asarray(npz[d["tag_values_key"]])
        mom = (np.asarray(npz[d["mom_key"]], dtype=np.float32)
               if d["mom_key"] in npz else np.full(len(wf), np.nan, np.float32))
    wf, count = _dch_window(wf, tt, tv, d)                      # truncate or full+pad
    count = count.reshape(-1, 1)
    if mom_window is not None:
        m, tol = mom_window
        sel = np.abs(mom - float(m)) <= float(tol)
        wf, count, mom = wf[sel], count[sel], mom[sel]
    return wf, count, mom


def load_dch_concat(config, root, split="train", eval_set=None,
                    max_files=None, max_events=None):
    """Load + concatenate raw waveforms for a split or eval set (pion/kaon,
    momentum-filtered). Used by the `fs` baseline, which needs the raw waveforms
    (not cached embeddings). Returns (X, y, mom)."""
    files = resolve_dch_files(config, root, split=split, eval_set=eval_set)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError("No DCH files (split=%s eval_set=%s) under %s"
                                % (split, eval_set, root))
    mom_window = None
    if eval_set:
        ev = config["eval"]
        mom_window = (float(ev["momentum"]), float(ev.get("mom_tol", 0.25)))

    xs, ys, ms = [], [], []
    for f in files:
        X, y, m = load_dch_one(config, f, mom_window=mom_window)
        if len(X) == 0:
            continue
        xs.append(X); ys.append(y); ms.append(m)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    m = np.concatenate(ms, axis=0)
    if max_events:
        X, y, m = X[:max_events], y[:max_events], m[:max_events]
    return X, y, m


# =========================================================================
# DRO -- dual-readout calorimeter (the global task).
#
# Different output format from DCH: instead of one primary-cluster count, DRO
# regresses the Cherenkov + Scintillation light components [c, s] (and t0) that
# make up one summed crystal waveform. Raw is .h5 (`waveform` (N, ~6246), `labels`
# (N, >=3) = [c, s, t0]). The waveform is downsampled to the benchmark rate, padded or
# truncated to a fixed multiple of the backbone's patch size, and each target is divided
# by a fixed divisor so the MSE is balanced across components.
#
# The divisor is part of the objective, not just A unit change. It enters the loss
# squared, so the effective weight on target i is w_i * (mean_i / D_i)^2. err68 is
# divisor-invariant, so a mis-set divisor skews training while every reported number still
# looks correct -- change `target_divisors` and `target_weights` together or not at all.
# =========================================================================
def _dro_prep_wave(wf, downsample, context_len, pad_mode="edge"):
    """Stride-downsample each waveform then pad/truncate to context_len (a fixed
    multiple of 32). Pads the tail (scintillation slow component lives there, so
    we never truncate it away) with the edge value."""
    wf = np.asarray(wf, dtype=np.float32)
    if downsample and downsample > 1:
        wf = wf[:, ::downsample]
    L = wf.shape[1]
    if L >= context_len:
        wf = wf[:, :context_len]
    else:
        wf = np.pad(wf, ((0, 0), (0, context_len - L)), mode=pad_mode)
    return np.ascontiguousarray(wf, dtype=np.float32)


def _dro_cfg(config):
    d = config["data"]
    ti = list(d["target_indices"])
    div = np.asarray(d.get("target_divisors", [1.0] * len(ti)), dtype=np.float32)
    return (d["wf_key"], d["labels_key"], int(d.get("downsample", 1)),
            int(d["context_len"]), ti, div)


def load_dro_one(config, path, _return_raw_len=False):
    """Load a single DRO .h5 -> (X, y, aux). y = normalized [c, s] targets;
    aux = raw t0 (stashed in the 'momentum' cache slot -- unused by the metric,
    kept so nothing is thrown away and the leakage check has a signal column).

    `_return_raw_len` additionally returns the raw waveform width (before the
    pad/truncate in _dro_prep_wave) so load_dro_concat can refuse a mixed-width
    dataset. Internal; default off so every existing 3-tuple caller is unchanged."""
    import h5py                                 # lazy: Mac selftest never needs it
    wf_key, lab_key, ds, clen, ti, div = _dro_cfg(config)
    with h5py.File(path, "r") as h:
        if wf_key not in h or lab_key not in h:
            raise KeyError("DRO file %s lacks %r and/or %r (has: %s)"
                           % (path, wf_key, lab_key, sorted(h.keys())))
        wf = np.asarray(h[wf_key][:], dtype=np.float32)        # (n, ~6246)
        labels = np.asarray(h[lab_key][:], dtype=np.float32)   # (n, >=3)
    raw_len = int(wf.shape[1]) if wf.ndim == 2 else -1
    # Name the file in the error. The bare IndexError from labels[:, ti] fires after gigabytes
    # are already read and identifies nothing, which costs a debug cycle on a cluster. NOTE we
    # raise rather than zero-pad to 3 columns: padding would train t0 against fabricated
    # zeros, which trains silently and is indistinguishable from a real result.
    if labels.ndim != 2 or labels.shape[1] <= max(ti):
        raise ValueError(
            "DRO file %s has labels with shape %s, but target_indices %s needs at least %d "
            "columns. Refusing to fabricate the missing target(s)."
            % (path, getattr(labels, "shape", None), list(ti), max(ti) + 1))
    X = _dro_prep_wave(wf, ds, clen)
    y = (labels[:, ti] / div).astype(np.float32)               # (n, n_targets)
    aux = (labels[:, 2].astype(np.float32) if labels.shape[1] > 2
           else np.zeros(len(X), np.float32))
    if _return_raw_len:
        return X, y, aux, raw_len
    return X, y, aux


def resolve_dro_files(config, root, split="train", eval_set=None):
    """List raw DRO .h5 files for a split (train/test) or an eval set
    (electron/kaon, from the config `eval:` globs)."""
    if eval_set:
        pattern = config["eval"]["%s_glob" % eval_set]
    else:
        pattern = config["data"]["train_glob" if split == "train" else "test_glob"]
    return _resolve(root, pattern)


def load_dro_concat(config, root, split="train", eval_set=None,
                    max_files=None, max_events=None):
    """Load + concatenate DRO waveforms + [c,s] targets for a split or eval set.
    Used by the `fs` baseline (needs raw waveforms). Returns (X, y, aux)."""
    files = resolve_dro_files(config, root, split=split, eval_set=eval_set)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError("No DRO .h5 files (split=%s eval_set=%s) under %s"
                                % (split, eval_set, root))
    xs, ys, aa = [], [], []
    raw_lens = {}                      # path -> raw waveform width, before pad/truncate
    for f in files:
        X, y, a, raw_len = load_dro_one(config, f, _return_raw_len=True)
        if len(X) == 0:
            continue
        raw_lens[f] = raw_len
        xs.append(X); ys.append(y); aa.append(a)

    # RAW-LENGTH guard -- do not remove.
    # _dro_prep_wave truncates to context_len when the raw waveform is longer and edge-pads when
    # it is shorter, and it is applied per file above -- so np.concatenate (the one thing that
    # would naturally catch a width mismatch) can never fire. Two globs with different raw
    # lengths were therefore harmonized to context_len in total silence.
    #
    # Why that is dangerous, not merely untidy:
    #   * a constant edge-padded tail is a Perfect species tag. A head can read particle type
    #     off the pad boundary and shortcut the C/S regression -- the degenerate-ratio shortcut
    #     already recorded for DRO 2c.
    #   * if a file is longer than context_len it is truncated, which removes the scintillation
    #     slow component that _dro_prep_wave's own docstring promises never to cut.
    # Both are invisible at runtime and would only show up as inexplicably good or bad numbers.
    #
    # Reachable whenever a glob spans productions whose raw waveform lengths differ.
    if len(set(raw_lens.values())) > 1:
        by_len = {}
        for p, L in raw_lens.items():
            by_len.setdefault(L, []).append(os.path.basename(p))
        detail = "; ".join("%d samples: %s%s" % (L, ", ".join(sorted(v)[:3]),
                                                 " (+%d more)" % (len(v) - 3) if len(v) > 3 else "")
                           for L, v in sorted(by_len.items()))
        raise ValueError(
            "DRO raw waveform lengths DISAGREE across the %d matched file(s) -- %s. "
            "_dro_prep_wave would silently pad the short ones and truncate the long ones to "
            "context_len=%d, making the pad boundary a species/file tag and (for the long "
            "files) cutting the scintillation tail. Refusing to build a mixed-width dataset. "
            "Narrow the glob, or resample the files to a common length."
            % (len(raw_lens), detail, _dro_cfg(config)[3]))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    a = np.concatenate(aa, axis=0)
    if max_events:
        # max_events truncates after concatenation, so it takes the first max_events rows
        # in sorted-glob order. With a multi-species glob that is not a random sample: files
        # sort 'BSO_e-_...' < 'BSO_kaon0L_...', so a truncated pool is pure electron until the
        # electron files are exhausted. Measured: --max-events 2000 gives s-mean 1.33 (electron
        # mode) against 1.78 for the full pool. Safe for per-step timing, invalid for anything
        # that needs the real target distribution. Warn rather than silently mislead.
        if max_events < len(X) and len(files) > 1:
            print("[data_loader] WARNING: --max-events %d truncates the FIRST rows in sorted "
                  "file order across %d files, which is NOT a random or stratified sample. "
                  "With a multi-species glob the kept rows may be a single species."
                  % (max_events, len(files)))
        X, y, a = X[:max_events], y[:max_events], a[:max_events]
    return X, y, a


# --- generic dataset dispatch (used by the extract/train scripts) ----------
def input_length(config):
    """Encoder input length for this dataset (DCH truncation vs DRO context)."""
    d = config["data"]
    if config.get("dataset") == "dro":
        return int(d["context_len"])
    return int(d["truncate_samples"])


def resolve_files(config, root, split="train", eval_set=None):
    if config.get("dataset") == "dro":
        return resolve_dro_files(config, root, split=split, eval_set=eval_set)
    return resolve_dch_files(config, root, split=split, eval_set=eval_set)


def load_one(config, path, eval_set=None):
    """Load one raw file -> (X, y, aux), dispatching on dataset. For DCH an
    eval_set applies the momentum window; DRO has no momentum window."""
    if config.get("dataset") == "dro":
        return load_dro_one(config, path)
    mom_window = None
    if eval_set:
        ev = config["eval"]
        mom_window = (float(ev["momentum"]), float(ev.get("mom_tol", 0.25)))
    return load_dch_one(config, path, mom_window=mom_window)


def load_concat(config, root, split="train", eval_set=None,
                max_files=None, max_events=None):
    if config.get("dataset") == "dro":
        return load_dro_concat(config, root, split=split, eval_set=eval_set,
                               max_files=max_files, max_events=max_events)
    return load_dch_concat(config, root, split=split, eval_set=eval_set,
                           max_files=max_files, max_events=max_events)


def peak_target_3class_from_tags(tag_times, tag_values, length):
    """Per-sample 3-class target for the ftpc (peak+count) scheme: (N, length)
    int64 with 0 = noise, 1 = primary, 2 = secondary at each sample -- the dense
    version of tag_values. `(target == 1).sum(axis=1)` is the resolvable primary
    count (the K/pi separation target)."""
    tt = np.asarray(tag_times)
    tv = np.asarray(tag_values)
    L = int(length)
    N = tt.shape[0]
    tgt = np.zeros((N, L), dtype=np.int64)
    for cls in (1, 2):                        # 2 written after 1; overlaps -> 2 wins
        m = (tv == cls) & (tt >= 0) & (tt < L)
        rows, cols = np.nonzero(m)
        if len(rows):
            tgt[rows, tt[rows, cols].astype(np.int64)] = cls
    return tgt


def load_dch_peakcount_one(config, path, mom_window=None):
    """ftpc loader: (X, ps_target(N,L) int {0,1,2}, mom, count(N,1)).

    Honors `truncate_samples` (None/missing -> full non-truncated waveform) and pads
    X + target to a multiple of `pad_multiple`
    (default 32, the TimesFM patch) so tokens tile exactly; padded tail = noise(0)."""
    d = config["data"]
    trunc = d.get("truncate_samples")
    pad_mult = int(d.get("pad_multiple", 32))
    with np.load(path) as npz:
        wf = np.asarray(npz[d["wf_key"]], dtype=np.float32)
        if trunc:
            wf = wf[:, :int(trunc)]
        tt = np.asarray(npz[d["tag_times_key"]])
        tv = np.asarray(npz[d["tag_values_key"]])
        mom = (np.asarray(npz[d["mom_key"]], dtype=np.float32)
               if d["mom_key"] in npz else np.full(len(wf), np.nan, np.float32))
    L = wf.shape[1]
    if L % pad_mult:
        newL = ((L // pad_mult) + 1) * pad_mult
        wf = np.pad(wf, ((0, 0), (0, newL - L)))
        L = newL
    ps = peak_target_3class_from_tags(tt, tv, L)
    # tt>=0 guard so `count` and the per-sample map agree except on the rare two-
    # primaries-on-one-sample case (where count can exceed (ps==1).sum -- inherent
    # to a per-sample representation; sep_peak then mildly undercounts dense events).
    count = ((tv == 1) & (tt >= 0) & (tt < L)).sum(axis=1).astype(np.float32).reshape(-1, 1)
    if mom_window is not None:
        m, tol = mom_window
        sel = np.abs(mom - float(m)) <= float(tol)
        wf, ps, mom, count = wf[sel], ps[sel], mom[sel], count[sel]
    return wf, ps, mom, count


def load_dch_peakcount_concat(config, root, split="train", eval_set=None,
                              max_files=None, max_events=None):
    """Concatenate ftpc data across files. Returns (X, ps_target, mom, count)."""
    files = resolve_dch_files(config, root, split=split, eval_set=eval_set)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError("No DCH files (split=%s eval_set=%s) under %s"
                                % (split, eval_set, root))
    mom_window = None
    if eval_set:
        ev = config["eval"]
        mom_window = (float(ev["momentum"]), float(ev.get("mom_tol", 0.25)))
    xs, ps, ms, cs = [], [], [], []
    for f in files:
        X, p, m, c = load_dch_peakcount_one(config, f, mom_window=mom_window)
        if len(X) == 0:
            continue
        xs.append(X); ps.append(p); ms.append(m); cs.append(c)
    if not xs:
        raise ValueError("load_dch_peakcount_concat: no events survived "
                         "(split=%s eval_set=%s) -- check the momentum window / globs"
                         % (split, eval_set))
    X = np.concatenate(xs, 0); P = np.concatenate(ps, 0)
    M = np.concatenate(ms, 0); C = np.concatenate(cs, 0)
    if max_events:
        X, P, M, C = X[:max_events], P[:max_events], M[:max_events], C[:max_events]
    return X, P, M, C


