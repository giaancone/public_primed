"""Cache a teacher's predictions and pooled embeddings over the training pool.

Writes one .npz so `train_distill.py --teacher-preds` can distill from a teacher without
loading it at student-training time. The teacher's (GPU) inference runs once here; every
student seed and fraction afterwards trains cheaply against the cache.

Two signals are stored:

  preds  (N, n_targets)   the teacher's output. For a strong regressor this is close to the
                          label itself, so on its own it carries little beyond the labels.
  embeds (N, embed_dim)   the pooled representation before the head. This is what feature
                          distillation matches, and it is the signal that can exceed the
                          labels: the teacher sees the full-rate waveform while the student
                          sees a downsampled view, so the embedding encodes structure the
                          student's own input has already discarded.

  --scheme ft   rebuild the fine-tuned model and load its checkpoint; predictions run on raw
                waveforms, since instance normalization is internal to the backbone.
  --scheme st   load a small model saved by the from-scratch trainer; predictions use the
                same per-waveform normalization it was trained with.
  --stub        swap in the local stub backbone so the whole path runs with no GPU.

The cache is ROW-ALIGNED, not keyed. It is written in the order
`data_loader.load_concat(...)` returns with the same config, root and --max-files, and
`train_distill` indexes it positionally. Call both with the same arguments; a length guard
there catches a mismatch, but a same-length reordering would not be caught.

Pure ASCII.
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data_loader                 # noqa: E402
from src import student as st           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scheme", required=True, choices=["st", "ft", "ftpc"])
    ap.add_argument("--teacher", default=None, help="teacher weights (.pt / ckpt). Comma-"
                    "separated = ENSEMBLE (E2): preds + embeds AVERAGED over the seeds.")
    ap.add_argument("--regions", type=int, default=None,
                    help="cache the teacher embedding pooled over K TRACE-REGIONS (E1): "
                         "embeds become (N, K, D) instead of (N, D). One K=8 cache serves "
                         "smaller K (train_distill --regions re-pools). ft/ftpc only.")
    ap.add_argument("--stub", action="store_true", help="ft/ftpc: use the local StubBackbone")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=None,
                    help="ft inference batch (default 64; use ~16 at the 6272 full-wave "
                         "context to avoid OOM)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=0,
                    help="predict in chunks of N events, writing a resumable shard after each. "
                         "0 = one pass (old behavior). A killed job then loses at most one "
                         "chunk instead of the whole cache.")
    ap.add_argument("--out", required=True, help="output .npz of soft targets")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset = config.get("dataset", "dch")

    print("[cache] loading raw train waveforms ...")
    X, y, _ = data_loader.load_concat(config, args.root, split="train",
                                      max_files=args.max_files, max_events=args.max_events)
    print("[cache] dataset=%s  X=%s y=%s  scheme=%s" % (dataset, X.shape, y.shape, args.scheme))

    # ensemble (E2): comma-separated --teacher -> average preds + embeds over the seeds.
    teachers = args.teacher.split(",") if args.teacher else [None]
    if args.regions and args.scheme == "st":
        raise SystemExit("[cache] --regions needs a backbone (ft/ftpc), not st")
    # Chunked + resumable. One pass over 500k DCH events is hours; an interactive
    # allocation that dies takes all of it. With --chunk, each block of events is predicted and
    # written to its own shard, so a restart re-does at most one chunk.
    #
    # The model is rebuilt per chunk (~30-60 s: a 1.1 GB checkpoint plus the TimesFM backbone).
    # That is deliberate -- it keeps _one_teacher untouched, and 5 rebuilds cost ~5 minutes
    # against hours of re-prediction. Correctness beats the saving.
    #
    # Shards are keyed to (out, chunk, n). A shard written with a different --chunk or a
    # different event count would silently misalign rows, so the filename carries both and a
    # mismatch simply does not match.
    def _shard(tk_i, lo, hi):
        return "%s.part%d_%d-%d_n%d.npz" % (args.out, tk_i, lo, hi, len(X))

    preds_sum = embeds_sum = None
    model = None
    step = int(args.chunk) if args.chunk and args.chunk > 0 else len(X)
    bounds = [(lo, min(lo + step, len(X))) for lo in range(0, len(X), step)]
    for ti, tk in enumerate(teachers):
        pp, ee = [], []
        for lo, hi in bounds:
            sh = _shard(ti, lo, hi)
            if os.path.exists(sh):
                d = np.load(sh)
                pp.append(d["p"]); ee.append(d["e"])
                print("[cache] RESUME shard %d-%d from %s" % (lo, hi, os.path.basename(sh)),
                      flush=True)
                continue
            if len(bounds) > 1:
                _phase("chunk %d-%d of %d ..." % (lo, hi, len(X)))
            p_, e_, model = _one_teacher(args.scheme, config, tk, X[lo:hi], y[lo:hi],
                                         args.device, args.stub, args.batch, args.regions)
            if len(bounds) > 1:
                # Write to a temp name and rename: a rename is atomic, so a job killed
                # mid-write can never leave a shard that looks complete.
                # The temp name must end in .npz. np.savez appends ".npz" when the
                # path lacks it, so savez("x.npz.tmp") writes "x.npz.tmp.npz" and the rename
                # below then fails on a file that was never created. Caught by
                # verified by the chunked-resume path.
                tmp = sh[:-4] + ".tmp.npz"
                np.savez(tmp, p=p_, e=e_)
                os.replace(tmp, sh)            # atomic: no half-written shard can look complete
            pp.append(p_); ee.append(e_)
        p = np.concatenate(pp, axis=0) if len(pp) > 1 else pp[0]
        e = np.concatenate(ee, axis=0) if len(ee) > 1 else ee[0]
        if model is None:                       # every chunk resumed -> rebuild for param count
            _p, _e, model = _one_teacher(args.scheme, config, tk, X[:2], y[:2],
                                         args.device, args.stub, args.batch, args.regions)
        preds_sum = p if preds_sum is None else preds_sum + p
        embeds_sum = e if embeds_sum is None else embeds_sum + e
    preds = (preds_sum / len(teachers)).astype(np.float32)
    embeds = (embeds_sum / len(teachers)).astype(np.float32)
    if len(teachers) > 1:
        print("[cache] ENSEMBLE: averaged %d teachers" % len(teachers))
    if preds.shape != (len(X), y.shape[1]):
        raise SystemExit("[cache] preds shape %s != (%d,%d)"
                         % (preds.shape, len(X), y.shape[1]))
    if embeds.shape[0] != len(X):
        raise SystemExit("[cache] embeds rows %d != %d" % (embeds.shape[0], len(X)))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tparams = sum(p.numel() for p in model.parameters())
    # Save the targets too: a CROSS-RESOLUTION distill caches the teacher on a
    # different config (full 6272) than the student trains on (downsampled 640). The
    # events/order are identical (same glob), and y is divisor-normalized identically,
    # so train_distill can assert its loaded y == these targets row-for-row and catch
    # any misalignment loudly instead of silently pairing mismatched events.
    np.savez(args.out, preds=preds, embeds=embeds, embed_dim=embeds.shape[-1],
             regions=(embeds.shape[1] if embeds.ndim == 3 else 0),
             targets=y.astype(np.float32), n=len(X), n_targets=y.shape[1],
             dataset=dataset, scheme=args.scheme, teacher_params=tparams)
    print("[cache] wrote %s  preds=%s embeds=%s targets=%s (aligned to %d train rows)"
          % (args.out, preds.shape, embeds.shape, y.shape, len(X)))
    # Only after the real cache exists -- otherwise a crash here would delete the resume points.
    for ti in range(len(teachers)):
        for lo, hi in bounds:
            sh = _shard(ti, lo, hi)
            if os.path.exists(sh):
                os.remove(sh)


def _phase(msg, t0=None):
    """Timestamped phase marker. This script has no inner progress bar -- predict_counts and
    predict_embed live in src/ and are shared, so instrumenting them is not worth the risk. Two
    markers per teacher is enough to tell a slow run from a hung one and to extrapolate the
    finish: a 500k-event DCH cache is two passes of comparable cost."""
    import time
    now = time.time()
    print("[cache] %s%s" % (msg, "" if t0 is None else "  (%.1f min)" % ((now - t0) / 60.0)),
          flush=True)
    return now


def _one_teacher(scheme, config, teacher, X, y, device, stub, batch, regions):
    """Build one teacher and return (preds (N,n_tgt), embeds (N,D) or (N,K,D), model).
    embeds are the pooled global embedding, or K trace-region embeddings if regions>0."""
    if scheme == "st":
        model, ck = st.load_model(teacher, device=device)
        if ck["length"] != X.shape[1] or ck["n_targets"] != y.shape[1]:
            raise SystemExit("[cache] teacher/data shape mismatch")
        preds = st.predict(model, X, device=device)
        embeds = _penultimate(model, st._standardize(X), device)
    elif scheme == "ft":
        import torch as _th
        from src import teacher as tch
        fcfg = config["ft"]
        backbone = tch._build_backbone(fcfg, X.shape[1], device, stub)
        model = tch.FineTuneModel(backbone, fcfg.get("head_hidden", [128]), y.shape[1]).to(device)
        if teacher:
            import torch
            ck = torch.load(teacher, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print("[cache] loaded ft weights from %s" % teacher)
        elif not stub:
            raise SystemExit("[cache] --scheme ft needs --teacher (or --stub)")
        _t = _phase("pass 1/2: predictions over %d events (batch %s) ..." % (len(X), batch))
        # Eval() + no_grad at the call site. teacher._predict_batched (:544) does
        # neither, unlike teacher_dch._predict_raw (:285-290) which does both, and _one_teacher
        # never called .eval() -- so a freshly built model predicted in train mode while building
        # an autograd graph it immediately threw away. There is no Dropout or BatchNorm in
        # FineTuneModel (Linear + gelu only), so the arithmetic was never wrong; the cost was
        # memory and speed. Fixed here rather than in _predict_batched because that helper has
        # other callers whose behavior must not move.
        model.eval()
        with _th.no_grad():
            preds = tch.predict(model, X, device=device, batch=batch)
        _t = _phase("pass 1/2 done; pass 2/2: embeddings ...", _t)
        with _th.no_grad():
            embeds = (_regional_embed(model, X, regions, device, batch or 64) if regions
                      else tch.predict_embed(model, X, device=device, batch=batch))
        _phase("pass 2/2 done", _t)
    else:                                                          # ftpc
        from src import teacher_dch as tdch
        fcfg = config["ftpc"]
        if not teacher and not stub:
            raise SystemExit("[cache] --scheme ftpc needs --teacher (the ftpc ckpt)")
        if teacher:
            model, _ = tdch.load_peakcount(teacher, fcfg, device=device, stub=stub)
            print("[cache] loaded ftpc weights from %s" % teacher)
        else:
            L = X.shape[1]
            bb = tdch._build_backbone(fcfg, L, device, True)
            model = tdch.PeakCountModel(bb, L, fcfg.get("head_hidden", [128]),
                                       n_classes=fcfg.get("n_classes", 3)).to(device)
        _t = _phase("pass 1/2: predictions over %d events (batch %s) ..." % (len(X), batch))
        preds = tdch.predict_counts(model, X, device=device, batch=batch).reshape(-1, 1)
        _t = _phase("pass 1/2 done; pass 2/2: embeddings ...", _t)
        embeds = (_regional_embed(model, X, regions, device, batch or 64) if regions
                  else tdch.predict_embed(model, X, device=device, batch=batch))
        _phase("pass 2/2 done", _t)
    return np.asarray(preds, np.float32), np.asarray(embeds, np.float32), model


def _regional_embed(model, X, K, device, bs):
    """Teacher token embeddings pooled over K trace-regions -> (N, K, D). Uses the
    backbone token sequence (B, T, D) directly (ft/ftpc both expose .backbone)."""
    import torch
    import torch.nn.functional as F
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)
            tok = model.backbone(xb)                              # (B, T, D)
            r = F.adaptive_avg_pool1d(tok.transpose(1, 2), int(K)).transpose(1, 2)  # (B,K,D)
            outs.append(r.cpu().numpy())
    return np.concatenate(outs, 0)


def _penultimate(model, Xs, device, bs=1024):
    """Capture the input to the model's last nn.Linear (its penultimate features)
    over already-standardized Xs -> (N, feat_dim). Teacher-agnostic via a hook."""
    import torch
    import torch.nn as nn
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise SystemExit("[cache] teacher has no Linear layer to hook for embeds")
    cap = {}
    h = last.register_forward_hook(lambda mod, inp, out: cap.__setitem__("f", inp[0].detach()))
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xs), bs):
            xb = torch.as_tensor(Xs[i:i + bs], dtype=torch.float32, device=device)
            model(xb)
            outs.append(cap["f"].cpu().numpy())
    h.remove()
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    main()
