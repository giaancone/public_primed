"""Student training: the from-scratch baseline and the distilled arm.

Both arms of the comparison live here. `train_fs` trains a small network on raw waveforms with
no teacher at all; `train_student` adds the teacher terms. Holding everything else identical
between them -- architecture, input pipeline, labeled events, optimizer and step budget -- is
what makes the difference between the two attributable to the teacher.

No foundation model and no embeddings are involved: the student consumes the downsampled,
scaled waveform directly.

Pure ASCII.
"""

import numpy as np
import torch
import torch.nn as nn

from src.metrics import regression_metrics


def _standardize(X):
    """Per-waveform z-score (RevIN-like) -- each event normalized by its own
    mean/std so the cnn is not at the mercy of absolute amplitude.

    This makes the input exactly invariant to PER-EVENT energy scale. Since c and s
    are photoelectron
    counts, i.e. energy-like, a model fed this cannot recover their magnitude -- only
    their shape. That is invisible to the C/S ratio (a common factor cancels), which is
    why the ratio can look excellent while c and s do not. It also annihilates any
    upstream additive or affine preprocessing such as baseline subtraction or a global
    min-max map.

    The published edge model does not do this. It fits one global min-max scaler on the
    training set and applies the same affine map to every event, so per-event amplitude
    survives. Use `input_norm="global"` for that behavior."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def fit_global_minmax(X, hi_pct=None):
    """one (min, max) over all events and samples of the train split.

    `hi_pct` (default None = use the exact maximum) takes `hi` as that percentile of the train
    samples instead. A handful of extreme samples can otherwise compress every ordinary
    waveform into a small part of [0, 1]. It remains one global affine map, so per-event
    amplitude survives and the hardware cost is unchanged; outliers simply land above 1."""
    X = np.asarray(X, dtype=np.float32)
    hi = float(X.max()) if hi_pct is None else float(np.percentile(X, float(hi_pct)))
    return {"lo": float(X.min()), "hi": hi}


def apply_global_minmax(X, sc):
    """Apply a train-fitted global scaler. preserves per-event amplitude (affine)."""
    X = np.asarray(X, dtype=np.float32)
    rng_ = (sc["hi"] - sc["lo"]) or 1.0
    return ((X - sc["lo"]) / rng_).astype(np.float32)


def _prep_inputs(X, cfg=None, scaler=None):
    """Input normalization dispatch.

    cfg["input_norm"]:
      "per_event" (default) -> _standardize; unchanged legacy behavior.
      "global"              -> one global min-max map. Requires a scaler fitted on
                               train and passed in for val/test (never refit downstream).
      "none"                -> pass through (input already prepared upstream).
    """
    mode = str((cfg or {}).get("input_norm", "per_event")).lower()
    if mode == "global":
        if scaler is None:
            raise ValueError("input_norm='global' needs a TRAIN-fitted scaler "
                             "(fit_global_minmax on the train split) -- refusing to "
                             "refit here, which would leak the eval range")
        return apply_global_minmax(X, scaler)
    if mode == "per_event_amp":
        if scaler is None:
            raise ValueError("input_norm='per_event_amp' needs a TRAIN-fitted amplitude "
                             "scaler (fit_amp_stats on the train split) -- refusing to "
                             "refit here, which would leak the eval statistics")
        return np.concatenate([_standardize(X), apply_amp_feats(X, scaler)],
                              axis=1).astype(np.float32)
    if mode == "none":
        return np.asarray(X, dtype=np.float32)
    return _standardize(X)


# ---------------------------------------------------------------------------------------
# PER-EVENT normalization *plus* The amplitude it throws away.
#
# The two existing options each give up something. "per_event" z-scores every waveform, so
# every input is well-conditioned but the per-event amplitude is destroyed exactly -- and c
# and s are photoelectron counts, i.e. energy-like, so their magnitude becomes unrecoverable.
# "global" keeps amplitude but leaves small-amplitude events
# poorly conditioned, since one affine map has to serve the whole dynamic range.
#
# This mode takes both: z-score the shape, and hand the discarded scale back to the network
# as explicit features (mean, std, max, integral of the raw trace), each standardized by
# TRAIN-fitted statistics. The network then sees a well-conditioned shape and the magnitude,
# instead of trading one for the other.
#
# Motivation: on the 10k student c and s trade against each other under reweighting, which
# says capacity is being reallocated, not added. This adds information, so it is not
# zero-sum -- and s is the more energy-dependent component, so it should help s specifically.
#
# NOTE: appends 4 columns, so the input width grows by 4. Intended for the dense ("fc")
# student, where the input is a flat vector and extra features are natural. For the conv
# student these would be read as 4 extra time steps, which is not what they mean -- injecting
# them there needs a concat after the conv trunk instead, which is not implemented.
# ---------------------------------------------------------------------------------------
def _amp_feats(X):
    """Raw per-event scale descriptors, in the order used everywhere: mean/std/max/sum."""
    X = np.asarray(X, dtype=np.float32)
    return np.stack([X.mean(axis=1), X.std(axis=1), X.max(axis=1), X.sum(axis=1)],
                    axis=1).astype(np.float32)


def fit_amp_stats(X):
    """Fit the amplitude-feature normalizer on the train split only."""
    F = _amp_feats(X)
    return {"mu": [float(v) for v in F.mean(axis=0)],
            "sd": [float(v) for v in (F.std(axis=0) + 1e-8)]}


def apply_amp_feats(X, sc):
    """Standardize the amplitude features with train statistics (never refit)."""
    F = _amp_feats(X)
    mu = np.asarray(sc["mu"], dtype=np.float32)
    sd = np.asarray(sc["sd"], dtype=np.float32)
    return ((F - mu) / sd).astype(np.float32)


class MLPFlat(nn.Module):
    """Plain fully-connected network over the downsampled waveform.

    flatten -> fc(hidden...) -> n_targets, ReLU between. No convolution and no pooling. It
    ends in a Linear so the feature-distillation hook has a layer to tap.

    Parameter count is dominated by `length * hidden[0]`, so the input must be downsampled
    before it reaches this: a full-rate waveform would multiply the model size by the
    decimation factor."""

    def __init__(self, length, hidden=(16, 24, 8), n_targets=1):
        super().__init__()
        dims = [int(length)] + [int(h) for h in hidden]
        layers = []
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):            # x: (B, L)
        return self.net(x)           # -> (B, n_targets)



def build_model(cfg, length, n_targets):
    """Build the student. Only the plain fully-connected model is supported.

    This release ships the one architecture the paper reports: a flat mlp over the
    downsampled waveform, matching the published edge model. `arch` is still read from
    the config so an unrecognised value fails loudly rather than silently substituting
    a different network.
    """
    arch = str(cfg.get("arch", "fc")).lower()
    if arch != "fc":
        raise ValueError("unknown arch %r -- this release ships 'fc' only" % arch)
    return MLPFlat(length, tuple(cfg.get("hidden", [16, 24, 8])), n_targets)


def train_fs(X, y, cfg, seed=0, fraction=100.0, device="cpu", verbose=False):
    """Train the cnn on `fraction`% of raw waveforms. Returns (model, metrics)."""
    rng = np.random.default_rng(seed)
    # This path cannot honor input_norm. It normalizes before the train/val split, so a
    # train-fitted global scaler is not constructible here without reordering the function. It
    # used to call _standardize() unconditionally and silently ignore cfg["input_norm"], which
    # meant every `fs` / `convmlp` / `st` / `st_big` / `cnn` DRO curve was per-event z-scored --
    # i.e. scale-blind -- while `fc_little` correctly used global MinMax. The repo's own
    # measurement of that gap is c ~4.5 / s ~4.6 z-scored against 1.53 / 2.46 global: a 2-3x
    # handicap on c and s (the C/S ratio is unaffected, a common per-event factor cancels).
    # Failing loudly is the fix for the silence. Actually supporting global norm here needs the
    # fitting a train-only scaler needs the split to happen first, which changes behavior
    # for existing callers -- so this path raises instead of silently normalizing differently.
    _mode = str((cfg or {}).get("input_norm", "per_event")).lower()
    if _mode not in ("per_event", "none"):
        raise ValueError(
            "train_fs cannot honor input_norm=%r: it normalizes before the train/val split, so "
            "a TRAIN-fitted scaler cannot be built here, and silently falling back to a per-event "
            "z-score would make the model scale-blind without saying so. Use the distillation "
            "path (train_student), which does support it, or move the split above the "
            "normalization in train_fs first." % _mode)
    Xs = _standardize(X)
    n = len(Xs)
    perm = rng.permutation(n)
    n_use = max(int(round(n * fraction / 100.0)), 32)
    idx = perm[:n_use]

    n_val = max(int(round(len(idx) * cfg.get("val_fraction", 0.1))), 8)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.as_tensor(Xs[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=device)
    yva_np = y[val_idx]

    torch.manual_seed(seed)
    model = build_model(cfg, Xs.shape[1], y.shape[1]).to(device)
    # No-parameter tuning knobs (default off -> fs/st baselines byte-identical):
    #   weight_decay>0 -> AdamW; lr_schedule='cosine' -> cosine LR decay;
    #   augment>0 -> Gaussian input noise (std, in z-scored units) at train time.
    wd = float(cfg.get("weight_decay", 0.0))
    lr = float(cfg.get("lr", 1e-3))
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
           if wd > 0 else torch.optim.Adam(model.parameters(), lr=lr))
    tw = cfg.get("target_weights")
    if tw is None:
        loss_fn = nn.MSELoss()
    else:                                 # per-target weighted MSE (DRO [c,s,t0])
        _w = torch.as_tensor(tw, dtype=torch.float32, device=device).reshape(1, -1)
        loss_fn = lambda pred, true: (_w * (pred - true) ** 2).mean()  # noqa: E731
    bs, epochs = cfg.get("batch_size", 256), cfg.get("epochs", 40)
    augment = float(cfg.get("augment", 0.0))
    n_batches = max((len(Xtr) + bs - 1) // bs, 1)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_batches)
             if str(cfg.get("lr_schedule", "")).lower() == "cosine" else None)

    model.train()
    for ep in range(epochs):
        order = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = order[i:i + bs]
            xb = Xtr[b]
            if augment > 0:               # additive Gaussian noise (train-time only)
                xb = xb + augment * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr[b])
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("  fs epoch %d loss %.4f lr %.2e" % (ep, float(loss), opt.param_groups[0]["lr"]))

    model.eval()
    with torch.no_grad():
        pred = model(Xva).cpu().numpy()
    return model, regression_metrics(pred, yva_np)


def predict(model, X, device="cpu", batch=4096, cfg=None, scaler=None):
    """Batched inference. Chunked so a large eval set (e.g. 100k pion/kaon events)
    never materializes one giant activation tensor -- a wide model (st_big, 96ch)
    would OOM on an unbatched 100k forward (~18 GB per conv activation)."""
    # Eval must use the same transform as training. A model trained with a global
    # scaler that is then predicted with per-event z-scoring gives meaningless
    # numbers, silently -- so the scaler travels on the model.
    Xs = _prep_inputs(X, cfg if cfg is not None else getattr(model, "_x_cfg", None),
                      scaler if scaler is not None else getattr(model, "_x_scaler", None))
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            xb = torch.as_tensor(Xs[i:i + batch], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    if not outs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
# Weight persistence + knowledge distillation.                                #
#                                                                             #
# save_model() persists a trained net (state_dict + the arch cfg + input      #
# length + n_targets) so it reloads with no guesswork via load_model(). That  #
# is how a trained teacher is captured once rather than re-paying the GPU     #
# cost on every run. train_student() then distills it into a small student:   #
# loss = alpha * (student vs true label) + (1 - alpha) * (student vs the      #
# teacher's prediction), scored with the same metric as the control arm.      #
# --------------------------------------------------------------------------- #
def save_model(path, model, arch_cfg, length, n_targets, meta=None):
    """Persist a from-scratch model so load_model() can rebuild it exactly.
    arch_cfg is the config block used to build it (must carry 'arch' + dims)."""
    torch.save({"state_dict": model.state_dict(),
                "arch_cfg": {k: arch_cfg[k] for k in arch_cfg},
                "length": int(length),
                "n_targets": int(n_targets),
                "meta": dict(meta or {})}, path)


def load_model(path, device="cpu"):
    """Rebuild + load a model saved by save_model(). Returns (model, ckpt-dict).
    The model is in eval() mode and on `device`."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch_cfg"], ck["length"], ck["n_targets"]).to(device)
    sd = ck["state_dict"]
    # Back-compat: SpectralTCN teachers saved before the head Dropout module was
    # inserted have the final Linear at head.2; the current head puts a (param-free)
    # Dropout at head.2 and the Linear at head.3. Same math -- remap the legacy keys
    # so old teachers still load. Guarded to fire only for that exact shift (never
    # for CNN1D students, whose head Linear legitimately lives at head.2).
    msd = model.state_dict()
    if ("head.3.weight" in msd and "head.2.weight" not in msd
            and "head.2.weight" in sd and "head.3.weight" not in sd):
        for k in ("weight", "bias"):
            sd["head.3.%s" % k] = sd.pop("head.2.%s" % k)
    model.load_state_dict(sd)
    model.eval()
    return model, ck


def _teacher_predict(teacher, Xtr, chunk=4096):
    """Teacher soft targets on already-standardized, on-device inputs, in
    chunks so a 400k-event tensor never OOMs the GPU. No grad through teacher."""
    teacher.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xtr), chunk):
            outs.append(teacher(Xtr[i:i + chunk]))
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- #
# sparsification -- train a slightly larger model, then magnitude-prune the      #
# dense weights down to a few-thousand-EFFECTIVE-param edge model (the literature #
# recipe: harvest slack rather than train tiny-and-dense). L1 during training     #
# concentrates the signal into fewer weights so more can be pruned near-losslessly.#
# --------------------------------------------------------------------------- #
_PRUNABLE = (nn.Conv1d, nn.Linear)


def prunable_weight_modules(model):
    """(name, module) for every Conv1d/Linear -- the tensors we prune/regularize
    (weights only; biases + norms stay dense, they are a negligible param count)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _PRUNABLE)]


def l1_penalty(model):
    """Sum |w| over prunable weights -- add lambda*this to the loss to induce
    sparsity (drives small weights toward 0 so pruning removes more, cheaper)."""
    return sum(m.weight.abs().sum() for _, m in prunable_weight_modules(model))


def global_magnitude_masks(model, sparsity):
    """Binary masks that zero the globally-smallest `sparsity` fraction of prunable
    weights (0<=sparsity<1). One shared magnitude threshold across all Conv/Linear
    weights, so capacity flows to whichever layer needs it (the head, here)."""
    sparsity = float(sparsity)
    allw = torch.cat([m.weight.detach().abs().reshape(-1)
                      for _, m in prunable_weight_modules(model)])
    k = int(sparsity * allw.numel())
    thr = torch.kthvalue(allw, k).values.item() if k > 0 else -1.0
    return {n: (m.weight.detach().abs() > thr).to(m.weight.dtype)
            for n, m in prunable_weight_modules(model)}


def apply_masks(model, masks):
    """Zero the pruned weights in-place (call after every optimizer step during
    fine-tuning so pruned weights stay at 0)."""
    with torch.no_grad():
        for n, m in prunable_weight_modules(model):
            if n in masks:
                m.weight.mul_(masks[n])


def effective_params(model, masks=None):
    """(nonzero, dense_total). nonzero = surviving prunable weights (by `masks`, or
    by current zeros) + all non-prunable params (biases/norms). This is the number
    that matters for an edge deploy, not the dense total."""
    dense_total = sum(p.numel() for p in model.parameters())
    prunable_total, nz = 0, 0
    for n, m in prunable_weight_modules(model):
        prunable_total += m.weight.numel()
        if masks is not None and n in masks:
            nz += int(masks[n].sum().item())
        else:
            nz += int((m.weight != 0).sum().item())
    other = dense_total - prunable_total          # biases + norm params (stay dense)
    return nz + other, dense_total


def finetune_masked(model, X, y, masks, cfg, seed=0, fraction=100.0, device="cpu",
                    epochs=None, teacher=None, teacher_embed=None, alpha=1.0,
                    feat_weight=0.0, feat_warmup=0.3):
    """Fine-tune `model` with `masks` RE-ENFORCED after every optimizer step (pruned
    weights stay 0) while the survivors recover. Uses the same objective as
    train_student -- alpha*label + (1-alpha)*teacher-output + feat_weight*feature --
    so a distilled model recovers under distillation and does not drift back toward the
    labels-only solution. teacher/teacher_embed are arrays aligned row-for-row to X.
    Labels-only (teacher=None, feat_weight=0) is the default -> unchanged behavior."""
    apply_masks(model, masks)
    rng = np.random.default_rng(seed)
    # train/recover mismatch guard. This also called _standardize() unconditionally. The
    # prune-recovery path trains the dense model with
    # st.train_student -- which does honor input_norm: global -- and then recovered it here
    # with per-event z-scored inputs. The recovery step was optimizing against an input the
    # deployed model never sees. The model carries its own transform (model._x_cfg /
    # model._x_scaler, stamped by train_student), so prefer that when present and refuse to
    # guess otherwise.
    _mcfg = getattr(model, "_x_cfg", None)
    _mscaler = getattr(model, "_x_scaler", None)
    if _mcfg is not None:
        Xs = _prep_inputs(X, _mcfg, _mscaler)      # match however the model was trained
    else:
        _mode = str((cfg or {}).get("input_norm", "per_event")).lower()
        if _mode not in ("per_event", "none"):
            raise ValueError(
                "finetune_masked was asked for input_norm=%r but the model carries no stamped "
                "transform (_x_cfg/_x_scaler), so the training-time scaler cannot be recovered. "
                "Re-train through train_student (which stamps it) or pass a per_event model."
                % _mode)
        Xs = _standardize(X)
    perm = rng.permutation(len(Xs))
    n_use = max(int(round(len(Xs) * fraction / 100.0)), 32)
    idx = perm[:n_use]
    Xtr = torch.as_tensor(Xs[idx], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
    a = float(alpha)
    t_tr = (torch.as_tensor(np.asarray(teacher)[idx], dtype=torch.float32, device=device)
            if teacher is not None else None)
    do_feat = teacher_embed is not None and float(feat_weight) > 0.0
    te_tr = (torch.as_tensor(np.asarray(teacher_embed)[idx], dtype=torch.float32)  # CPU
             if do_feat else None)

    projector, cap, hook = None, {}, None
    if do_feat:
        last = None
        for mdl in model.modules():
            if isinstance(mdl, nn.Linear):
                last = mdl
        hook = last.register_forward_hook(lambda mod, inp, out: cap.__setitem__("f", inp[0]))
        with torch.no_grad():
            model(Xtr[:2])
        projector = nn.Linear(cap["f"].shape[1], te_tr.shape[1]).to(device)
    params = list(model.parameters()) + (list(projector.parameters()) if projector else [])
    opt = torch.optim.Adam(params, lr=cfg.get("finetune_lr", cfg.get("lr", 1e-3)))
    tw = cfg.get("target_weights")
    if tw is None:
        wmse = lambda p, t: ((p - t) ** 2).mean()                    # noqa: E731
    else:
        _w = torch.as_tensor(tw, dtype=torch.float32, device=device).reshape(1, -1)
        wmse = lambda p, t: (_w * (p - t) ** 2).mean()               # noqa: E731
    ep = int(epochs if epochs is not None else cfg.get("finetune_epochs", 20))
    bs = cfg.get("batch_size", 256)
    fw = float(feat_weight)
    warm = max(1, int(float(feat_warmup) * ep * max(1, (len(Xtr) + bs - 1) // bs)))
    step = 0
    model.train()
    for _ in range(ep):
        order = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            pred = model(Xtr[b])
            loss = a * wmse(pred, ytr[b])
            if t_tr is not None:
                loss = loss + (1.0 - a) * wmse(pred, t_tr[b])
            if do_feat:
                ramp = min(1.0, step / warm)
                te_b = te_tr[b.cpu()].to(device)         # move this batch's embeds
                loss = loss + fw * ramp * ((_l2norm(projector(cap["f"]))
                                            - _l2norm(te_b)) ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            apply_masks(model, masks)         # keep pruned weights at 0
            step += 1
    if hook is not None:
        hook.remove()
    model.eval()
    return model


def gmp_sparsity(progress, target):
    """Gradual magnitude pruning sparsity schedule (cubic, the TF-Model-Optimization
    default): 0 -> target as progress goes 0 -> 1. Prunes fast early, slows near the end
    so the survivors have time to adapt. Used for SPARSE-AWARE training (train with the
    growing mask), which beats train-dense-then-prune at the same final sparsity."""
    p = min(1.0, max(0.0, float(progress)))
    return float(target) * (1.0 - (1.0 - p) ** 3)


def _l2norm(z):
    """L2 normalize over the last axis (scale-free feature-distill target). Works for
    (B, D) global embeds and (B, K, D) regional embeds alike."""
    return z / (z.norm(dim=-1, keepdim=True) + 1e-6)


def _rkd_dist(z):
    """Mean-normalized pairwise-distance matrix over a batch (RKD-D). z: (B, D)."""
    d = torch.cdist(z, z)
    return d / (d.mean() + 1e-6)


def _init_frontend_pca(model, Xtr_np):
    """Initialize the first Linear(L->K) of a dense full-waveform student with the
    top-K PCA directions of the (already-standardized) training waveforms -- data-driven
    'matched filters' so a tiny funnel starts from an informative global projection
    instead of random noise. No teacher forward needed. No-op unless the first Linear
    consumes the full trace (in_features == L), so it silently skips conv/other students."""
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not lins:
        return
    fe = lins[0]
    L = int(Xtr_np.shape[1])
    if fe.in_features != L:                     # not a full-waveform dense front-end -> skip
        return
    K = fe.out_features
    Xc = np.asarray(Xtr_np, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    m = min(len(Xc), 4096)                       # svd on a subsample for speed
    _, _, Vt = np.linalg.svd(Xc[:m], full_matrices=False)
    comps = Vt[:K]                               # (min(K, rank), L) unit-norm rows
    with torch.no_grad():
        w = torch.as_tensor(comps, dtype=fe.weight.dtype, device=fe.weight.device)
        if w.shape[0] < K:                       # pad with zeros if fewer components than K
            w = torch.cat([w, torch.zeros(K - w.shape[0], L, dtype=w.dtype,
                                          device=w.device)], dim=0)
        fe.weight.copy_(w)
        if fe.bias is not None:
            fe.bias.zero_()



def init_output_bias(model, cfg, y_train):
    """Opt-in (cfg out_bias_init: mean): start the head's bias at the train-target mean.

    A narrow funnel starts with a near-constant output while the target has a non-zero mean,
    so the first optimizer steps are spent closing that offset. Every bias moves at once by
    roughly the learning rate, which can saturate the narrow layer before it learns anything.
    Starting the head at the train-target mean removes the offset, so the first steps go into
    fitting structure instead.

    Default off, so every shipped
    number stays byte-identical until a config opts in.
    """
    if str(cfg.get("out_bias_init", "none")).lower() != "mean":
        return
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        lins[-1].bias.copy_(torch.as_tensor(np.asarray(y_train, np.float32).mean(axis=0),
                                            dtype=lins[-1].bias.dtype))

def train_student(X, y, teacher, cfg, seed=0, fraction=100.0, device="cpu",
                  alpha=0.5, verbose=False, teacher_embed=None, feat_weight=0.0,
                  feat_warmup=0.3, rkd_weight=0.0, plot_path=None, label_seed=None):
    """Distill `teacher` into a small student defined by cfg (arch/dims).
    `teacher` is either a model (predicted live) or a precomputed soft-target
    array aligned row-for-row to X -- the teacher-agnostic path, so an `ft`
    (200M TimesFM) teacher distills through cached predictions without ever
    loading it here. alpha in [0,1] balances the true-label loss vs matching the
    teacher's soft output (alpha=1 -> ignore teacher output; alpha=0 -> pure output
    mimicry). Mirrors train_fs()'s split/standardization so metrics are comparable.

    feature distillation: if
    `teacher_embed` (an (N, D) array of the teacher's pooled embedding, row-aligned
    to X) is given with feat_weight>0, an added loss term pulls a learned projection
    of the student's penultimate features toward the teacher's embedding (both
    L2-normalized -> scale-free), ramped over the first `feat_warmup` fraction of
    steps. This teaches the teacher's representation, not just its output. The four
    arms map to (alpha, feat_weight): labels=(1,0), output=(0.5,0), feature=(1,w),
    both=(0.5,w). feat_weight=0 -> byte-identical to the old output-only path."""
    # Which events are labeled vs how the net is initialized. `seed` normally drives
    # both. `label_seed` splits them so the labeled subset can be pinned to the TEACHER's draw.
    #
    # Why that matters. The teacher at fraction f chose its labels with default_rng(its seed)
    # (teacher.py:351-357, teacher_dch.py:143-150 -- identical code to this). With a
    # different student seed the two draws are disjoint, so a point labeled "1%" was informed by
    # the teacher's 1,000 events and the student's other 1,000: 2% of the truth behind a 1% claim.
    # label_seed=0 makes the student reuse the teacher's exact events, so the chain consumes f%
    # and nothing more.
    #
    # cost: every seed then shares one labeled subset, so the error bar measures initialization
    # variance, not data-subset variance. both arms must use the same label_seed or the paired
    # comparison is between students trained on different data.
    rng = np.random.default_rng(seed if label_seed is None else label_seed)
    n = len(X)
    if isinstance(teacher, np.ndarray) and len(teacher) != n:
        raise ValueError("teacher preds (%d) not aligned to X (%d)" % (len(teacher), n))
    perm = rng.permutation(n)
    n_use = max(int(round(n * fraction / 100.0)), 32)
    idx = perm[:n_use]

    n_val = max(int(round(len(idx) * cfg.get("val_fraction", 0.1))), 8)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # global MinMax: Fit on the train subset, then normalize everything with it.
    #
    # The split must happen above this line so a train-only fit is possible at all. Per-event
    # z-scoring is exactly scale-invariant, so it destroys the per-event amplitude the
    # dual-readout targets are computed from: a model fed that input can recover the ratio of
    # the components but not their magnitudes. A separation-power metric barely notices,
    # because it is itself a ratio and the scale cancels.
    #
    # Fitting on tr_idx only matters: fitting on all of X would leak the val/holdout range
    # into the normalization. Nothing between here and the fit touches Xs.
    if str(cfg.get("input_norm", "per_event")).lower() == "global" \
            and cfg.get("_x_scaler") is None:
        cfg = dict(cfg)
        cfg["_x_scaler"] = fit_global_minmax(X[tr_idx])
    Xs = _prep_inputs(X, cfg, cfg.get("_x_scaler"))

    Xtr = torch.as_tensor(Xs[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=device)
    yva_np = y[val_idx]

    # teacher's soft targets aligned to tr_idx: index a precomputed preds array,
    # or run the teacher model live on the same standardized train inputs.
    if isinstance(teacher, np.ndarray):
        t_tr = torch.as_tensor(teacher[tr_idx], dtype=torch.float32, device=device)
    else:
        t_tr = _teacher_predict(teacher, Xtr)

    # FEATURE-distill target: the teacher embedding aligned to tr_idx. global if it is
    # (N, D); MULTI-REGION (E1) if (N, K, D) -> matched to the student's region_features.
    do_feat = teacher_embed is not None and float(feat_weight) > 0.0
    te = np.asarray(teacher_embed) if do_feat else None
    regional = do_feat and te.ndim == 3
    # Keep the (large) teacher embeddings on CPU and move each batch to the GPU in the
    # loop -- DCH regional at 500k events is ~18 GB, which would OOM if placed on-device.
    te_tr_np = te[tr_idx] if do_feat else None       # fancy-index -> already a copy

    # ------------------------------------------------------------------ feat_center
    # Why centering is needed.
    #
    # The feature loss is a squared distance between L2-normalized vectors. That is only a
    # useful training signal if the teacher's embeddings actually differ between events.
    # measured on the DCH teacher cache (120k events, 1280-d):
    #
    #     DCH   mean pairwise cosine 0.9950   min 0.9176   per-dim sd/|mean| 0.30
    #     DRO   mean pairwise cosine 0.8641   min -0.4517  per-dim sd/|mean| 1.86
    #
    # DCH embeddings are 99.5% identical across events. So "match the teacher's embedding"
    # is very nearly "output this one constant vector": the projector learns it immediately and
    # then nothing further is learned -- the feature loss stays flat from the first epoch to the
    # last. Raising feat_weight in that state only spends more of the loss on reproducing a
    # constant, so the term is not merely useless but actively costly.
    #
    # The information is not missing from the embedding -- the teacher's own head reads the
    # same pooled vector. It lives in the small component that varies event to event, and a
    # uniform squared distance over ~1000 dimensions cannot see it past the large component
    # that does not. Centering makes the target the deviation from typical rather than the
    # vector itself, leaving only the varying component to match.
    #   "mean"   subtract the per-dimension mean
    #   "zscore" subtract the mean and divide by the per-dimension sd (default eps guards
    #            dead dimensions, which are common in a pooled transformer embedding)
    # Statistics come from tr_idx only -- the same train-only rule the input scaler follows.
    # Default off: enabling it changes every feature-distilled run, so it must be an arm.
    _fc = str(cfg.get("feat_center", "none")).lower()
    if do_feat and _fc in ("mean", "zscore", "std"):
        _mu = te_tr_np.mean(axis=0, keepdims=True)
        te_tr_np = te_tr_np - _mu
        if _fc in ("zscore", "std"):
            _sd = te_tr_np.std(axis=0, keepdims=True)
            te_tr_np = te_tr_np / np.maximum(_sd, 1e-6)
        if verbose:
            _u = te_tr_np.reshape(len(te_tr_np), -1)
            _u = _u / np.maximum(np.linalg.norm(_u, axis=1, keepdims=True), 1e-12)
            _n = min(len(_u) // 2, 1000)
            if _n > 1:
                print("  feat_center=%s -> pairwise cosine now %.4f (was ~0.995 on DCH)"
                      % (_fc, float((_u[:_n] @ _u[_n:2 * _n].T).mean())))

    te_tr = (torch.as_tensor(te_tr_np, dtype=torch.float32) if do_feat else None)

    torch.manual_seed(seed)
    student = build_model(cfg, Xs.shape[1], y.shape[1]).to(device)
    if str(cfg.get("frontend_init", "none")).lower() == "pca":
        _init_frontend_pca(student, Xs[tr_idx])   # data-driven matched-filter init

    # Projector maps the student's features to the teacher's embed dim (trains with the
    # student). global: hook the input to the last Linear (penultimate). regional: use
    # the student's region_features (needs ConvMLP), pooled to K regions.
    projector, cap, hook, K = None, {}, None, None
    if do_feat and regional:
        if not hasattr(student, "region_features"):
            raise ValueError("regional feature distill needs a student with region_features, which this release does not ship")
        K = int(te_tr.shape[1])
        with torch.no_grad():
            rf = student.region_features(Xtr[:2], K)          # (2, K, C)
        projector = nn.Linear(rf.shape[2], te_tr.shape[2]).to(device)
    elif do_feat:
        lins = [m for m in student.modules() if isinstance(m, nn.Linear)]
        if not lins:
            raise ValueError("feature distill: student has no Linear to hook")
        # feat_tap picks where the feature-distill signal enters the student:
        #   "penult"/"last"/None (default, unchanged) -> hook the input to the last Linear
        #     = the penultimate features. For a funnel (e.g. 16-24-8-3) this is the narrow
        #     8-dim waist -- forcing it to mimic a 1280-d teacher embedding strangles the
        #     bottleneck (a measured failure mode).
        #   "front"/"wide"/<int k> -> hook the output of an early (wide) Linear (front-end,
        #     index k, default 0), so the teacher's representation gradient reaches the trunk
        #     without passing through the waist. This is the training-only auxiliary head
        #     (the projector is never part of the student -> deleted at deploy).
        tap = cfg.get("feat_tap", "penult")
        if tap in (None, "penult", "last"):
            hook = lins[-1].register_forward_hook(
                lambda mod, inp, out: cap.__setitem__("f", inp[0]))
        else:
            ti = 0 if tap in ("front", "wide", "frontend") else int(tap)
            if not (0 <= ti < len(lins) - 1):
                raise ValueError("feat_tap index %r out of range for %d Linear layers "
                                 "(must be an early layer, not the head)" % (tap, len(lins)))
            hook = lins[ti].register_forward_hook(
                lambda mod, inp, out: cap.__setitem__("f", out))
        with torch.no_grad():
            student(Xtr[:2])
        projector = nn.Linear(cap["f"].shape[1], te_tr.shape[1]).to(device)

    params = list(student.parameters()) + (list(projector.parameters()) if projector else [])
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3))
    tw = cfg.get("target_weights")
    if tw is None:
        wmse = lambda p, t: ((p - t) ** 2).mean()                    # noqa: E731
    else:
        _w = torch.as_tensor(tw, dtype=torch.float32, device=device).reshape(1, -1)
        wmse = lambda p, t: (_w * (p - t) ** 2).mean()               # noqa: E731
    bs, epochs = cfg.get("batch_size", 256), cfg.get("epochs", 40)
    a, fw, rkd_w = float(alpha), float(feat_weight), float(rkd_weight)
    l1w = float(cfg.get("l1_weight", 0.0))         # sparsity-inducing L1 (prunability)
    # RATIO-CONSISTENCY loss (fixes the ~25% seed instability where a seed learns good
    # c/s but a bad C/S ratio). Penalizes (pred_c/true_c - pred_s/true_s)^2 == (relerr_c
    # - relerr_s)^2 on events with true c,s>0 -> drives the C and S relative errors to
    # match so they cancel in R=c/s. Uses training labels only (never the holdout).
    ratio_w = float(cfg.get("ratio_weight", 0.0))
    ratio_cols = cfg.get("ratio_cols")             # [c_idx, s_idx]; None or weight 0 -> off
    steps_per_epoch = max(1, (len(Xtr) + bs - 1) // bs)
    warmup_steps = max(1, int(float(feat_warmup) * epochs * steps_per_epoch))
    step = 0
    hist_total, hist_feat = [], []                 # per-epoch loss curves
    hist_lab, hist_tea, hist_fw = [], [], []       # per-epoch weighted term contributions

    # Gradual magnitude pruning (sparse-aware training): ramp sparsity 0 -> prune_target
    # over epochs [prune_start, prune_end], re-masking each epoch and enforcing the mask
    # after every step. The student trains with the growing mask, so it adapts to being
    # sparse (better than train-dense-then-prune). prune_target=0 -> off (unchanged).
    prune_target = float(cfg.get("prune_target", 0.0))
    gmp = prune_target > 0.0
    prune_start, prune_end = float(cfg.get("prune_start", 0.2)), float(cfg.get("prune_end", 0.8))
    gmp_masks = None

    # BEST-EPOCH checkpointing (select_best_val): evaluate validation every epoch and keep the
    # best checkpoint, rather than returning whatever the last epoch produced. At small label
    # fractions a run is only a few hundred optimizer steps -- short and noisy -- so the last
    # epoch is not reliably the best one. At full data the two mostly agree.
    #
    # Default off: enabling it changes the returned model for every caller, so callers that
    # want it pass it explicitly rather than having it switched on globally.
    #
    # The validation loss is against true labels only, never the teacher. A
    # teacher-inclusive criterion would select the model that best imitates the teacher rather
    # than the one that best predicts, and would not be comparable to the control arm, which
    # has no teacher to be scored against.
    sel_best = bool(cfg.get("select_best_val", False))
    best_vl, best_state, best_ep = float("inf"), None, -1
    hist_val = []

    def _val_loss():
        student.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva), 4096):
                vp.append(student(Xva[i:i + 4096]))
            p = torch.cat(vp) if vp else torch.zeros((0, y.shape[1]), device=device)
            yv = torch.as_tensor(yva_np, dtype=torch.float32, device=device)
            v = float(wmse(p, yv).detach()) if len(p) else float("inf")
        student.train()
        return v

    student.train()
    for ep in range(epochs):
        if gmp:
            prog = (ep / max(epochs, 1) - prune_start) / max(1e-6, prune_end - prune_start)
            s_now = gmp_sparsity(prog, prune_target)
            if s_now > 0.0:
                gmp_masks = global_magnitude_masks(student, s_now)
                apply_masks(student, gmp_masks)
        order = torch.randperm(len(Xtr), device=device)
        ep_tot, ep_feat, nb = 0.0, 0.0, 0
        # PER-TERM accounting (diagnostic only -- never touches the gradient). `ep_feat`
        # already existed but accumulates the raw `fl`: no `fw`, no `ramp`. That is the number
        # you want for "is the feature target being matched", and the wrong number for "how
        # much of the loss is the feature term" -- the two differ by fw*ramp, which is 3.56e-4
        # Accumulate the three weighted contributions separately, exactly as they enter
        # `loss`, and let the caller compute shares. Reading the raw unweighted series as a
        # loss share misrepresents a term whose weight is a unit conversion, not a fraction.
        ep_lab, ep_tea, ep_fw = 0.0, 0.0, 0.0
        for i in range(0, len(Xtr), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            pred = student(Xtr[b])                  # populates cap['f'] if hooked
            _lab = a * wmse(pred, ytr[b])
            _tea = (1.0 - a) * wmse(pred, t_tr[b])
            loss = _lab + _tea
            ep_lab += float(_lab.detach()); ep_tea += float(_tea.detach())
            fl_val = 0.0
            if do_feat:
                ramp = min(1.0, step / warmup_steps)
                te_b = te_tr[b.cpu()].to(device)                      # move this batch's embeds
                if regional:
                    sp = projector(student.region_features(Xtr[b], K))   # (B, K, D)
                    fl = ((_l2norm(sp) - _l2norm(te_b)) ** 2).sum(dim=-1).mean()
                else:
                    proj = projector(cap["f"])                           # (B, D)
                    fl = ((_l2norm(proj) - _l2norm(te_b)) ** 2).sum(dim=-1).mean()
                    if rkd_w > 0.0:                                       # relational (E3)
                        fl = fl + rkd_w * ((_rkd_dist(proj) - _rkd_dist(te_b)) ** 2).mean()
                loss = loss + fw * ramp * fl
                fl_val = float(fl.detach())
                ep_fw += fw * ramp * fl_val        # the weighted contribution, as it enters loss
            if ratio_w > 0.0 and ratio_cols is not None:   # ratio-consistency (train labels only)
                ci, si = int(ratio_cols[0]), int(ratio_cols[1])
                tc, ts = ytr[b][:, ci], ytr[b][:, si]
                pc, ps = pred[:, ci], pred[:, si]
                m = (tc > 1e-6) & (ts > 1e-6)
                if m.any():
                    tc, ts, pc, ps = tc[m], ts[m], pc[m], ps[m]
                    # stable cross-product: (pc*ts - ps*tc) == tc*ts*(relerr_c - relerr_s), so it
                    # is 0 exactly when the c/s relative errors match (cancel in R=c/s). Normalize
                    # by the batch-mean tc*ts (a bounded scalar) -> no per-event small-denominator
                    # blow-up (the old pc/tc form diverged when a true c or s was near zero).
                    cross = pc * ts - ps * tc
                    rl = (cross ** 2).mean() / ((tc * ts).mean() + 1e-6)
                    loss = loss + ratio_w * rl
            if l1w > 0.0:
                loss = loss + l1w * l1_penalty(student)
            loss.backward()
            opt.step()
            if gmp_masks is not None:
                apply_masks(student, gmp_masks)     # keep the GMP-pruned weights at 0
            step += 1
            ep_tot += float(loss.detach()); ep_feat += fl_val; nb += 1
        hist_total.append(ep_tot / max(nb, 1)); hist_feat.append(ep_feat / max(nb, 1))
        if sel_best:
            vl = _val_loss()
            hist_val.append(vl)
            # strict improvement, so ties keep the earlier epoch -- the simpler model, and it
            # makes the choice deterministic rather than "whichever came last".
            if vl < best_vl:
                # Detach + clone: state_dict() returns references to live tensors, so without
                # the copy every "saved" checkpoint would track the model and all of them
                # would end up identical to the final weights.
                best_vl, best_ep = vl, ep
                best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        _d = max(nb, 1)
        hist_lab.append(ep_lab / _d); hist_tea.append(ep_tea / _d); hist_fw.append(ep_fw / _d)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            # Shares of the three weighted terms. Denominator is their sum, not hist_total --
            # ratio/L1 penalties also land in `loss`, and including them would make the three
            # shares silently not add to 100%.
            _s = hist_lab[-1] + hist_tea[-1] + hist_fw[-1]
            _s = _s if _s > 0 else 1.0
            print("  sd epoch %d loss %.6g feat_raw %.4f | label %.1f%% teacher %.1f%% "
                  "feature %.1f%%" % (ep, hist_total[-1], hist_feat[-1],
                                      100 * hist_lab[-1] / _s, 100 * hist_tea[-1] / _s,
                                      100 * hist_fw[-1] / _s))

    if hook is not None:
        hook.remove()
    # Restore the best epoch before anything else reads the weights -- in particular before the
    # GMP re-mask below, so the exported sparsity pattern belongs to the model we actually keep.
    if sel_best and best_state is not None:
        student.load_state_dict(best_state)
    if gmp:                                          # enforce the exact final target sparsity
        gmp_masks = global_magnitude_masks(student, prune_target)
        apply_masks(student, gmp_masks)
    # batched validation forward -- an un-batched student(Xva) OOMs on big val sets
    # (DCH 500k events -> a ~9.6 GiB conv activation). Batch it.
    student.eval()
    vps = []
    with torch.no_grad():
        for i in range(0, len(Xva), 4096):
            vps.append(student(Xva[i:i + 4096]).cpu().numpy())
    pred = np.concatenate(vps) if vps else np.zeros((0, y.shape[1]), np.float32)
    metrics = regression_metrics(pred, yva_np)
    metrics["loss_history"] = hist_total
    # the balance, measured. The sweep over (alpha, feat_weight) is a sweep over how the loss
    # is split between ground truth, the teacher's answer, and the teacher's representation --
    # but `feat_weight` is a unit conversion, not a share (fl is L2-normalized and scale-free,
    # the MSE terms carry the target's units), so the same fw means different balances at
    # different target scales. Persist the shares so an arm can be described by what it
    # actually optimized rather than by the raw weight that produced it.
    _lt = {"label": hist_lab, "teacher": hist_tea, "feature": hist_fw}
    metrics["loss_terms"] = _lt
    _last = hist_lab[-1] + hist_tea[-1] + hist_fw[-1] if hist_lab else 0.0
    if _last > 0:
        metrics["loss_shares_final"] = {k: v[-1] / _last for k, v in _lt.items()}
    metrics["feat_share_final"] = (hist_fw[-1] / _last) if _last > 0 else 0.0
    if sel_best:
        metrics["val_loss_history"] = hist_val
        metrics["best_val_loss"] = best_vl
        metrics["best_epoch"] = best_ep
        metrics["last_epoch"] = epochs - 1
        metrics["select_best_val"] = True
    if gmp:
        eff, tot = effective_params(student, gmp_masks)
        metrics["effective_params"], metrics["target_sparsity"] = eff, prune_target
    # Stamp the input transform onto the model so predict() cannot silently disagree with
    # training. A model trained under a global scaler but predicted with per-event
    # z-scoring produces meaningless numbers and nothing would flag it.
    student._x_cfg = dict(cfg)
    student._x_scaler = cfg.get("_x_scaler")
    metrics["input_norm"] = str(cfg.get("input_norm", "per_event"))
    return student, metrics
