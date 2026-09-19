"""Fine-tuned backbone with a multi-task peak + count head.

One backbone drives a head with N + 1 outputs:

  * N per-sample cells -- a 3-class label per waveform sample (0 noise, 1 primary,
    2 secondary), the dense form of the simulation's per-sample tags.
  * one count cell -- the primary-cluster count, which is the reported quantity.

The per-sample cells give the encoder a dense, localized signal: learn where the peaks are
and which are primary. The count cell optimizes the reported target directly. An optional
consistency term ties the two, requiring the regressed count to agree with the count implied
by the per-sample map; `consistency_weight = 0` disables it and leaves the two heads coupled
only through the shared backbone.

PER-SAMPLE resolution is PATCH-LOCAL. The backbone emits one embedding per 32-sample
patch, so a 3,008-sample waveform becomes 94 tokens. The per-sample head expands each token
back to its own 32 samples through a shared `Linear(embed_dim -> patch * n_classes)`; each
token therefore predicts only the samples it covers. The count head mean-pools the tokens and
passes them through an mlp to a single output.

Pure ASCII.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ft_finetune import _build_backbone, _amp_dtype, _autocast_ctx   # noqa: E402


class PeakCountModel(nn.Module):
    """backbone -> (per-sample 3-class logits (B,L,C), count (B,1))."""

    def __init__(self, backbone, input_len, head_hidden=(128,), n_classes=3,
                 patch=32):
        super().__init__()
        self.backbone = backbone
        self.n_classes = int(n_classes)
        self.patch = int(getattr(backbone, "patch", patch))
        self.n_tokens = int(input_len) // self.patch
        emb = backbone.embed_dim
        # per-sample head: each token -> its patch*n_classes sample logits
        self.ps_head = nn.Linear(emb, self.patch * self.n_classes)
        # count head: mean-pooled tokens -> mlp -> 1
        dims = [emb] + list(head_hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.count_head = nn.Sequential(*layers)

    def forward(self, x):                       # x: (B, L)
        tok = self.backbone(x)                  # (B, T, emb)
        b, t, _ = tok.shape
        # ftpc is the first model to require token_count * patch == input_len (fx/ft
        # mean-pool, so they never caught a token/length mismatch). Guard loudly so
        # a real-TimesFM token-count surprise fails on the first batch, not silently
        # via misaligned per-sample supervision.
        if t * self.patch != x.shape[1]:
            raise ValueError(
                "ftpc token/length mismatch: backbone gave %d tokens x patch %d = %d "
                "!= input length %d (pad input to a multiple of patch, or fix patch)"
                % (t, self.patch, t * self.patch, x.shape[1]))
        ps = self.ps_head(tok)                  # (B, T, patch*C)
        ps = ps.reshape(b, t * self.patch, self.n_classes)   # (B, L, C)
        cnt = self.count_head(tok.mean(dim=1))  # (B, 1)
        return ps, cnt

    def embed(self, x):
        """Pooled embed_dim representation (the same mean-pooled tokens the count head
        consumes) -- the teacher 'features' for feature distillation into a student."""
        return self.backbone(x).mean(dim=1)     # (B, embed_dim)


def class_weights_from_targets(ps_target, n_classes=3, device="cpu"):
    """Inverse-frequency class weights (noise is ~97% of samples, so CE would
    collapse to all-noise without this -- the same sparse-positive trap pf hit)."""
    c = np.bincount(np.asarray(ps_target).reshape(-1).astype(np.int64),
                    minlength=n_classes).astype(np.float64)
    c = np.maximum(c, 1.0)
    w = c.sum() / (n_classes * c)               # normalized inverse frequency
    return torch.tensor(w, dtype=torch.float32, device=device)


def multitask_loss(ps_logits, cnt_pred, ps_target, cnt_target,
                   class_weights=None, w_count=1.0, w_consistency=0.0):
    """CE(per-sample 3-class) + w_count*MSE(count) [+ w_consistency*MSE(count,
    summed primary prob)]. Set w_consistency=0 to drop the tie exactly.

    The consistency target is the peak map's own primary count -- the number of
    samples argmaxed to primary (class 1), detached so it acts purely as a target
    for the count head (the tie pulls the count head toward the CE-driven peak map,
    never the peak map toward the count, so it cannot fight CE or collapse the
    classes). Using the argmax count (not the summed softmax prob) avoids the
    uncalibrated-softmax inflation that would otherwise drag the count head to
    ~L/3. Callers still ramp w_consistency over a warmup for a gentle start."""
    c = ps_logits.shape[-1]
    ce = F.cross_entropy(ps_logits.reshape(-1, c),
                         ps_target.reshape(-1).long(), weight=class_weights)
    cnt_pred = cnt_pred.reshape(-1)
    count_mse = F.mse_loss(cnt_pred, cnt_target.reshape(-1).float())
    total = ce + w_count * count_mse
    cons = ps_logits.new_zeros(())
    if w_consistency > 0.0:
        peak_count = (ps_logits.detach().argmax(dim=-1) == 1).sum(dim=1).float()
        cons = F.mse_loss(cnt_pred, peak_count)
        total = total + w_consistency * cons
    return total, {"ce": float(ce.detach()), "count_mse": float(count_mse.detach()),
                   "consistency": float(cons.detach())}


def _primary_metrics(ps_logits, ps_target):
    """Per-sample PRIMARY-class diagnostics on a held-out slice: recall, the
    predicted-primary rate (all-noise guard), and count MAE via argmax."""
    pred = ps_logits.argmax(axis=-1)
    tgt = ps_target
    is_p = tgt == 1
    tp = float(((pred == 1) & is_p).sum())
    recall = tp / max(float(is_p.sum()), 1.0)
    pred_pos_rate = float((pred == 1).mean())
    true_pos_rate = float(is_p.mean())
    cnt_argmax = (pred == 1).sum(axis=1)
    cnt_true = is_p.sum(axis=1)
    mae = float(np.abs(cnt_argmax - cnt_true).mean())
    return {"primary_recall": recall, "pred_primary_rate": pred_pos_rate,
            "true_primary_rate": true_pos_rate, "count_mae_argmax": mae}


def train_peakcount(X, ps_y, cnt_y, cfg, seed=0, fraction=100.0, device="cpu",
                    stub=False, input_len=None, verbose=False, ckpt_path=None,
                    ckpt_every=1, plot_path=None):
    """Fine-tune backbone + multi-task head on `fraction`% of raw waveforms.
    ps_y: (N, L) int {0,1,2} per-sample target. cnt_y: (N,) primary count.
    Returns (model, metrics). Mirrors train_ft (AdamW, cosine, grad-clip 1.0,
    head_lr > backbone lr).

    Crash-safe: if `ckpt_path` is given, a full resume checkpoint
    (model+opt+sched+epoch, and it doubles as the saved weights) is written every
    `ckpt_every` epochs; if that file already exists on entry, training resumes
    from its last epoch (a killed fine-tune restarts where it stopped). This makes
    a full retrain robust to job time limits."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    ps_y = np.asarray(ps_y, dtype=np.int64)
    cnt_y = np.asarray(cnt_y, dtype=np.float32).reshape(-1)
    n = len(X)
    perm = rng.permutation(n)
    n_use = max(int(round(n * fraction / 100.0)), 32)
    idx = perm[:n_use]
    n_val = max(int(round(len(idx) * cfg.get("val_fraction", 0.1))), 8)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    L = input_len or X.shape[1]

    torch.manual_seed(seed)
    backbone = _build_backbone(cfg, L, device, stub)
    model = PeakCountModel(backbone, L, cfg.get("head_hidden", [128]),
                           n_classes=int(cfg.get("n_classes", 3)),
                           patch=int(cfg.get("patch", 32))).to(device)

    head_params = list(model.ps_head.parameters()) + list(model.count_head.parameters())
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    # uniform backbone LR by default; layer-wise LR decay if cfg['layer_lr_decay'] is set
    from .ft_finetune import lr_param_groups
    opt = torch.optim.AdamW(lr_param_groups(head_params, model.backbone, cfg),
                            weight_decay=cfg.get("weight_decay", 0.01))
    epochs = int(cfg.get("epochs", 15))
    bs = int(cfg.get("batch_size", 64))
    accum = max(int(cfg.get("grad_accum", 1)), 1)       # micro-batches per optimizer step
    micro_per_epoch = (len(tr_idx) + bs - 1) // bs
    opt_per_epoch = (micro_per_epoch + accum - 1) // accum
    n_steps = max(epochs * opt_per_epoch, 1)            # optimizer steps (LR cosine + warmup)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    w_count = float(cfg.get("count_weight", 1.0))
    w_cons = float(cfg.get("consistency_weight", 0.0))
    # ramp consistency 0 -> w_cons over the first `warmup` fraction of steps so CE
    # builds a sparse peak map before the count/peak-map tie is enforced.
    warmup_steps = max(int(n_steps * float(cfg.get("consistency_warmup", 0.3))), 1)
    cw = class_weights_from_targets(ps_y[tr_idx], model.n_classes, device) \
        if cfg.get("class_weighted", True) else None
    amp_dtype = _amp_dtype(cfg.get("amp"))       # bf16 mixed precision for full-ft (None=off)

    Xtr = torch.as_tensor(X[tr_idx], dtype=torch.float32)
    pstr = torch.as_tensor(ps_y[tr_idx], dtype=torch.int64)
    cntr = torch.as_tensor(cnt_y[tr_idx], dtype=torch.float32)

    # resume from a prior checkpoint (mid-fine-tune restart after a killed job)
    start_epoch = 0
    start_sub = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch = int(ck.get("epoch", 0))
        start_sub = int(ck.get("sub_step", 0))       # QUARTER-EPOCH resume point (0 = epoch start)
        print("    [ftpc] resumed %s from epoch %d/%d (micro-batch %d)"
              % (ckpt_path, start_epoch, epochs, start_sub))

    grad_flowed = False
    first_loss = last_loss = None
    step = start_epoch * opt_per_epoch + start_sub // accum   # opt-steps already done (LR warmup)
    n_sub = max(1, int(cfg.get("ckpt_sub", 4)))              # quarter-epoch checkpoints per epoch
    hist = {"total": [], "ce": [], "count_mse": [], "consistency": []}  # per-step, for the loss curve
    epoch_hist = []                                                     # per-epoch means, for the JSON
    model.train()
    for ep in range(start_epoch, epochs):
        # shuffle seeded by (seed, ep) so the batch order is identical whether the run is
        # continuous or resumed (whole-epoch or mid-epoch) -> the resumed run is numerically
        # equivalent to the single-shot run.
        _g = torch.Generator().manual_seed(int(seed) * 100003 + ep)
        order = torch.randperm(len(Xtr), generator=_g)
        ep_acc = {"total": 0.0, "ce": 0.0, "count_mse": 0.0, "consistency": 0.0, "n": 0}
        n_micro = (len(Xtr) + bs - 1) // bs
        resume_sub = start_sub if ep == start_epoch else 0   # skip micro-batches done pre-crash
        # QUARTER-EPOCH checkpoints (only when there are enough micro-batches to matter).
        sub_ck = ckpt_path is not None and n_micro >= 2 * n_sub
        quarter = max(1, n_micro // n_sub)
        next_ck = ((resume_sub // quarter) + 1) * quarter if sub_ck else (n_micro + 1)
        opt.zero_grad()
        for mi, i in enumerate(range(0, len(Xtr), bs)):
            if mi < resume_sub:                 # already trained pre-crash -> skip (order reproduced)
                continue
            b = order[i:i + bs]
            xb = Xtr[b].to(device)
            psb = pstr[b].to(device)
            cnb = cntr[b].to(device)
            wc = w_cons * min(1.0, step / warmup_steps)     # consistency warmup
            with _autocast_ctx(amp_dtype, device):   # bf16 forward (backward outside)
                ps_logits, cnt_pred = model(xb)
                loss, comps = multitask_loss(ps_logits, cnt_pred, psb, cnb,
                                             class_weights=cw, w_count=w_count,
                                             w_consistency=wc)
            (loss / accum).backward()                # scale -> accumulated grad == big-batch grad
            if bb_params and not grad_flowed:
                g = sum(float(p.grad.abs().sum()) for p in bb_params if p.grad is not None)
                grad_flowed = g > 0
            if (mi + 1) % accum == 0 or mi == n_micro - 1:   # step at the accumulation boundary
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                # mid-epoch (quarter) checkpoint at a clean (grads-zeroed) boundary: resume
                # epoch `ep` at micro-batch mi+1. not the last micro-batch (epoch-end saves below).
                if sub_ck and (mi + 1) >= next_ck and (mi + 1) < n_micro:
                    save_peakcount(ckpt_path, model, cfg, L, opt=opt, sched=sched,
                                   epoch=ep, sub_step=mi + 1)
                    next_ck += quarter
            lv = float(loss.item())
            first_loss = lv if first_loss is None else first_loss
            last_loss = lv
            hist["total"].append(lv)
            for k in ("ce", "count_mse", "consistency"):
                hist[k].append(comps[k]); ep_acc[k] += comps[k]
            ep_acc["total"] += lv; ep_acc["n"] += 1
        n = max(ep_acc["n"], 1)
        epoch_hist.append({"epoch": ep + 1, **{k: ep_acc[k] / n for k in
                          ("total", "ce", "count_mse", "consistency")}})
        if ckpt_path and ((ep + 1) % ckpt_every == 0 or ep == epochs - 1):
            save_peakcount(ckpt_path, model, cfg, L, opt=opt, sched=sched, epoch=ep + 1, sub_step=0)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print("    ftpc epoch %d loss %.4f (ce %.4f count_mse %.4f cons %.4f)"
                  % (ep, last_loss, comps["ce"], comps["count_mse"], comps["consistency"]))

    model.eval()
    ps_val, cnt_val = _predict_raw(model, X[val_idx], device, bs)
    m = _primary_metrics(ps_val, ps_y[val_idx])
    m["count_mae_head"] = float(np.abs(cnt_val.reshape(-1) - cnt_y[val_idx]).mean())
    m.update(first_loss=first_loss, last_loss=last_loss,
             grad_flowed=(bool(grad_flowed) if bb_params else None),
             trainable_params=int(n_train), total_params=int(n_total),
             consistency_weight=w_cons, loss_history=epoch_hist)
    model._ftpc_bs = bs
    return model, m


def _predict_raw(model, X, device="cpu", bs=64):
    """Return (per-sample class logits argmax-ready (N,L,C), count (N,1))."""
    model.eval()
    ps_out, cnt_out = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)
            ps, cnt = model(xb)
            ps_out.append(ps.cpu().numpy())
            cnt_out.append(cnt.cpu().numpy())
    return np.concatenate(ps_out, 0), np.concatenate(cnt_out, 0)


def predict_counts(model, X, device="cpu", batch=None):
    """Primary-count per event from the count head (the separation input)."""
    bs = batch or getattr(model, "_ftpc_bs", 64)
    _, cnt = _predict_raw(model, X, device, bs)
    return cnt.reshape(-1)


def predict_peakmap(model, X, device="cpu", batch=None):
    """Per-sample 3-class argmax label map (N, L)."""
    bs = batch or getattr(model, "_ftpc_bs", 64)
    ps, _ = _predict_raw(model, X, device, bs)
    return ps.argmax(axis=-1)


def predict_embed(model, X, device="cpu", batch=None):
    """Pooled embedding per event (N, embed_dim) -- the ftpc teacher's representation
    for feature distillation. No grad; batched so a full train set never OOMs."""
    bs = batch or getattr(model, "_ftpc_bs", 64)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)
            outs.append(model.embed(xb).cpu().numpy())
    return np.concatenate(outs, 0)


def eval_counts(model, X, device="cpu", batch=None):
    """one batched forward per event set -> (count_head (N,), peakmap_count (N,)),
    computing the argmax-primary count inside the loop so the full (N, L, C) logits
    are never materialized. Use this for separation eval (the 60k/40k pion/kaon sets
    would otherwise allocate ~2 GB and take 4 forward passes)."""
    bs = batch or getattr(model, "_ftpc_bs", 64)
    model.eval()
    head, peak = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)
            ps, cnt = model(xb)
            head.append(cnt.reshape(-1).cpu().numpy())
            peak.append((ps.argmax(dim=-1) == 1).sum(dim=1).to(torch.float32).cpu().numpy())
    return np.concatenate(head), np.concatenate(peak)


def save_peakcount(path, model, cfg, length, opt=None, sched=None, epoch=0, sub_step=0):
    """Save weights (+ optional opt/sched/epoch/sub_step for mid-fine-tune resume).
    sub_step>0 marks a QUARTER-EPOCH (mid-epoch) checkpoint: resume at micro-batch
    `sub_step` of epoch `epoch`. sub_step=0 is a normal epoch-boundary checkpoint."""
    payload = {"model": model.state_dict(), "arch": "ftpc",
               "head_hidden": list(cfg.get("head_hidden", [128])),
               "n_classes": int(model.n_classes), "patch": int(model.patch),
               "length": int(length)}
    if opt is not None:
        payload.update(opt=opt.state_dict(), sched=sched.state_dict(), epoch=int(epoch),
                       sub_step=int(sub_step))
    tmp = path + ".tmp"                       # atomic write (no half-file on kill)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_peakcount(path, cfg, device="cpu", stub=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    backbone = _build_backbone(cfg, ck["length"], device, stub)
    model = PeakCountModel(backbone, ck["length"], ck["head_hidden"],
                           n_classes=ck["n_classes"], patch=ck["patch"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
