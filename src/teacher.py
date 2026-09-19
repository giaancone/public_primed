"""End-to-end fine-tuning of the TimesFM backbone on a waveform regression target.

The backbone weights are trainable and the task loss backpropagates through the transformer,
so the encoder adapts to detector waveforms rather than merely being read out.

The optimizer recipe follows the official TimesFM fine-tuning example -- AdamW, cosine
schedule, gradient clipping at 1.0 -- with three deliberate differences, each forced by how
this checkpoint loads:

  1. Load through the `timesfm` package, not `transformers`. Loading
     `google/timesfm-2.5-200m-pytorch` through `transformers` silently random-initializes
     every weight because the key layouts do not match: `from_pretrained` reports all
     transformer layers, the input projection and the output projection as missing and
     newly initialized. The run then trains from scratch while appearing to fine-tune. Use
     `timesfm.TimesFM_2p5_200M_torch.from_pretrained`.
  2. A regression head is driven off the last-layer hidden states instead of the
     forecasting loss.
  3. Call `forward()` directly for A differentiable encode. The package's
     `forecast()` / `model.decode()` path is wrapped in `torch.no_grad()` and crosses a
     numpy boundary, so no gradient reaches the backbone through it -- captured hidden
     states come back with `requires_grad=False`. See TimesFMBackbone, which reimplements
     the normalization and patching that path performs internally.

Two backbones share one interface (`.embed_dim`, `forward(x) -> (B, tokens, embed_dim)`,
gradient-enabled), so the training loop, head, loss and evaluation are identical for both:

  * TimesFMBackbone   -- the real model (lazy `timesfm` import, GPU).
  * TorchStubBackbone -- a small differentiable conv-to-token network, so the whole loop is
    runnable with no TimesFM install and no GPU.

Pure ASCII.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from src.metrics import regression_metrics, make_loss


# ------------------------------------------------------------------------- #
# Memory-aware helpers: bf16 autocast for the forward pass.                 #
# off by default; enable via the `amp` config key when activation memory at   #
# a long context is the binding constraint.                                   #
# ------------------------------------------------------------------------- #
def _apply_freeze_to(model, freeze, n):
    """Freeze the backbone except its last `n` transformer blocks.

    This release ships the partial fine-tune the paper reports: the backbone is frozen
    and only `model.stacked_xf[-n:]` plus the task head train (~20M of 231M). A full
    fine-tune of every backbone weight was also run during development but is not the
    model reported here, so `freeze` must be "last_n"."""
    if freeze != "last_n":
        raise ValueError("freeze=%r -- this release ships 'last_n' only" % freeze)
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.stacked_xf[-n:]:
        for p in layer.parameters():
            p.requires_grad_(True)


def lr_param_groups(head_params, backbone, cfg):
    """Build AdamW param groups. Head at `head_lr`. Backbone at a single uniform `lr`
    by default -- but if `cfg['layer_lr_decay']` (xi in (0,1)) is set, use LAYER-WISE
    LR decay (llrd): the top transformer block (nearest the head) gets `lr`, and each
    block toward the input is xi x smaller (block i from the input -> lr * xi^(nblk-1-i)).
    Non-block backbone params (tokenizer/embeddings/final norm) get the smallest (bottom)
    LR. Shared by train_ft (DRO) and train_peakcount (DCH). Falls back to uniform when
    xi is unset/>=1 or the backbone has no `stacked_xf` (e.g. the CPU stub)."""
    head_lr = cfg.get("head_lr", cfg.get("lr", 1e-4) * 10)
    lr = float(cfg.get("lr", 1e-4))
    groups = [{"params": list(head_params), "lr": head_lr}]
    bb_trainable = [p for p in backbone.parameters() if p.requires_grad]
    if not bb_trainable:
        return groups
    xi = cfg.get("layer_lr_decay")
    blocks = list(getattr(getattr(backbone, "model", None), "stacked_xf", []) or [])
    if not xi or float(xi) >= 1.0 or not blocks:
        groups.append({"params": bb_trainable, "lr": lr})       # uniform (default, unchanged)
        return groups
    xi = float(xi)
    nblk = len(blocks)
    seen = set()
    for i, blk in enumerate(blocks):        # i=0 nearest input (bottom) ... nblk-1 nearest head (top)
        ps = [p for p in blk.parameters() if p.requires_grad]
        for p in ps:
            seen.add(id(p))
        if ps:
            groups.append({"params": ps, "lr": lr * (xi ** (nblk - 1 - i))})
    other = [p for p in bb_trainable if id(p) not in seen]       # tokenizer/embeddings/final norm
    if other:
        groups.append({"params": other, "lr": lr * (xi ** (nblk - 1))})   # smallest (bottom) LR
    return groups


def _amp_dtype(amp):
    """Map a cfg `amp` value to an autocast dtype (None = off). Only bf16 is supported
    for training (it needs no GradScaler); fp16 is rejected loudly rather than silently
    under-scaling gradients."""
    if amp in (None, False, "", "none", "off"):
        return None
    a = str(amp).lower()
    if a in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("amp=%r unsupported (use 'bf16' or omit; fp16 would need a "
                     "GradScaler this path does not wire up)" % (amp,))


def _autocast_ctx(amp_dtype, device):
    """Context manager for the forward pass: torch.autocast if amp_dtype set, else a
    trivial no-op. Backward runs outside it (standard AMP pattern; bf16 needs no scaler)."""
    if amp_dtype is None:
        import contextlib
        return contextlib.nullcontext()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=dev_type, dtype=amp_dtype)


class TorchStubBackbone(nn.Module):
    """Tiny differentiable stand-in for the TimesFM backbone (tests only).

    A strided Conv1d turns the raw waveform into one feature vector per 32-sample
    patch -- the same (B, n_tokens, embed_dim) contract the real backbone exposes --
    so the pooling + head + backprop path is byte-for-byte the code that runs with
    TimesFM. It is not a foundation model; it just has enough learnable capacity to
    prove the loop reduces loss and that gradients reach the backbone."""

    def __init__(self, input_len, embed_dim=64, patch=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = patch
        self.n_tokens = input_len // patch
        self.proj = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=patch, stride=patch), nn.GELU())

    def forward(self, x):                     # x: (B, L)
        h = self.proj(x.unsqueeze(1))         # (B, embed_dim, n_tokens)
        return h.transpose(1, 2)              # (B, n_tokens, embed_dim)


class TimesFMBackbone(nn.Module):
    """The TimesFM backbone as a differentiable encoder.

    Loaded through the `timesfm` package, which is the only layout that yields the real
    pretrained weights -- see the module docstring for what `transformers` does instead.

    `freeze="last_n"` trains the last `freeze_n` transformer blocks plus the head and holds
    the rest frozen. It is the only supported setting in this release.

    The differentiable forward is the crux of this class. TimesFM's
    `forecast()` -> `model.decode()` path is wrapped in `torch.no_grad()` and crosses a numpy
    boundary, so nothing backpropagates through it -- an approach that hooks hidden states out
    of `forecast()` trains the head and leaves the backbone untouched, silently. Instead
    `self.model.forward()` is called directly and the preprocessing that path would have
    applied is reproduced here using TimesFM's own `revin` / `update_running_stats`:

        full-context RevIN -> reshape into `p`-sample patches
        -> per-patch causal running-stats RevIN -> mask-zero -> model.forward

    This yields the same embeddings the frozen path produces while carrying gradients into the
    backbone. No `compile()` is needed, since that only wires up the decode path being
    bypassed. Traces are fixed length and the context is a multiple of `p`, so there is no
    padding and the masks are all-False."""

    def __init__(self, checkpoint, context_len=512, batch_size=64,
                 freeze="last_n", freeze_n=2, seed=0):
        super().__init__()
        import timesfm                        # lazy: Mac stub selftest never needs it
        from timesfm.torch.util import revin, update_running_stats
        self._revin, self._running_stats = revin, update_running_stats
        self._tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
        self.model = self._tfm.model          # registered submodule -> .to()/params propagate
        self.embed_dim = int(self.model.tokenizer.output_layer.out_features)  # 1280
        self.patch = int(self.model.p)        # 32
        self.batch_size = batch_size
        _apply_freeze_to(self.model, freeze, freeze_n)
        # Full-231M ft at the full-waveform context needs activation checkpointing to
        # fit in 40 GB -> enable it on the transformer stack when asked (real backbone
        # only; the stub has no stacked_xf).
        self._trainable_backbone = any(p.requires_grad for p in self.model.parameters())

    def _apply_freeze(self, freeze, n):
        _apply_freeze_to(self.model, freeze, n)   # head_only -> backbone frozen

    def forward(self, x):                     # x: (B, L) -> (B, tokens, embed_dim)
        revin, running_stats, p = self._revin, self._running_stats, self.patch
        # 1) full-context RevIN (what `_compiled_decode` applies before decode)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        xn = revin(x, mu, sigma, reverse=False)
        # 2) patchify into p-sample patches; fixed-length traces -> no padding
        pi = xn.reshape(x.shape[0], -1, p)
        pm = torch.zeros_like(pi, dtype=torch.bool)
        # 3) per-patch causal running-stats RevIN (what `model.decode` applies)
        B, dev = pi.shape[0], pi.device
        n = torch.zeros(B, device=dev)
        rmu = torch.zeros(B, device=dev)
        rsig = torch.zeros(B, device=dev)
        cmu, csig = [], []
        for i in range(pi.shape[1]):
            (n, rmu, rsig), _ = running_stats(n, rmu, rsig, pi[:, i], pm[:, i])
            cmu.append(rmu); csig.append(rsig)
        normed = revin(pi, torch.stack(cmu, 1), torch.stack(csig, 1), reverse=False)
        normed = torch.where(pm, torch.zeros_like(normed), normed)
        # 4) the model's own forward (tokenizer -> stacked_xf), grad-enabled
        (_, output_embeddings, _, _), _ = self.model.forward(normed, pm)
        # Guard during training only. The failure it catches -- the package internally
        # stripping grad, leaving the backbone frozen while the run looks healthy -- only
        # matters when gradients must flow. Under eval/no_grad, requires_grad=False is
        # expected, so checking there would fire spuriously.
        if self.training and self._trainable_backbone and not output_embeddings.requires_grad:
            raise RuntimeError(
                "backbone forward produced non-grad hidden states -- the "
                "differentiable-forward assumption no longer holds (timesfm internals "
                "changed?). The backbone would train nothing; investigate before trusting.")
        return output_embeddings


class FineTuneModel(nn.Module):
    """backbone -> mean-pool tokens -> mlp head -> n_targets (count)."""

    def __init__(self, backbone, hidden, n_targets):
        super().__init__()
        self.backbone = backbone
        dims = [backbone.embed_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers += [nn.Linear(dims[-1], n_targets)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        tok = self.backbone(x)                # (B, tok, embed_dim)
        return self.head(tok.mean(dim=1))     # mean-pool -> (B, n_targets)

    def embed(self, x):
        """Pooled embed_dim (1280-d for TimesFM) representation before the head --
        the teacher 'features' used for feature/hint distillation. Unlike the scalar
        output (which for a good regressor ~= the label, so it teaches ~nothing), the
        embedding carries the foundation model's learned pulse representation."""
        return self.backbone(x).mean(dim=1)   # (B, embed_dim)


def _build_backbone(cfg, input_len, device, stub):
    if stub or cfg.get("stub"):
        return TorchStubBackbone(input_len, embed_dim=cfg.get("stub_dim", 64))
    return TimesFMBackbone(cfg["checkpoint"], context_len=input_len,
                           batch_size=cfg.get("batch_size", 64),
                           freeze=cfg.get("freeze", "last_n"),
                           freeze_n=cfg.get("freeze_n", 2),
                           )


def _predict_batched(model, X, device, bs):
    """Batched inference. `no_grad` is not optional here: the params still require grad, so
    without it the full autograd graph -- including the 196-iteration per-patch RevIN loop --
    is built and discarded for every batch. The DRO holdout is 40,000 events at bs=8 = 5,000
    batches per cell, x12 cells. predict_embed() below always had this; this path did not."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32).to(device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def predict_embed(model, X, device="cpu", batch=None):
    """Teacher pooled embedding over X (B, embed_dim), for feature distillation.
    Mirrors predict(); raw X (TimesFM RevIN is internal). No grad through teacher."""
    X = np.asarray(X, dtype=np.float32)
    bs = batch or getattr(model, "_ft_bs", 64)
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32).to(device)
            outs.append(model.embed(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _save_ckpt(path, model, opt, sched, next_epoch, first_loss, last_loss, grad_flowed,
               sub_step=0, hist=None):
    """Atomically write a resume checkpoint: model + optimizer + scheduler state +
    where to resume (+ original first_loss/last_loss/grad_flowed). This file is also
    the saved fine-tuned weights -- ck['model'] is the state_dict.

    `next_epoch` + `sub_step` say exactly where to resume: at micro-batch `sub_step`
    of epoch `next_epoch`. sub_step=0 is a normal epoch-boundary checkpoint; sub_step>0
    is a QUARTER-EPOCH (mid-epoch) checkpoint so a crash/preemption costs <=1/4 epoch,
    not a whole (~2.5 h) epoch.

    `hist` = (epoch_hist, sub_hist, step_losses). Persisting these is not optional.
    They used to be re-initialized empty on every resume, so with --epochs-per-run the final
    JSON row and the loss png contained only the last chunk's epoch -- destroying exactly the
    diagnostic that separates undertraining from overfitting, which is why sub_hist was added
    in the first place. Cost: ~80k floats at fraction 100, i.e. <1 MB against a ~1.1 GB
    checkpoint. Stored under one key so an old checkpoint (no "hist") resumes fine."""
    tmp = path + ".tmp"
    payload = {"model": model.state_dict(), "opt": opt.state_dict(),
               "sched": sched.state_dict(), "epoch": int(next_epoch),
               "sub_step": int(sub_step),
               "first_loss": first_loss, "last_loss": last_loss,
               "grad_flowed": bool(grad_flowed)}
    if hist is not None:
        _eh, _sh, _sl = hist[0], hist[1], hist[2]
        payload["hist"] = {"epoch_hist": list(_eh), "sub_hist": list(_sh),
                           "step_losses": [float(v) for v in _sl]}
        # IN-EPOCH accumulators. Without these the epoch mean after a mid-epoch resume covers
        # only the tail of the epoch (ep_sum/ep_n reset at the epoch top and the pre-crash
        # micro-batches are skipped). With wall-clock checkpointing on a multi-hour epoch a
        # mid-epoch resume is the norm, so that bias would be systematic, not occasional.
        if len(hist) > 3 and hist[3] is not None:
            payload["hist"]["acc"] = [float(v) for v in hist[3]]   # ep_sum, ep_n, sub_sum, sub_n
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_ft(X, y, cfg, seed=0, fraction=100.0, device="cpu", stub=False,
             input_len=None, verbose=False, ckpt_path=None, ckpt_every=1,
             plot_path=None, max_epochs_this_run=None):
    """Fine-tune backbone+head on `fraction`% of raw waveforms. Returns
    (model, metrics). no external standardization: the backbone normalizes each instance
    internally, so raw traces are fed in.

    If `ckpt_path` is given, a resume checkpoint (model+optimizer+scheduler+epoch)
    is written every `ckpt_every` epochs; if that file already exists on entry,
    training resumes from it (mid-fine-tune restart after a killed job), and the
    file doubles as the saved fine-tuned weights."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    perm = rng.permutation(n)
    n_use = max(int(round(n * fraction / 100.0)), 32)
    idx = perm[:n_use]
    n_val = max(int(round(len(idx) * cfg.get("val_fraction", 0.1))), 8)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    L = input_len or X.shape[1]

    torch.manual_seed(seed)
    backbone = _build_backbone(cfg, L, device, stub)
    model = FineTuneModel(backbone, cfg.get("head_hidden", [128]), y.shape[1]).to(device)

    head_params = list(model.head.parameters())
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # Reference recipe: AdamW + cosine schedule + grad-clip 1.0. Head gets a higher LR
    # than the (pretrained) backbone. With cfg['layer_lr_decay'] set, the backbone uses
    # layer-wise LR decay (top blocks fast, early blocks stable); else a uniform backbone LR.
    opt = torch.optim.AdamW(lr_param_groups(head_params, model.backbone, cfg),
                            weight_decay=cfg.get("weight_decay", 0.01))
    epochs = cfg.get("epochs", 15)
    bs = cfg.get("batch_size", 64)
    accum = max(int(cfg.get("grad_accum", 1)), 1)      # micro-batches per optimizer step
    micro_per_epoch = (len(tr_idx) + bs - 1) // bs
    opt_per_epoch = (micro_per_epoch + accum - 1) // accum
    n_steps = max(epochs * opt_per_epoch, 1)           # optimizer steps (LR cosine T_max)
    # Optional linear warmup before the cosine decay, for long contexts where the first few
    # hundred steps are unstable at the target learning rate. Opt-in: `warmup_frac` absent or
    # 0 gives the plain cosine below.
    # do not enable this MID-RUN. A SequentialLR state_dict cannot be loaded into a
    # CosineAnnealingLR or vice versa, so toggling it breaks resume from an existing checkpoint.
    _warm = float(cfg.get("warmup_frac", 0.0) or 0.0)
    if _warm > 0.0:
        n_warm = max(1, min(int(round(_warm * n_steps)), n_steps - 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0 / max(n_warm, 1),
                                               end_factor=1.0, total_iters=n_warm),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps - n_warm, 1))],
            milestones=[n_warm])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    # Per-target weighted MSE when cfg has `target_weights` (DRO uses [4,1,0.3]
    # over [c,s,t0]); None -> plain MSELoss, so the DCH single-target path is
    # byte-for-byte unchanged. Same helper the metrics module provides.
    loss_fn = make_loss(cfg.get("target_weights"), device)
    amp_dtype = _amp_dtype(cfg.get("amp"))      # bf16 mixed precision for full-ft (None=off)

    # Mid-fine-tune resume: reload model/opt/scheduler and continue from the ckpt.
    start_epoch = 0
    start_sub = 0                               # micro-batch to resume at within start_epoch
    first_loss = last_loss = None
    grad_flowed = False
    resumed_hist = ([], [], [], [])             # (epoch_hist, sub_hist, step_losses, acc)
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = int(ck.get("epoch", 0))
        start_sub = int(ck.get("sub_step", 0))   # 0 for old (epoch-boundary) checkpoints
        first_loss = ck.get("first_loss"); last_loss = ck.get("last_loss")
        grad_flowed = bool(ck.get("grad_flowed", False))
        _rh = ck.get("hist") or {}          # {} for checkpoints written before hist was saved
        resumed_hist = (list(_rh.get("epoch_hist", [])),
                        list(_rh.get("sub_hist", [])),
                        list(_rh.get("step_losses", [])),
                        list(_rh.get("acc", []) or []))
        print("    [ft] resumed %s from epoch %d/%d (micro-batch %d)%s"
              % (ckpt_path, start_epoch, epochs, start_sub,
                 ("  +%d epoch / %d sub loss points"
                  % (len(resumed_hist[0]), len(resumed_hist[1]))) if _rh else
                 "  (no loss history in ckpt -- pre-fix file, curve starts here)"))

    Xtr = torch.as_tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32)

    # Seeded from the checkpoint (empty on a fresh start) so the curve spans the whole
    # fine-tune, not just the current --epochs-per-run chunk.
    epoch_hist, sub_hist_resumed, step_losses = list(resumed_hist[0]), list(resumed_hist[1]), \
        list(resumed_hist[2])                   # per-epoch means, sub-epoch marks, per-step
    # Sub-epoch history, logged at the same cadence as the checkpoints. A few per-epoch train
    # numbers cannot distinguish undertraining from overfitting, and at small batch sizes the
    # raw per-step curve is too noisy to read; a quarter-epoch train curve plus a per-epoch
    # validation loss separates the two. Cost: one extra forward pass over validation per epoch.
    sub_hist = sub_hist_resumed                 # [{epoch_frac, loss}] at quarter-epoch marks
    # One-epoch-at-a-time control: cap how many epochs this invocation runs, so a job under
    # a wall-clock limit can train one epoch, checkpoint and exit. --resume continues from the
    # saved epoch with model, optimizer and scheduler state intact, preserving the LR curve.
    end_epoch = epochs if max_epochs_this_run is None \
        else min(epochs, start_epoch + max(int(max_epochs_this_run), 1))
    n_sub = max(1, int(cfg.get("ckpt_sub", 4)))     # quarter-epoch checkpoints per epoch
    # WALL-CLOCK checkpoint cadence -- the robust way to bound work loss.
    # `ckpt_sub` is a count per epoch, so bounding loss to N minutes with it requires knowing
    # the epoch time T in advance and solving ckpt_sub = ceil(T / (N - val_cost)) -- which needs
    # a preflight, is wrong if T shifts (different node, different pool), and silently
    # under-provisions if the guess is low. `ckpt_minutes` needs none of that: it saves whenever
    # more than that many minutes have elapsed since the last save, so the bound holds whatever
    # T turns out to be. Both triggers are active; either one fires.
    # 0 / unset = off (previous behavior exactly).
    ck_min = float(cfg.get("ckpt_minutes", 0) or 0)
    last_ck_t = time.time()
    n_ck_time = 0                               # how many saves the clock triggered (reported)
    if ck_min > 0 and ckpt_path:
        print("    [ft] wall-clock checkpoints every %.0f min (bounds work loss regardless of "
              "epoch length)" % ck_min)
    model.train()
    for ep in range(start_epoch, end_epoch):
        # shuffle seeded by (seed, ep) so the batch order for a given epoch is the same
        # whether the run is continuous or resumed (whole-epoch or mid-epoch) -> the
        # resumed run is numerically equivalent to the single-shot run.
        _g = torch.Generator().manual_seed(int(seed) * 100003 + ep)
        order = torch.randperm(len(Xtr), generator=_g)
        # Resuming mid-epoch: carry the pre-crash partial sums so the epoch mean is over the
        # whole epoch, not just the part after the restart. Only for the epoch being resumed
        # into; every later epoch starts clean.
        if ep == start_epoch and start_sub > 0 and len(resumed_hist[3]) == 4:
            ep_sum, ep_n, sub_sum, sub_n = (float(resumed_hist[3][0]), int(resumed_hist[3][1]),
                                            float(resumed_hist[3][2]), int(resumed_hist[3][3]))
        else:
            ep_sum, ep_n = 0.0, 0
            sub_sum, sub_n = 0.0, 0
        n_micro = (len(Xtr) + bs - 1) // bs
        resume_sub = start_sub if ep == start_epoch else 0   # skip micro-batches done pre-crash
        # QUARTER-EPOCH checkpoints (only when there are enough micro-batches to matter).
        sub_ck = ckpt_path is not None and n_micro >= 2 * n_sub
        quarter = max(1, n_micro // n_sub)
        next_ck = ((resume_sub // quarter) + 1) * quarter if sub_ck else (n_micro + 1)
        opt.zero_grad()
        for mi, i in enumerate(range(0, len(Xtr), bs)):
            if mi < resume_sub:                 # already trained pre-crash -> skip (order is reproduced)
                continue
            b = order[i:i + bs]
            xb = Xtr[b].to(device)
            yb = ytr[b].to(device)
            with _autocast_ctx(amp_dtype, device):   # bf16 forward (backward outside)
                loss = loss_fn(model(xb), yb)
            (loss / accum).backward()                # scale -> accumulated grad == big-batch grad
            if bb_params and not grad_flowed:  # confirm the backbone actually learns
                g = sum(float(p.grad.abs().sum()) for p in bb_params if p.grad is not None)
                grad_flowed = g > 0
            if (mi + 1) % accum == 0 or mi == n_micro - 1:   # step at the accumulation boundary
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                # mid-epoch checkpoint at a clean (grads-zeroed) boundary: resume epoch
                # `ep` at micro-batch mi+1. not the last micro-batch (epoch-end saves below).
                # Two independent triggers, both only at this clean (grads-zeroed) boundary
                # and never on the final micro-batch (the epoch-end save covers that):
                #   count: every `quarter` micro-batches   (ckpt_sub, needs sub_ck)
                #   clock: every `ck_min` minutes          (ckpt_minutes, works at any cell size)
                _due_count = sub_ck and (mi + 1) >= next_ck
                _due_clock = (ck_min > 0 and ckpt_path is not None
                              and (time.time() - last_ck_t) >= ck_min * 60.0)
                if (_due_count or _due_clock) and (mi + 1) < n_micro:
                    _save_ckpt(ckpt_path, model, opt, sched, ep, first_loss, last_loss,
                               grad_flowed, sub_step=mi + 1,
                               hist=(epoch_hist, sub_hist, step_losses,
                                     (ep_sum, ep_n, sub_sum, sub_n)))
                    last_ck_t = time.time()
                    if _due_clock and not _due_count:
                        n_ck_time += 1
                    # advance the count cursor past here so a clock-triggered save does not
                    # leave a stale `next_ck` in the past and fire again on the next step
                    while next_ck <= mi + 1:
                        next_ck += quarter
            lv = loss.item()
            first_loss = lv if first_loss is None else first_loss
            last_loss = lv
            step_losses.append(lv); ep_sum += lv; ep_n += 1
            sub_sum += lv; sub_n += 1
            if (mi + 1) % quarter == 0 or mi == n_micro - 1:
                sub_hist.append({"epoch_frac": round(ep + (mi + 1) / float(n_micro), 4),
                                 "loss": sub_sum / max(sub_n, 1)})
                sub_sum, sub_n = 0.0, 0
        # PER-EPOCH validation loss -- the number that separates undertraining (train and val
        # both still falling) from overfitting (train falls, val rises). Same loss_fn and the
        # same bf16 autocast as training, so the two curves are directly comparable.
        _vl = None
        if len(val_idx):
            model.eval()
            with torch.no_grad():
                _vs, _vn = 0.0, 0
                for _i in range(0, len(val_idx), bs):
                    _b = val_idx[_i:_i + bs]
                    _xb = torch.as_tensor(X[_b], dtype=torch.float32, device=device)
                    _yb = torch.as_tensor(y[_b], dtype=torch.float32, device=device)
                    with _autocast_ctx(amp_dtype, device):
                        _vs += float(loss_fn(model(_xb), _yb)) * len(_b)
                    _vn += len(_b)
                _vl = _vs / max(_vn, 1)
            model.train()
        epoch_hist.append({"epoch": ep + 1, "loss": ep_sum / max(ep_n, 1), "val_loss": _vl,
                           "lr": float(opt.param_groups[0]["lr"])})
        if ckpt_path and ((ep + 1) % ckpt_every == 0 or ep == epochs - 1
                          or ep == end_epoch - 1):   # epoch boundary (always save last epoch)
            _save_ckpt(ckpt_path, model, opt, sched, ep + 1, first_loss, last_loss,
                       grad_flowed, sub_step=0,
                       hist=(epoch_hist, sub_hist, step_losses))
            last_ck_t = time.time()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print("    ft epoch %d loss %.4f" % (ep, last_loss))


    model.eval()
    pred = _predict_batched(model, X[val_idx], device, bs)
    m = regression_metrics(pred, y[val_idx])
    m.update(first_loss=first_loss, last_loss=last_loss,
             grad_flowed=(bool(grad_flowed) if bb_params else None),
             trainable_params=int(n_train), total_params=int(n_total),
             loss_history=epoch_hist, loss_history_sub=sub_hist,
             warmup_frac=_warm, lr_schedule=("warmup+cosine" if _warm > 0 else "cosine"),
             ckpt_minutes=(ck_min or None), ckpt_time_triggered=int(n_ck_time),
             completed=bool(end_epoch >= epochs),   # False if more epochs remain (resume)
             epochs_done=int(end_epoch), epochs_total=int(epochs))
    model._ft_bs = bs
    return model, m


def predict(model, X, device="cpu", batch=None):
    X = np.asarray(X, dtype=np.float32)
    bs = batch or getattr(model, "_ft_bs", 64)
    return _predict_batched(model, X, device, bs)
