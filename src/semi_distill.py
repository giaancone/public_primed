"""Semi-supervised distillation: supervise the student with the teacher on the unlabeled pool.

`student.train_student` consults the teacher only on the labeled events. At a small label
budget that is the regime where the teacher has least to add: its prediction on a labeled event
is close to the label the student already has, and the feature term is matched on the same few
events. Meanwhile the cache already holds teacher predictions and embeddings for every waveform
in the pool, labeled or not.

`train_student_semi` changes exactly one thing relative to `train_student`. The teacher-output
and teacher-feature terms are applied over `tr_idx U unl_idx` -- every event except validation --
while the truth term still uses only the `tr_idx` labels. Everything that defines the label
budget is held identical: the same seeded permutation, the same labeled events, the same
validation events (supervised by labels only, never by the teacher), the same train-only input
scaler, and the same model, initialization and projector.

Per optimizer step:

    pool batch  (size bs, from tr_idx U unl_idx)  ->  (1 - alpha) * MSE(student, teacher)
                                                      + feat_weight * ramp * feature loss
    label batch (size bs_lab, cycling tr_idx)     ->  alpha * MSE(student, label)

`alpha = 0` is pure teacher mimicry: labels leave the loss entirely, but still select the
reported epoch.

`teacher` may be any row-aligned `(N, n_targets)` array, so a cached foundation-model prediction
and a small model trained on the same labels are interchangeable here.

Pure ASCII.
"""

import numpy as np
import torch
import torch.nn as nn

from src import student as st
from src.metrics import regression_metrics


def label_budget_split(n, fraction, seed, val_fraction):
    """The same label-budget split `student.train_student` uses, plus the unlabeled rest.

    Returns (tr_idx, val_idx, unl_idx). Validation is carved from inside the labeled budget,
    so a stated fraction is the total truth consumed, not the truth plus a separate val set.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_use = max(int(round(n * fraction / 100.0)), 32)
    idx = perm[:n_use]
    n_val = max(int(round(len(idx) * val_fraction)), 8)
    return idx[n_val:], idx[:n_val], perm[n_use:]          # tr_idx, val_idx, unl_idx


def train_student_semi(X, y, teacher, cfg, seed=0, fraction=1.0, device="cpu", alpha=0.5,
                       teacher_embed=None, feat_weight=0.0, feat_warmup=0.3, epochs=60,
                       batch_size=256, batch_size_lab=None, n_unlabeled=None, steps=None,
                       lr_schedule=None, select_on="label", n_teacher_holdout=10000,
                       ft_epochs=0, ft_lr=1e-4, ft_batch=32,
                       verbose=False, label_seed=None):
    """n_unlabeled: keep only the first k unlabeled events (dose-response; 0 = teacher on the
    labeled events only). steps: if set, overrides `epochs` so every pool size gets the same
    number of optimizer steps.

    Teacher-exploitation options (all off by default):
      lr_schedule="cosine"   cosine-decay the LR to 0 over the whole run ("patient" teacher).
      select_on="teacher"    pick the best epoch by MSE(student, teacher) on `n_teacher_holdout`
                             unlabeled pool events held out of training -- needs no labels, and
                             is ~80x larger than the 120-event labeled val set.
      ft_epochs > 0          two-stage: after teacher training, fine-tune on the labeled events
                             only (lr ft_lr, batch ft_batch), best label-val epoch kept -- the
                             teacher-trained model itself is a candidate, so it can only improve
                             label-val loss."""
    n = len(X)
    teacher = np.asarray(teacher, dtype=np.float32)
    if len(teacher) != n:
        raise ValueError("teacher (%d rows) not aligned to X (%d)" % (len(teacher), n))
    # label_seed pins the labeled subset to the TEACHER's draw, so teacher and student consume
    # the same f% of the truth rather than f% each. None -> use the run seed.
    tr_idx, val_idx, unl_idx = label_budget_split(
        n, fraction, seed if label_seed is None else label_seed,
        cfg.get("val_fraction", 0.1))
    if n_unlabeled is not None:
        unl_idx = unl_idx[:int(n_unlabeled)]
    th_idx = np.zeros(0, dtype=unl_idx.dtype)
    if select_on == "teacher":
        th_idx, unl_idx = unl_idx[:int(n_teacher_holdout)], unl_idx[int(n_teacher_holdout):]
    pool_idx = np.concatenate([tr_idx, unl_idx])            # never contains a val event
    assert not np.intersect1d(pool_idx, val_idx).size
    assert not np.intersect1d(pool_idx, th_idx).size

    cfg = dict(cfg)
    if str(cfg.get("input_norm", "per_event")).lower() == "global":
        cfg["_x_scaler"] = st.fit_global_minmax(X[tr_idx], cfg.get("minmax_hi_pct"))
    Xs = st._prep_inputs(X, cfg, cfg.get("_x_scaler"))

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)   # noqa: E731
    Xpool, Tpool = t(Xs[pool_idx]), t(teacher[pool_idx])
    Xtr, ytr = t(Xs[tr_idx]), t(y[tr_idx])
    Xva, yva_np = t(Xs[val_idx]), y[val_idx]
    Xth, Tth = t(Xs[th_idx]), t(teacher[th_idx])
    fit_rows = np.random.default_rng(seed + 991).choice(len(pool_idx),
                                                         min(10000, len(pool_idx)), replace=False)
    fw, a = float(feat_weight), float(alpha)
    do_feat = teacher_embed is not None and fw > 0.0
    te_pool = None
    if do_feat:
        te_pool = np.asarray(teacher_embed[pool_idx], dtype=np.float32)
        fc = str(cfg.get("feat_center", "none")).lower()
        if fc in ("mean", "zscore", "std"):                  # stats over the pool (no val)
            te_pool = te_pool - te_pool.mean(axis=0, keepdims=True)
            if fc in ("zscore", "std"):
                te_pool = te_pool / np.maximum(te_pool.std(axis=0, keepdims=True), 1e-6)
        te_pool = t(te_pool)

    torch.manual_seed(seed)
    student = st.build_model(cfg, Xs.shape[1], y.shape[1]).to(device)
    if str(cfg.get("frontend_init", "none")).lower() == "pca":
        st._init_frontend_pca(student, Xs[tr_idx])
    st.init_output_bias(student, cfg, y[tr_idx])

    projector, cap, hook = None, {}, None
    if do_feat:
        lins = [m for m in student.modules() if isinstance(m, nn.Linear)]
        tap = cfg.get("feat_tap", "penult")
        if tap in (None, "penult", "last") or len(lins) < 2:
            hook = lins[-1].register_forward_hook(
                lambda mod, inp, out: cap.__setitem__("f", inp[0]))
        else:
            ti = 0 if tap in ("front", "wide", "frontend") else int(tap)
            hook = lins[ti].register_forward_hook(
                lambda mod, inp, out: cap.__setitem__("f", out))
        with torch.no_grad():
            student(Xtr[:2])
        projector = nn.Linear(cap["f"].shape[1], te_pool.shape[1]).to(device)

    params = list(student.parameters()) + (list(projector.parameters()) if projector else [])
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3))
    # Per-target loss weights. A single-target task wants a flat mean; a multi-target task
    # whose components live on very different scales needs cfg["target_weights"], the same
    # weighting student applies. Absent from the config -> flat mean.
    # Ratio-consistency knobs, read once; absent -> the term is inactive.
    _rw = float(cfg.get("ratio_weight", 0.0))
    _rc = cfg.get("ratio_cols")
    _tw = cfg.get("target_weights")
    if _tw is None:
        mse = lambda p, q: ((p - q) ** 2).mean()             # noqa: E731
    else:
        _w = torch.as_tensor(_tw, dtype=torch.float32, device=device).reshape(1, -1)
        mse = lambda p, q: (_w * (p - q) ** 2).mean()        # noqa: E731
    bs = int(batch_size)
    bsl = int(batch_size_lab or min(bs, len(tr_idx)))
    steps_per_epoch = (len(pool_idx) + bs - 1) // bs
    if steps is not None:
        epochs = max(1, int(np.ceil(int(steps) / steps_per_epoch)))
    total_steps = epochs * steps_per_epoch
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
             if lr_schedule == "cosine" else None)
    warmup = max(1, int(float(feat_warmup) * epochs * steps_per_epoch))
    lab_order, lab_ptr, step = torch.randperm(len(Xtr), device=device), 0, 0
    best_vl, best_state, best_ep = float("inf"), None, -1
    hist_val, hist_tot, hist_th = [], [], []

    def _eval(model, Xe, Te):
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(Xe[i:i + 8192]) for i in range(0, len(Xe), 8192)])
            v = float(mse(p, Te))
        model.train()
        return v

    yva_t = t(yva_np)
    student.train()
    for ep in range(epochs):
        order = torch.randperm(len(Xpool), device=device)
        ep_tot, nb = 0.0, 0
        for i in range(0, len(Xpool), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            pred = student(Xpool[b])
            loss = (1.0 - a) * mse(pred, Tpool[b])
            if do_feat:
                zs = projector(cap["f"])
                zt = te_pool[b]
                fl = ((st._l2norm(zs) - st._l2norm(zt)) ** 2).sum(dim=-1).mean()
                loss = loss + fw * min(1.0, step / warmup) * fl
            # Draw a label batch when either the truth term (alpha) or the ratio-consistency
            # term is active. Guarding on alpha alone would silently drop ratio-consistency at
            # alpha=0, the one setting where no label batch is otherwise needed.
            if a > 0.0 or (_rw > 0.0 and _rc is not None):
                if lab_ptr + bsl > len(lab_order):
                    lab_order, lab_ptr = torch.randperm(len(Xtr), device=device), 0
                lb = lab_order[lab_ptr:lab_ptr + bsl]
                lab_ptr += bsl
                _pl = student(Xtr[lb])
                if a > 0.0:
                    loss = loss + a * mse(_pl, ytr[lb])
                # Ratio-consistency, enabled by cfg ratio_weight/ratio_cols; a no-op when the
                # config omits them. it must match the control arm. student applies
                # this term too, so omitting it here would leave the distilled arm as the only
                # arm not optimizing the ratio, and the measured difference between arms would
                # confound the teacher with a missing loss term.
                # Label batch only: a teacher batch has no true ratio to be consistent with.
                # Same stable cross-product form as student:1126-1131 -- (pc*ts - ps*tc) is
                # zero exactly when the c and s relative errors match (and so cancel in R = c/s),
                # normalized by the batch-mean tc*ts so a near-zero true c or s cannot blow up.
                if _rw > 0.0 and _rc is not None:
                    _ci, _si = int(_rc[0]), int(_rc[1])
                    _tc, _ts = ytr[lb][:, _ci], ytr[lb][:, _si]
                    _pc, _ps = _pl[:, _ci], _pl[:, _si]
                    _m = (_tc > 1e-6) & (_ts > 1e-6)
                    if _m.any():
                        _tc, _ts, _pc, _ps = _tc[_m], _ts[_m], _pc[_m], _ps[_m]
                        _cross = _pc * _ts - _ps * _tc
                        loss = loss + _rw * ((_cross ** 2).mean() / ((_tc * _ts).mean() + 1e-6))
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            ep_tot += float(loss.detach()); nb += 1
        hist_tot.append(ep_tot / max(nb, 1))
        vl = _eval(student, Xva, yva_t)
        hist_val.append(vl)
        crit = vl
        if select_on == "teacher":
            crit = _eval(student, Xth, Tth)
            hist_th.append(crit)
        if crit < best_vl:
            best_vl, best_ep = crit, ep
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("  semi epoch %d loss %.6g val %.6g sel %.6g" % (ep, hist_tot[-1], vl, crit))

    if hook is not None:
        hook.remove()
    if best_state is not None:
        student.load_state_dict(best_state)

    # diagnostics of How well the student fits the teacher (capacity vs generalisation)
    fit_train = _eval(student, Xpool[t(fit_rows).long()], Tpool[t(fit_rows).long()])
    fit_th = _eval(student, Xth, Tth) if len(th_idx) else None

    ft_info = {}
    if ft_epochs and ft_epochs > 0:
        opt2 = torch.optim.Adam(student.parameters(), lr=float(ft_lr))
        best2 = _eval(student, Xva, yva_t)
        state2, ep2 = {k: v.detach().clone() for k, v in student.state_dict().items()}, -1
        for e2 in range(int(ft_epochs)):
            o = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(Xtr), int(ft_batch)):
                b = o[i:i + int(ft_batch)]
                opt2.zero_grad()
                mse(student(Xtr[b]), ytr[b]).backward()
                opt2.step()
            v2 = _eval(student, Xva, yva_t)
            if v2 < best2:
                best2, ep2 = v2, e2
                state2 = {k: v.detach().clone() for k, v in student.state_dict().items()}
        student.load_state_dict(state2)
        ft_info = {"ft_best_epoch": ep2, "ft_best_val_loss": best2}

    student.eval()
    with torch.no_grad():
        pv = student(Xva).cpu().numpy()
    m = regression_metrics(pv, yva_np)
    m.update({"loss_history": hist_tot, "val_loss_history": hist_val,
              "teacher_holdout_history": hist_th, "select_on": select_on,
              "best_val_loss": best_vl, "best_epoch": best_ep, "last_epoch": epochs - 1,
              "select_best_val": True, "n_labeled_train": int(len(tr_idx)),
              "n_val": int(len(val_idx)), "n_pool": int(len(pool_idx)),
              "n_unlabeled": int(len(unl_idx)), "n_teacher_holdout": int(len(th_idx)),
              "epochs_run": int(epochs), "steps": int(step), "lr_schedule": lr_schedule,
              "fit_mse_teacher_train": fit_train,
              "fit_mse_teacher_holdout": fit_th,
              "input_norm": str(cfg.get("input_norm", "per_event")), **ft_info})
    student._x_cfg = dict(cfg)
    student._x_scaler = cfg.get("_x_scaler")
    return student, m
