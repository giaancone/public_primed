"""Distill a saved teacher into the small student.

Loads a cached teacher (predictions and, optionally, pooled embeddings) and trains a student
whose loss mixes the true label with the teacher's output:

    alpha * MSE(student, label) + (1 - alpha) * MSE(student, teacher)
                                + feat_weight * feature term

Runs the same label-fraction x seed scan and reports the same metric as the from-scratch
baseline, so distilled and control students are directly comparable:

  * DCH -> kaon/pion separation power, which needs both eval sets.
  * DRO -> err68 on c, s and the C/S ratio, on a fixed holdout or an external slice.

The student architecture and the loss mix come from the config's scheme block, overridable on
the command line. No foundation model is loaded here -- the teacher arrives as a cache.

Pure ASCII.
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data_loader                 # noqa: E402
from src import student as st           # noqa: E402
from src import separation as sep_mod       # noqa: E402
from src import dro_metric                  # noqa: E402


def _save_student(args, scfg, student, length, n_targets, frac, seed, dataset,
                  target_names=None):
    """Persist a trained student so it can be compressed later. No-op without --save-students.

    layout matters. The export step reads `model` / `scfg` / `input_len` /
    `n_targets`, not the `state_dict` /
    `arch_cfg` / `length` layout of student.save_model. Writing the wrong one fails at the
    next step, an hour later, rather than here.

    `scfg["_x_scaler"]` must travel with the weights. The student is trained under global
    MinMax fitted on its own train split; predicting it later under per-event z-scoring gives
    meaningless numbers silently (student.py:556). export_student_npz copies it into the
    npz meta, and pt_to_keras carries it into the .h5 pipeline.
    """
    d = getattr(args, "save_students", "")
    if not d:
        return None
    import torch
    os.makedirs(d, exist_ok=True)
    # The scaler does not come back in scfg. student.py:850 does `cfg = dict(cfg)` before
    # inserting `_x_scaler`, so the fit lands on a local copy and the caller's scfg never sees
    # it. (That copy is deliberate -- mutating the caller would leak seed 0's scaler into every
    # later seed via the `is None` guard.) The fitted scaler is attached to the model object at
    # student.py:1037, and `state_dict()` drops plain attributes, so read it off the model
    # here. Without this the .pt is unusable for compression: export_student_npz has no scaler
    # to copy into the npz meta and the compressed model is scored under the wrong input
    # normalization, silently.
    scfg = {k: v for k, v in scfg.items()}
    if scfg.get("_x_scaler") is None:
        sc = getattr(student, "_x_scaler", None)
        if sc is not None:
            scfg["_x_scaler"] = {"lo": float(sc["lo"]), "hi": float(sc["hi"])}
    if str(scfg.get("input_norm", "per_event")).lower() == "global" \
            and scfg.get("_x_scaler") is None:
        raise SystemExit("input_norm=global but no _x_scaler to save -- refusing to write a "
                         "student that cannot be compressed (scripts/train_distill.py)")

    hid = "-".join(str(int(h)) for h in scfg.get("hidden", []))
    path = os.path.join(d, "%s_%s_h%s_f%g_s%d.pt"
                        % (dataset, args.scheme, hid, float(frac), int(seed)))
    torch.save({"model": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                "scfg": {k: v for k, v in scfg.items()},
                "input_len": int(length), "n_targets": int(n_targets),
                "target_names": list(target_names or []),
                "student_downsample": int(args.downsample or 1),
                "count_divisor": float(getattr(args, "count_divisor", 0.0) or 0.0),
                "teacher": os.path.basename(args.teacher_preds or args.teacher or ""),
                "distill_mode": args.distill_mode, "alpha": float(args.alpha),
                "feat_weight": float(args.feat_weight),
                "fraction": float(frac), "seed": int(seed)}, path)
    print("[sd] saved student -> %s" % path)
    return path


def check_aligned(y_loaded, cache_targets, atol=1e-4):
    """Row-alignment guard for CROSS-RESOLUTION distillation: the cache was built on
    a different config (full 6272) than the student trains on (downsampled 640). Same
    events/order + identical divisor-normalized targets -> y must match row-for-row.
    Returns (ok, msg). ok=False means the two loads are misaligned (different events,
    order, max-events, or target scaling) and distillation would pair wrong events."""
    if cache_targets is None:
        return True, "no cached targets (legacy cache) -- alignment not verified"
    a = np.asarray(y_loaded, dtype=np.float32)
    b = np.asarray(cache_targets, dtype=np.float32)
    if a.shape != b.shape:
        return False, "target shape %s != cache %s" % (a.shape, b.shape)
    if not np.allclose(a, b, atol=atol):
        nbad = int((np.abs(a - b) > atol).any(axis=1).sum())
        return False, "%d/%d rows differ (max |dy|=%.4g) -- MISALIGNED loads" % (
            nbad, len(a), float(np.abs(a - b).max()))
    return True, "verified row-aligned to cache (%d rows)" % len(a)


# Retry seeds are offset by this, not by +1. Retrying at `seed + 1` is safe only when
# one seed is trained at a time. With a seed list of 0..7 it collides with a seed already being
# trained: cell (frac, seed=2) would
# retry as seed 3 and silently duplicate cell (frac, seed=3): the "8 independent seeds" become 7,
# the duplicate pair is perfectly correlated, and the seed-paired comparison against the other
# arm quietly breaks. A stride larger than any seed list we will ever use avoids that entirely.
RETRY_SEED_STRIDE = 1000


def _resume_state(args, tag):
    """(done_cells, prior_rows) for --out, so a killed run restarts at the seed it died on.

    Why this exists. Results used to be written only after every (fraction, seed) cell
    finished, so an allocation that expired mid-cell threw away all 8 seeds -- hours of GPU time
    for nothing, repeatedly. Now each seed is appended and flushed as it completes: the most any
    interruption can cost is one seed. Mirrors the pattern already proven in
    scripts/train_teacher_dch.py:143-154.

    Resume is keyed on (fraction, seed), so re-running the same command picks up exactly where it
    stopped and re-running a different recipe into the same --out would silently mix two
    experiments -- write a new --out when the recipe changes.
    """
    if not args.out or not os.path.exists(args.out):
        return set(), []
    try:
        prev = json.load(open(args.out))
    except Exception as e:                      # truncated by a kill: start clean, say so
        print("[sd] %s: could not read %s (%s) -- starting fresh" % (tag, args.out, e))
        return set(), []
    rows = prev.get("runs", [])
    done = {(float(r["fraction"]), int(r["seed"])) for r in rows
            if r.get("fraction") is not None and r.get("seed") is not None}
    if done:
        print("[sd] %s RESUME: %d cell(s) already in %s -- skipping them"
              % (tag, len(done), args.out))
    return done, rows


def _flush(args, payload):
    """Write --out now. Temp file + atomic rename, so a kill mid-write cannot leave a
    half-written JSON that still parses and silently truncates the results."""
    if not args.out:
        return
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, args.out)


def _trainer(args):
    """`fs` normally; a _SemiTrainer when --semi is set. Cheap to build per cell (it only
    imports a module), so no global state and no way for one arm to silently inherit the
    other\'s trainer."""
    return _SemiTrainer(args) if getattr(args, "semi", False) else fs


class _SemiTrainer(object):
    """Duck-types `fs` so --semi routes to semi_distill.train_student_semi with no change to
    _train_with_retries or either call site.

    What --semi changes, and what it must not. The label budget is identical: the same
    seeded permutation, the same n_use labeled events, the same validation events (labels only).
    The one difference is that the teacher-output and teacher-feature terms run over
    `tr_idx U unl_idx` -- every event except validation -- instead of `tr_idx` alone. Without it
    the teacher is only ever consulted about events whose labels the student already has, which
    is why every distillation arm we have measured ~0 at low fractions.

    Two silent traps this exists to close.
      1. train_student reads `epochs`/`batch_size` from cfg; train_student_semi takes them as
         arguments defaulting to 60/256. Forwarding the cfg values explicitly is what stops a
         900-epoch recipe from quietly running 60.
      2. train_student_semi has no `rkd_weight` or `plot_path`. Silently dropping a nonzero
         rkd_weight would change the loss without saying so, so that is a hard error.
    """

    def __init__(self, args):
        from src import semi_distill                    # noqa: E402  (import only when used)
        self._semi = semi_distill
        self._a = args

    def train_student(self, X, y, teacher, cfg, seed=0, fraction=100.0, **kw):
        a = self._a
        if float(kw.pop("rkd_weight", 0.0) or 0.0) > 0.0:
            raise SystemExit("--semi does not implement --rkd-weight; drop one or the other")
        kw.pop("plot_path", None)                       # no per-epoch loss plot on this path
        # cfg carries the recipe; semi takes them positionally -- see trap 1 above.
        kw.setdefault("epochs", int(cfg.get("epochs", 60)))
        kw.setdefault("batch_size", int(cfg.get("batch_size", 256)))
        for name, attr in (("label_seed", "label_seed"),
                           ("select_on", "select_on"), ("n_teacher_holdout", "n_teacher_holdout"),
                           ("n_unlabeled", "n_unlabeled"), ("steps", "steps"),
                           ("lr_schedule", "lr_schedule"),                            ("batch_size_lab", "batch_size_lab")):
            v = getattr(a, attr, None)
            if v is not None:
                kw[name] = v
        return self._semi.train_student_semi(X, y, teacher, cfg, seed=seed,
                                             fraction=fraction, **kw)

    # _train_with_retries only ever touches .train_student; anything else falls through to the
    # real module so this stays a drop-in.
    def __getattr__(self, name):
        return getattr(fs, name)


def _train_with_retries(fs, args, X, y, teacher, scfg, seed, fraction, **kw):
    """train_student, plus an optional convergence gate applied identically to every arm.

    the rule: a run whose best validation loss never falls below a threshold did not converge.
    It is RE-RUN from a different initialization -- not deleted, and not trained longer.
    Selection is on validation and never touches the holdout.

    Why RE-RUN rather than drop. A non-converged cell is usually a run the recipe
    failed rather than an inherently bad seed: the same initialization can train perfectly well
    under a different batch size. Deleting the cell discards a model that a different roll
    produces fine, and biases any statistic computed over the survivors. Re-running keeps the
    seed count intact at the cost of one extra fit.

    The gate must be the same on both arms. The distilled arm and its no-teacher control
    are compared seed by seed. If preventing collapse is part of what the teacher does, gating
    only one arm hands that mechanism to one side and corrupts the comparison. This helper is
    used by both loops and driven by the same flag; there is deliberately no per-arm switch.

    The threshold is TASK-SPECIFIC and must be passed explicitly. It is a raw loss
    value, so it depends on the target normalization; a threshold that separates converged from
    collapsed runs on one task means nothing on another. Derive it from the observed bimodality
    of validation losses for the task at hand. Absent --fail-val-loss this function is a
    pass-through.

    What it cannot catch. The gate reads best_val_loss, so it only sees failures validation can
    see. DRO 10% seed 4 has best_val_loss 0.00082 -- squarely inside the healthy band -- and a
    test ratio of 6.45, failing on the s channel alone. That is a generalisation gap, not a
    training failure, and no validation-only rule may remove it. Expect the gate to reduce
    collapses, not eliminate them.

    Records `retry_attempts`, `retry_seeds` and `retry_val_losses` on the returned metrics so the
    retry count is reportable -- under a gate the interesting quantity stops being "did it
    collapse" and becomes "how many restarts did this arm need", which is a compute-cost claim
    that survives the protocol instead of being erased by it.
    """
    gate = getattr(args, "fail_val_loss", None)
    if gate is None:
        return st.train_student(X, y, teacher, scfg, seed=seed, fraction=fraction, **kw)

    if not scfg.get("select_best_val"):
        # best_val_loss is only recorded under --select-best-val (student.py:1163). Without
        # it the gate has nothing to read and would silently accept every run.
        raise SystemExit("--fail-val-loss requires --select-best-val (no best_val_loss recorded)")

    max_attempts = int(getattr(args, "max_retries", 4)) + 1
    tried, losses = [], []
    # Track the best attempt with a separate numeric score rather than comparing against the
    # stored metrics dict. A missing best_val_loss comes back as None, and `0.01 < None` is a
    # TypeError on Python 3 -- so an attempt that failed to record a loss would crash the whole
    # run on the next attempt's comparison. None scores as +inf: never better than a real number,
    # always replaceable, and it still survives as the fallback if every attempt is None.
    best_student, best_m, best_score = None, None, float("inf")
    for attempt in range(max_attempts):
        s = seed + RETRY_SEED_STRIDE * attempt
        student, m = st.train_student(X, y, teacher, scfg, seed=s, fraction=fraction, **kw)
        vl = m.get("best_val_loss")
        tried.append(s)
        losses.append(vl)
        score = vl if vl is not None else float("inf")
        if best_student is None or score < best_score:
            best_student, best_m, best_score = student, m, score
        if vl is not None and vl < gate:
            if attempt:
                print("[retry] frac=%s seed=%s CONVERGED on attempt %d (seed %d), val_loss %.5g"
                      % (fraction, seed, attempt + 1, s, vl))
            # A passing attempt always has vl < gate <= every failing attempt, so the minimum
            # is this attempt -- returning `best` here cannot silently hand back a failure.
            student, m = best_student, best_m
            break
        print("[retry] frac=%s seed=%s attempt %d (seed %d) FAILED the gate: val_loss %s >= %.5g"
              % (fraction, seed, attempt + 1, s, ("%.5g" % vl) if vl is not None else "None", gate))
    else:
        # Exhausted. Return the best attempt and say so -- a cell that never converged must not
        # be indistinguishable from one that converged on the first try.
        student, m = best_student, best_m
        print("[retry] frac=%s seed=%s EXHAUSTED %d attempts, none passed the gate; keeping the "
              "best (val_loss %s). This cell is still a FAILURE and should be screened."
              % (fraction, seed, max_attempts,
                 ("%.5g" % best_score) if best_score != float("inf") else "None"))
        m["retry_exhausted"] = True

    m["retry_attempts"] = len(tried)
    m["retry_seeds"] = tried
    m["retry_val_losses"] = losses
    m["retry_gate"] = gate
    return student, m


def _plot_path(args, dataset, frac, seed):
    """Per-cell student loss-curve png path (or None if no out/plot dir)."""
    if args.out is None and args.plot_dir is None:
        return None
    pdir = args.plot_dir or os.path.join(os.path.dirname(args.out) or ".", "plots")
    os.makedirs(pdir, exist_ok=True)
    return os.path.join(pdir, "%s_convmlp_%s_f%s_s%s_loss.png"
                        % (dataset, args.distill_mode or "distill", frac, seed))


def repool_regions(emb, K):
    """Re-pool a cached regional embedding (N, K0, D) to K regions (E1). K must divide
    K0; K=1 collapses to the global mean (N, D). A 2D cache (global) is returned as-is."""
    if emb.ndim == 2 or K is None:
        return emb
    N, K0, D = emb.shape
    if K == K0:
        return emb
    if K == 1:
        return emb.mean(axis=1)
    if K0 % K != 0:
        raise SystemExit("[sd] --regions %d must divide the cached %d regions" % (K, K0))
    return emb.reshape(N, K, K0 // K, D).mean(axis=2)


def resolve_arm(mode, alpha, feat_weight):
    """Map a distill-mode name to (alpha, feat_weight).
      labels  -> (1.0, 0)          label only (the floor / control)
      output  -> (alpha, 0)        label + teacher answers (predicted ~= label -> ~no-op)
      feature -> (1.0, feat_weight) label + teacher embedding (the treatment)
      both    -> (alpha, feat_weight) label + answers + embedding
      None    -> (alpha, feat_weight) legacy path, unchanged."""
    if mode == "labels":
        return 1.0, 0.0
    if mode == "output":
        return float(alpha), 0.0
    if mode == "feature":
        return 1.0, float(feat_weight)
    return float(alpha), float(feat_weight)     # 'both' and legacy(None)


def split_pika(X, sel_val_frac):
    """Split the pion and kaon evaluation sets into a selection and a test half.

    This is a comparability requirement, not a nicety. When a seed or an epoch is
    chosen using the evaluation sets, the reported number must come from events that played no
    part in that choice. This holds back a disjoint fraction for selection and scores on the
    rest. Scoring on all events instead computes a different number on a superset that includes
    the selection events, and the two are not comparable.

    The rng seed 20260728 and the `max(..., 1)` floor are copied verbatim. Each set gets its
    own permutation of its own length -- pion and kaon are split independently with the same
    seed, fixed across runs. Changing the seed silently reshuffles which events are held
    out and makes every new number incomparable to every old one.
    """
    r = np.random.default_rng(20260728).permutation(len(X))
    n = max(int(round(len(X) * float(sel_val_frac))), 1)
    return X[r[:n]], X[r[n:]]


def run_dch(args, config, scfg, teacher, X, y):
    length_scale = config["eval"]["length_scale"]
    pion = kaon = None
    truth_separation = None
    svf = float(getattr(args, "sel_val_frac", 0.0) or 0.0)
    try:
        Xp, tp, _ = data_loader.load_dch_concat(config, args.root, eval_set="pion")
        Xk, tk, _ = data_loader.load_dch_concat(config, args.root, eval_set="kaon")
        pion, kaon = (Xp, tp), (Xk, tk)
        # Ceiling stays on the full sets: it is a property of the data, not of a model, and
        # quoting it on a subset would make it drift with sel_val_frac for no reason.
        ts = sep_mod.separation_power(tp.ravel(), tk.ravel(), length_scale)
        truth_separation = ts["separation"]
        print("[sd] TRUTH-count separation (ceiling): %.3f sigma" % truth_separation)
        if svf > 0:
            Xp_v, Xp_t = split_pika(Xp, svf)
            Xk_v, Xk_t = split_pika(Xk, svf)
            pion, kaon = (Xp_t, tp), (Xk_t, tk)          # `separation` is the test split
            args._pika_val = (Xp_v, Xk_v)
            print("[sd] pi/ka split (rng 20260728, sel_val_frac=%.2f): "
                  "val=%d/%d  test=%d/%d (disjoint)"
                  % (svf, len(Xp_v), len(Xk_v), len(Xp_t), len(Xk_t)))
        else:
            args._pika_val = None
            print("[sd] pi/ka NOT split -- sigma is on ALL events. This is NOT comparable to "
                  "the published DCH student numbers; pass --sel-val-frac 0.3 for those.")
    except FileNotFoundError:
        print("[sd] no pion/kaon eval sets found -- MAE/R^2 only.")

    _done, results = _resume_state(args, "dch")
    for frac in scfg["fractions"]:
        for seed in scfg["seeds"]:
            if (float(frac), int(seed)) in _done:
                continue
            student, m = _train_with_retries(
                _trainer(args), args, X, y, teacher, scfg, seed=seed, fraction=frac,
                device=args.device, alpha=args.alpha,
                teacher_embed=args.teacher_embed,
                feat_weight=args.feat_weight,
                feat_warmup=args.feat_warmup, verbose=args.verbose,
                label_seed=args.label_seed,
                rkd_weight=args.rkd_weight,
                plot_path=_plot_path(args, "dch", frac, seed))
            row = {"fraction": frac, "seed": seed, **m}
            _save_student(args, scfg, student, X.shape[1], y.shape[1], frac, seed,
                          dataset="dch")
            if pion is not None:
                def _sep(Xp_, Xk_):
                    # no cfg/scaler passed on purpose: train_student stamps _x_cfg/_x_scaler
                    # onto the model (student.py:1037) and predict() falls back to them,
                    # so the global-MinMax transform travels with the student. Passing scfg
                    # here would also work but would re-read a dict whose _x_scaler the
                    # caller never received (student.py:850 rebinds a local copy).
                    a_ = st.predict(student, Xp_, device=args.device).ravel()
                    b_ = st.predict(student, Xk_, device=args.device).ravel()
                    return float(sep_mod.separation_power(a_, b_, length_scale)["separation"])
                row["separation"] = _sep(pion[0], kaon[0])
                if getattr(args, "_pika_val", None) is not None:
                    row["test_sep"] = row["separation"]      # keep both key names
                    row["val_sep"] = _sep(args._pika_val[0], args._pika_val[1])
            # The measured loss split -- what the (alpha, feat_weight) sweep is actually
            # sweeping. Carried per row so an arm can be described by the balance it achieved
            # rather than by the raw weight that produced it (student.train_student).
            for _k in ("loss_shares_final", "feat_share_final",
                       "best_epoch", "best_val_loss", "last_epoch"):
                if _k in m:
                    row[_k] = m[_k]
            results.append(row)
            # Flush every seed. An interruption now costs one seed, not the whole cell.
            _flush(args, {"runs": results, "truth_separation": truth_separation,
                          "length_scale": length_scale, "alpha": args.alpha})
            extra = ("  sep=%.3f" % row["separation"]) if "separation" in row else ""
            if "val_sep" in row:
                extra += " (val %.3f)" % row["val_sep"]
            if "feat_share_final" in row:
                extra += "  feat_share=%.1f%%" % (100 * row["feat_share_final"])
            print("[sd] frac=%5s seed=%d  mae=%.3f r2=%.3f%s"
                  % (frac, seed, m["mae"], m["r2"], extra))

    print("\n[sd] mean over seeds:")
    for frac in scfg["fractions"]:
        rs = [r for r in results if r["fraction"] == frac]
        line = "  frac=%5s  mae=%.3f  r2=%.3f" % (
            frac, np.mean([r["mae"] for r in rs]), np.mean([r["r2"] for r in rs]))
        if all("separation" in r for r in rs):
            line += "  sep=%.3f sigma" % np.mean([r["separation"] for r in rs])
        print(line)
    return {"runs": results, "truth_separation": truth_separation,
            "length_scale": length_scale, "alpha": args.alpha}


def run_dro(args, config, scfg, teacher, X, y):
    tnames = list(config["data"]["target_names"])
    divisors = config["data"].get("target_divisors")
    bench = dro_metric.BENCHMARK
    order = [n for n in tnames if n != "t0"]
    if "c" in tnames and "s" in tnames:
        order.append("ratio")
    if "t0" in tnames:
        order.append("t0")

    hf = float(config["eval"].get("holdout_fraction", 0.2))
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(X))
    nh = max(int(round(len(X) * hf)), 8)
    hidx, pidx = perm[:nh], perm[nh:]
    Xh, yh = X[hidx], y[hidx]
    X, y = X[pidx], y[pidx]
    # a preds-array teacher is aligned to the full X -> slice it by the same pidx
    # so it stays row-aligned with the post-holdout train pool. The teacher embedding
    # (feature distill) is aligned the same way.
    if isinstance(teacher, np.ndarray):
        teacher = teacher[pidx]
    te = args.teacher_embed[pidx] if args.teacher_embed is not None else None
    print("[sd] DRO eval on FIXED %d-event holdout, disjoint from %d-event train pool"
          % (nh, len(pidx)))

    _done, results = _resume_state(args, "dro")
    for frac in scfg["fractions"]:
        for seed in scfg["seeds"]:
            if (float(frac), int(seed)) in _done:
                continue
            student, m = _train_with_retries(
                _trainer(args), args, X, y, teacher, scfg, seed=seed, fraction=frac,
                device=args.device, alpha=args.alpha,
                teacher_embed=te, feat_weight=args.feat_weight,
                feat_warmup=args.feat_warmup, verbose=args.verbose,
                label_seed=args.label_seed,
                rkd_weight=args.rkd_weight,
                plot_path=_plot_path(args, "dro", frac, seed))
            row = {"fraction": frac, "seed": seed, **m}
            _save_student(args, scfg, student, X.shape[1], y.shape[1], frac, seed,
                          dataset="dro", target_names=tnames)
            pe = st.predict(student, Xh, device=args.device)
            row["err68_holdout"] = dro_metric.dro_metrics(pe, yh, tnames, divisors)
            results.append(row)
            # Flush every seed -- see run_dch.
            _flush(args, {"dataset": "dro", "target_names": tnames, "eval_sets": ["holdout"],
                          "benchmark": bench, "runs": results, "alpha": args.alpha})
            comp = " ".join(dro_metric.fmt(k, row["err68_holdout"][k])
                            for k in order if k in row["err68_holdout"])
            print("[sd] frac=%5s seed=%d  mae=%.3f r2=%.3f  err68[%s]"
                  % (frac, seed, m["mae"], m["r2"], comp))

    print("\n[sd] mean over seeds (err68 on holdout):")
    for frac in scfg["fractions"]:
        rs = [r for r in results if r["fraction"] == frac]
        line = "  frac=%5s  mae=%.3f  r2=%.3f  " % (
            frac, np.mean([r["mae"] for r in rs]), np.mean([r["r2"] for r in rs]))
        line += " ".join(dro_metric.fmt(k, float(np.mean([r["err68_holdout"][k] for r in rs])))
                          for k in order)
        print(line)
    return {"dataset": "dro", "target_names": tnames, "eval_sets": ["holdout"],
            "benchmark": bench, "runs": results, "alpha": args.alpha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True, help="data root")
    ap.add_argument("--teacher", default=None, help="st/fs model .pt (st.load_model); "
                    "predicted live")
    ap.add_argument("--teacher-preds", default=None, help="cached teacher soft targets "
                    ".npz (from cache_teacher_preds.py) -- teacher-agnostic (e.g. ft)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=None,
                    help="true-label vs teacher weight in [0,1] (default: distill block, else 0.5)")
    ap.add_argument("--distill-mode", default=None,
                    choices=["labels", "output", "feature", "both"],
                    help="distillation arm. Sets alpha + "
                         "feat_weight: labels=(1,0) output=(alpha,0) feature=(1,fw) "
                         "both=(alpha,fw). Overrides --alpha. feature/both need cached "
                         "embeds (--teacher-preds).")
    ap.add_argument("--feat-weight", type=float, default=1.0,
                    help="feature-distill (embedding) loss weight for feature/both modes")
    ap.add_argument("--feat-warmup", type=float, default=0.3,
                    help="fraction of steps to ramp the feature-distill term")
    ap.add_argument("--rkd-weight", type=float, default=0.0,
                    help="relational-distill (RKD) weight: match the teacher embedding's "
                         "pairwise geometry across the batch (E3). 0 = off.")
    ap.add_argument("--regions", type=int, default=None,
                    help="multi-region feature target (E1): re-pool the cached (N,K0,D) "
                         "embeds to K regions (K must divide K0). K=1 = global mean.")
    ap.add_argument("--plot-dir", default=None,
                    help="write a student loss-curve PNG per (fraction,seed) here "
                         "(default: <out-dir>/plots)")
    ap.add_argument("--lr", type=float, default=None, help="override the scheme LR (HP scan)")
    # Convergence gate, applied identically to every arm. A run whose best validation loss
    # never falls below this did not converge and is re-run from a different init. Off by
    # default. Derive the value per task from the observed bimodality of validation losses --
    # it is a raw loss and does not transfer across target normalizations.
    ap.add_argument("--fail-val-loss", type=float, default=None,
                    help="convergence gate: retry any cell whose best_val_loss >= this "
                         "(requires --select-best-val; applied to ALL arms equally)")
    ap.add_argument("--max-retries", type=int, default=4,
                    help="attempts after the first before giving up (default 4)")
    ap.add_argument("--epochs", type=int, default=None, help="override scheme epochs (HP scan)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the scheme batch_size. With epochs FIXED, this changes the "
                         "number of weight updates -- which is the whole point at small "
                         "fractions (batch 256 gives ~3 updates/epoch at 1%% of 120k rows, "
                         "batch 128 gives ~7). Default None = use the config block.")
    ap.add_argument("--prune-target", type=float, default=None,
                    help="SPARSE-AWARE training (GMP): final sparsity 0-1 ramped during "
                         "training (e.g. 0.8 -> ~7k effective). 0/omit = dense.")
    ap.add_argument("--prune-start", type=float, default=None, help="GMP: epoch-frac to begin (0.2)")
    ap.add_argument("--prune-end", type=float, default=None, help="GMP: epoch-frac to reach target (0.8)")
    ap.add_argument("--feat-center", default="none", choices=["none", "mean", "zscore"],
                    help="Center the teacher embeddings before matching them. Raw embeddings "
                         "can be nearly identical across events, in which case the feature "
                         "loss is effectively 'output this one constant vector': it is learned "
                         "immediately and then never trains, while still consuming loss weight. "
                         "Centering makes the target the deviation from the typical event, which "
                         "is where the event-to-event information lives -- the teacher's own "
                         "head reads this same pooled vector. 'mean' subtracts the per-dim "
                         "mean; 'zscore' also divides by the per-dim sd. Statistics are taken "
                         "from the train split only. Default none leaves behavior unchanged.")
    ap.add_argument("--select-best-val", action="store_true",
                    help="Keep the BEST-val-loss epoch instead of the last one, checkpointing "
                         "per epoch. Matters most at low label fractions, where 60 epochs is only "
                         "~300 steps and the last epoch is not reliably the best. OFF by "
                         "default: enabling it changes the returned model, so every arm it is "
                         "used on must be compared against a control that also uses it.")
    ap.add_argument("--sel-val-frac", type=float, default=0.0,
                    help="DCH ONLY. Hold back this fraction of the pion/kaon sets as a "
                         "disjoint selection split (rng 20260728). Set this whenever a seed "
                         "or epoch is chosen on the eval sets, so the reported number "
                         "comes from events that took no part in the choice. Default 0.0 "
                         "scores over ALL events, which is a different, incomparable number.")
    ap.add_argument("--fractions", default=None, help="comma-sep fraction override (e.g. 100)")
    ap.add_argument("--seeds", default=None, help="comma-sep seed override (e.g. 0)")
    ap.add_argument("--save-students", default="",
                    help="directory to persist each trained student as a .pt. WITHOUT THIS "
                         "The weights are discarded -- train_distill evaluates and drops the "
                         "model, so a run gives metrics you cannot compress. Written in the "
                         "exact layout the export step expects (model / scfg / "
                         "input_len / n_targets), so the deployment chain runs unchanged.")
    ap.add_argument("--count-divisor", type=float, default=0.0,
                    help="DCH ONLY. Divide the cluster-count TARGET *and* the cached teacher "
                         "predictions by this. Use 53 TO match the deployed student. "
                         "The count is 5-53 raw; the shipped student trained on count/53, "
                         "because a small-scale output is easier to reach from a standard "
                         "init than a large-scale one. It is also REQUIRED for 12-bit QAT: "
                         "quantized_bits(12,2) spans only +/-4, so a raw-count model returns "
                         "NaN. Separation power is a ratio in "
                         "matched units, so the divisor CANCELS in the metric -- it is an "
                         "optimization aid, not a change to what is reported.")
    # ---- semi-supervised distillation: teacher supervision over the unlabeled pool.
    ap.add_argument("--label-seed", type=int, default=None,
                    help="pin the LABELED subset to this seed (the teacher's, normally 0) while "
                         "--seeds still varies initialization. Without it the student draws a "
                         "DIFFERENT f%% than the teacher trained on, so the chain consumes 2f%% of "
                         "the truth behind an f%% claim. MUST be set identically on the distilled "
                         "arm and its control, or the paired comparison uses different data.")
    ap.add_argument("--semi", action="store_true",
                    help="apply the teacher over tr_idx U unlabeled pool instead of tr_idx only. "
                         "The label budget is unchanged; only the teacher's coverage changes. "
                         "REQUIRES a --teacher-preds cache built from the teacher trained at THIS "
                         "fraction, or the run leaks label information from a stronger teacher.")
    ap.add_argument("--select-on", default=None, choices=["label", "teacher"],
                    help="--semi: pick the best epoch by label validation (default) or by "
                         "teacher agreement on held-out UNLABELED events (no labels used).")
    ap.add_argument("--n-teacher-holdout", type=int, default=None,
                    help="--semi --select-on teacher: unlabeled events reserved for selection "
                         "and excluded from training.")
    ap.add_argument("--n-unlabeled", type=int, default=None,
                    help="--semi: cap the unlabeled pool (dose-response; 0 = teacher on the "
                         "labeled events only, i.e. the old behavior).")
    ap.add_argument("--steps", type=int, default=None,
                    help="--semi: fixed optimizer steps, overriding --epochs, so different pool "
                         "sizes get an equal budget.")
    ap.add_argument("--lr-schedule", default=None, choices=["cosine"])
    ap.add_argument("--batch-size-lab", type=int, default=None,
                    help="--semi: label-batch size (defaults to --batch-size).")
    ap.add_argument("--downsample", type=int, default=None,
                    help="override STUDENT data.downsample (input-resolution sweep). The "
                         "teacher cache is full-res regardless; alignment is by event so "
                         "the same cache serves any student resolution.")
    ap.add_argument("--context-len", type=int, default=None,
                    help="override STUDENT data.context_len to match --downsample")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-epoch student loss (passed to train_student)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scheme", default="distill",
                    help="config block for the STUDENT arch (default 'distill'; "
                         "e.g. 'distill_big' for a slightly larger student)")
    args = ap.parse_args()
    if bool(args.teacher) == bool(args.teacher_preds):
        raise SystemExit("[sd] pass exactly one of --teacher / --teacher-preds")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # student input-resolution override (teacher cache is full-res + event-aligned).
    if args.downsample is not None:
        # DRO reads data.downsample in _dro_prep_wave. DCH has no such key -- its student
        # decimation goes through data.student_downsample (src/data_loader._dch_window ->
        # student_prep.downsample). Route the same flag to whichever the dataset uses, so
        # `--downsample 5` means the same thing on both detectors.
        #
        # Why not put it in the config's data block: configs/dch_ftpc.yaml is shared with the
        # teacher (ftpc) runs, and the teacher must see the full 3008-sample waveform. A
        # student-only key living in `data` would silently downsample the teacher too. Setting
        # it here means it applies only when a student run explicitly asks for it.
        if str(config.get("dataset", "")).lower() == "dch":
            config["data"]["student_downsample"] = int(args.downsample)
            config["data"].setdefault("student_pool", "stride")   # deployed student used stride
        else:
            config["data"]["downsample"] = int(args.downsample)
    if args.context_len is not None:
        config["data"]["context_len"] = int(args.context_len)
    if args.downsample is not None or args.context_len is not None:
        print("[sd] STUDENT resolution override: downsample=%s student_downsample=%s "
              "context_len=%s" % (config["data"].get("downsample"),
                                  config["data"].get("student_downsample"),
                                  config["data"].get("context_len")))
    scfg = dict(config.get(args.scheme) or config.get("distill") or config.get("fs", config["probe"]))
    if args.lr is not None:
        scfg["lr"] = float(args.lr)                 # HP-scan overrides (E4)
    if args.epochs is not None:
        scfg["epochs"] = int(args.epochs)
    # Batch size override.
    #
    # With epochs fixed, batch size is A STEP-BUDGET knob. Doubling the batch halves the
    # number of weight updates, and at small label fractions there are few training rows to
    # begin with -- a handful of updates per epoch. Two runs of the same model that differ only
    # in batch size can therefore differ enormously at low fractions and agree at full data,
    # which looks like a modeling effect but is undertraining.
    #
    # It is a flag rather than a config edit so that one run can be compared against an existing
    # one by changing exactly this; editing the scheme block would change every future run of it.
    if args.batch_size is not None:
        scfg["batch_size"] = int(args.batch_size)
    if args.fractions:
        scfg["fractions"] = [float(x) if "." in x else int(x) for x in args.fractions.split(",")]
    if args.seeds:
        scfg["seeds"] = [int(x) for x in args.seeds.split(",")]
    if args.prune_target is not None:
        scfg["prune_target"] = float(args.prune_target)     # sparse-aware training (GMP)
    if args.prune_start is not None:
        scfg["prune_start"] = float(args.prune_start)
    if args.prune_end is not None:
        scfg["prune_end"] = float(args.prune_end)
    if args.select_best_val:
        scfg["select_best_val"] = True
    if args.feat_center and args.feat_center != "none":
        scfg["feat_center"] = args.feat_center
    dataset = config.get("dataset", "dch")
    if args.alpha is None:
        args.alpha = float(scfg.get("alpha", 0.5))

    # 4-arm experiment: distill-mode sets (alpha, feat_weight) explicitly.
    args.teacher_embed = None
    args.cache_targets = None
    args.alpha, args.feat_weight = resolve_arm(args.distill_mode, args.alpha, args.feat_weight)

    if args.teacher:                                   # live model teacher (st/fs)
        teacher, ck = st.load_model(args.teacher, device=args.device)
        n_teacher = sum(p.numel() for p in teacher.parameters())
        exp_len, exp_ntgt, tsrc = ck["length"], ck["n_targets"], ck["arch_cfg"].get("arch")
    else:                                              # cached preds teacher (any, incl ft)
        z = np.load(args.teacher_preds, allow_pickle=True)
        teacher = np.asarray(z["preds"], dtype=np.float32)
        n_teacher = int(z["teacher_params"]) if "teacher_params" in z else -1
        exp_len, exp_ntgt, tsrc = None, int(z["n_targets"]), "%s-preds" % z.get("scheme", "?")
        if "embeds" in z:
            args.teacher_embed = repool_regions(np.asarray(z["embeds"], dtype=np.float32),
                                                args.regions)
        args.cache_targets = np.asarray(z["targets"], dtype=np.float32) if "targets" in z else None

    # feature/both need a cached embedding; fail loudly if it is missing.
    if args.distill_mode in ("feature", "both") and args.teacher_embed is None:
        raise SystemExit("[sd] --distill-mode %s needs cached teacher embeds; re-run "
                         "cache_teacher_preds.py (it now saves 'embeds') and pass "
                         "--teacher-preds" % args.distill_mode)
    ed = None if args.teacher_embed is None else args.teacher_embed.shape[1]
    print("[sd] teacher=%s (%s params), student arch=%s  mode=%s alpha=%.2f feat_weight=%.2f embed_dim=%s"
          % (tsrc, n_teacher, scfg.get("arch", "cnn"), args.distill_mode or "(legacy)",
             args.alpha, args.feat_weight, ed))

    print("[sd] loading raw train waveforms ...")
    X, y, _ = data_loader.load_concat(config, args.root, split="train",
                                      max_files=args.max_files, max_events=args.max_events)
    print("[sd] dataset=%s  train: X=%s y=%s" % (dataset, X.shape, y.shape))
    if y.shape[1] != exp_ntgt or (exp_len is not None and X.shape[1] != exp_len):
        raise SystemExit("[sd] data (len=%d n_tgt=%d) does not match teacher "
                         "(len=%s n_tgt=%d)" % (X.shape[1], y.shape[1], exp_len, exp_ntgt))
    if isinstance(teacher, np.ndarray) and len(teacher) != len(X):
        raise SystemExit("[sd] cached preds (%d rows) not aligned to train X (%d rows) "
                         "-- use the SAME --config/--root/--max-files as the cache"
                         % (len(teacher), len(X)))
    if args.teacher_embed is not None and len(args.teacher_embed) != len(X):
        raise SystemExit("[sd] cached embeds (%d rows) not aligned to train X (%d rows) "
                         "-- use the SAME --config/--root/--max-files as the cache"
                         % (len(args.teacher_embed), len(X)))
    ok, msg = check_aligned(y, args.cache_targets)
    print("[sd] cross-resolution alignment: %s" % msg)
    if not ok:
        raise SystemExit("[sd] ALIGNMENT FAILED (%s). The cache and the student loaded "
                         "DIFFERENT events/order -- feature distill would pair wrong "
                         "events. Ensure cache + train use the SAME glob/root/max-events "
                         "(only downsample/context_len may differ)." % msg)
    # target normalization (DCH). Applied after check_aligned, which compares y against the
    # cache targets in raw units -- dividing first would fail that check for the wrong reason.
    # Must hit the teacher predictions too: the distillation loss compares the student output
    # against both the labels and the teacher preds, so scaling only one puts the two terms on
    # different scales and silently reweights them.
    _cdiv = float(getattr(args, "count_divisor", 0.0) or 0.0)
    if _cdiv > 0:
        if dataset != "dch":
            raise SystemExit("[sd] --count-divisor is DCH-only (dataset=%s)" % dataset)
        y = (np.asarray(y, dtype=np.float32) / _cdiv).astype(np.float32)
        if isinstance(teacher, np.ndarray):
            teacher = (teacher / _cdiv).astype(np.float32)
        _mx = float(max(np.abs(y).max(),
                        np.abs(teacher).max() if isinstance(teacher, np.ndarray) else 0.0))
        print("[sd] count_divisor=%g -> target max %.4f (teacher preds scaled too)"
              % (_cdiv, _mx))
        if _mx >= 4.0:
            raise SystemExit("[sd] target max %.4f >= 4.0 after /%g -- outside "
                             "quantized_bits(12,2) (+/-4), which is the precision the "
                             "deployed student is compressed at. Raise --count-divisor."
                             % (_mx, _cdiv))
    n_student = sum(p.numel() for p in
                    st.build_model(scfg, X.shape[1], y.shape[1]).parameters())
    ratio = ("%.0fx smaller than teacher" % (n_teacher / max(n_student, 1))
             if n_teacher > 0 else "teacher size unknown from cache")
    print("[sd] student: %d params (%s)" % (n_student, ratio))

    if dataset == "dro":
        out = run_dro(args, config, scfg, teacher, X, y)
    else:
        out = run_dch(args, config, scfg, teacher, X, y)
    out["student_params"] = n_student
    out["teacher_params"] = n_teacher
    out["distill_mode"] = args.distill_mode
    out["feat_weight"] = args.feat_weight
    out["feat_warmup"] = args.feat_warmup
    out["feat_center"] = args.feat_center
    # Record the protocol, not just the hyperparameters. A sigma is meaningless without
    # knowing which pi/ka events it was measured on, and the count divisor changes what
    # `feat_weight` means (1/53^2 is balanced, 1.0 is feature-dominated at the same divisor).
    # Both were previously reconstructable only from the launching shell script.
    out["sel_val_frac"] = float(getattr(args, "sel_val_frac", 0.0) or 0.0)
    out["count_divisor"] = (float(args.count_divisor) if args.count_divisor else None)
    out["alpha"] = args.alpha
    out["scheme"] = args.scheme
    out["max_events"] = args.max_events
    out["epochs"] = scfg.get("epochs")
    out["batch_size"] = scfg.get("batch_size")
    out["lr"] = scfg.get("lr")
    out["teacher_preds"] = args.teacher_preds
    out["student_downsample"] = args.downsample

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("[sd] wrote %s" % args.out)


if __name__ == "__main__":
    main()
