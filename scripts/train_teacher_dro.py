"""Fine-tune the foundation model on the dual-readout C/S/t0 regression.

Three targets are regressed jointly under a weighted MSE (the config `ft` block's `target_weights`), so the
low-signal timing target does not dilute the energy fit. Scoring is err68 on c, s and the
C/S ratio plus a t0 timing resolution in ns -- see src/dro_metric.py.

Evaluation uses a fixed random holdout carved out before the fraction scan and never trained
on, so no cell in the grid can have seen an evaluation event. `--external-eval` scores an
external per-species slice instead.

The backbone, head and training loop are shared with the drift-chamber runner
(src/teacher.py); only the target count, the loss weighting and the metric differ.

Runs the full fraction x seed grid. `--fractions` / `--seeds` subset it, `--resume` skips
completed pairs, and the JSON is rewritten after each cell. `--ckpt-dir` saves per-epoch
weights and resumes an interrupted fine-tune. `--stub` runs the whole path with no GPU.

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
from src import teacher as tch           # noqa: E402
from src import dro_metric                  # noqa: E402


def _num(s):
    """Parse a scan value keeping int-ness (100 stays 100, not 100.0)."""
    f = float(s)
    return int(f) if f == int(f) else f


def _write(path, runs, dataset, tnames, eval_names, bench):
    """Atomic-ish rewrite of the DRO results JSON (schema == runs/dro_fx.json)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    runs = sorted(runs, key=lambda r: (r["fraction"], r["seed"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"dataset": dataset, "target_names": tnames,
                   "eval_sets": eval_names, "benchmark": bench, "runs": runs},
                  f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True, help="DRO data root")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="cuda strongly recommended (fine-tunes 200M)")
    ap.add_argument("--stub", action="store_true", help="tiny torch backbone (no TimesFM/GPU) for testing")
    ap.add_argument("--fractions", default=None,
                    help="comma-sep override of the scan fractions (for chunking)")
    ap.add_argument("--seeds", default=None, help="comma-sep override of the scan seeds")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing --out, skipping (fraction,seed) pairs already in it")
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir to save per-run weight checkpoints; a killed fine-tune RESUMES "
                         "from its checkpoint. Pair with --resume.")
    ap.add_argument("--external-eval", action="store_true",
                    help="score err68 on the config's electron/kaon eval globs instead of a "
                         "fixed holdout (only safe if those files are disjoint from train_glob).")
    ap.add_argument("--plot-dir", default=None,
                    help="dir for per-cell training-loss PNGs (default: alongside --out, "
                         "else --ckpt-dir, under plots/)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs-per-run", type=int, default=None,
                    help="train at most N epochs per invocation, then checkpoint + exit "
                         "(one epoch per job; --resume continues). Requires "
                         "--ckpt-dir. Default: all epochs in one go.")
    args = ap.parse_args()
    if args.epochs_per_run is not None and not args.ckpt_dir:
        ap.error("--epochs-per-run requires --ckpt-dir (progress is carried by the "
                 "per-epoch checkpoint; without it every run would restart from epoch 0)")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if config.get("dataset") != "dro":
        raise SystemExit("train_teacher_dro.py is DRO-only (use train_teacher_dch.py for DCH).")
    fcfg = dict(config["ft"])
    if args.fractions:
        fcfg["fractions"] = [_num(x) for x in args.fractions.split(",")]
    if args.seeds:
        fcfg["seeds"] = [int(x) for x in args.seeds.split(",")]
    tnames = list(config["data"]["target_names"])
    divisors = config["data"].get("target_divisors")
    bench = dro_metric.BENCHMARK
    input_len = data_loader.input_length(config)

    # display order: energy err68 (c, s), then C/S ratio, then t0 resolution last.
    order = [n for n in tnames if n != "t0"]
    if "c" in tnames and "s" in tnames:
        order.append("ratio")
    if "t0" in tnames:
        order.append("t0")

    print("[ft-dro] loading raw train waveforms ...")
    X, y, _ = data_loader.load_dro_concat(config, args.root, split="train",
                                          max_files=args.max_files, max_events=args.max_events)
    print("[ft-dro] train: X=%s y=%s  (stub=%s device=%s)" % (X.shape, y.shape, args.stub, args.device))
    print("[ft-dro] scan: fractions=%s seeds=%s" % (fcfg["fractions"], fcfg["seeds"]))

    # Eval sets: external disjoint slice(s), or the fixed leak-free holdout.
    eval_sets = []                              # (name, X_eval, y_eval)
    if args.external_eval:
        for name in ("electron", "kaon"):
            try:
                Xe, te, _ = data_loader.load_dro_concat(config, args.root, eval_set=name)
                eval_sets.append((name, Xe, te))
            except (FileNotFoundError, KeyError):
                pass
        print("[ft-dro] eval on external slice(s): %s (ensure DISJOINT from train_glob!)"
              % ", ".join(s[0] for s in eval_sets))
        pool_X, pool_y = X, y
    if not eval_sets:
        hf = float(config["eval"].get("holdout_fraction", 0.2))
        rng = np.random.default_rng(12345)      # fixed seed -> same holdout every run
        perm = rng.permutation(len(X))
        nh = max(int(round(len(X) * hf)), 8)
        hidx, pidx = perm[:nh], perm[nh:]
        pool_X, pool_y = X[pidx], y[pidx]
        eval_sets.append(("holdout", X[hidx], y[hidx]))
        print("[ft-dro] eval on FIXED %d-event holdout (%.0f%%), disjoint from the "
              "%d-event train pool" % (nh, hf * 100, len(pidx)))
        # `X[pidx]` and `X[hidx]` are fancy-index copies, so the original full array is dead
        # weight from here on -- but it stays bound for all 12 (fraction, seed) cells. At the
        # two-species pool (200,000 x 6272 float32) that is 5.0 GB of host RAM held for nothing,
        # on top of the 4.0 GB pool + 1.0 GB holdout copies. Drop it.
        del X, y
    eval_names = [s[0] for s in eval_sets]
    print("[ft-dro] err68 benchmark (BSO 10GeV 312.5MHz, lower=better): "
          "C=%.2f%% S=%.2f%% C/S=%.2f%%" % (bench["c"], bench["s"], bench["ratio"]))

    # Resume: carry over already-completed (fraction,seed) runs and skip them.
    results = []
    done = set()
    if args.resume and args.out and os.path.exists(args.out):
        prev = json.load(open(args.out))
        results = list(prev.get("runs", []))
        done = {(r["fraction"], r["seed"]) for r in results}
        print("[ft-dro] resume: %d run(s) already in %s -- skipping those" % (len(results), args.out))

    # loss-curve PNGs land here by default (every run) -- next to --out, else --ckpt-dir
    plot_dir = args.plot_dir or (os.path.join(os.path.dirname(args.out) or ".", "plots")
                                 if args.out else (os.path.join(args.ckpt_dir, "plots")
                                                   if args.ckpt_dir else "plots"))
    for frac in fcfg["fractions"]:
        for seed in fcfg["seeds"]:
            if (frac, seed) in done:
                print("[ft-dro] skip (already done): frac=%s seed=%d" % (frac, seed))
                continue
            ckpt_path = None
            if args.ckpt_dir:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt_dir, "ftdro_f%s_s%s.pt" % (frac, seed))
            plot_path = os.path.join(plot_dir, "%s_f%s_s%s_loss.png"
                                     % (config.get("name", "dro_ft"), frac, seed))
            model, m = tch.train_ft(pool_X, pool_y, fcfg, seed=seed, fraction=frac,
                                   device=args.device, stub=args.stub, input_len=input_len,
                                   verbose=args.verbose, ckpt_path=ckpt_path,
                                   ckpt_every=fcfg.get("ckpt_every", 1), plot_path=plot_path,
                                   max_epochs_this_run=args.epochs_per_run)
            if not m.get("completed", True):
                # one-epoch-at-a-time: this seed has more epochs to go. The .pt checkpoint
                # holds the state; do not write the result JSON or mark (frac,seed) done --
                # else --resume would skip this seed on the next invocation. Skip the (costly)
                # holdout eval until the seed is actually finished; just re-run to continue.
                print("[ft-dro] frac=%5s seed=%d  epoch %s/%s done  loss %.3f->%.3f  "
                      "-- checkpoint saved (%s); re-run to continue"
                      % (frac, seed, m.get("epochs_done"), m.get("epochs_total"),
                         m["first_loss"], m["last_loss"], ckpt_path))
                continue
            row = {"fraction": frac, "seed": seed, "mae": m["mae"], "r2": m["r2"],
                   "trainable_params": m.get("trainable_params"),
                   "grad_flowed": m.get("grad_flowed"),
                   # Persist the loss curves in the results JSON so they survive independently
                   # of any plotting step. loss_history is per epoch (train, val and lr);
                   # loss_history_sub is the per-quarter-epoch train curve, which resolves the
                   # early transient that a per-epoch curve smooths away.
                   "loss_history": m.get("loss_history"),
                   "loss_history_sub": m.get("loss_history_sub"),
                   "lr_schedule": m.get("lr_schedule"),
                   "warmup_frac": m.get("warmup_frac"),
                   "first_loss": m.get("first_loss"), "last_loss": m.get("last_loss")}
            for name, Xe, te in eval_sets:
                pe = tch.predict(model, Xe, device=args.device)
                row["err68_%s" % name] = dro_metric.dro_metrics(pe, te, tnames, divisors)
            results.append(row)
            done.add((frac, seed))
            if args.out:                       # incremental write -- survive a GPU timeout
                _write(args.out, results, "dro", tnames, eval_names, bench)
            hd = row["err68_%s" % eval_names[0]]
            comp = " ".join(dro_metric.fmt(k, hd[k]) for k in order if k in hd)
            gf = "" if m.get("grad_flowed") is None else (" grad_flowed=%s" % m["grad_flowed"])
            print("[ft-dro] frac=%5s seed=%d  mae=%.3f r2=%.3f  loss %.3f->%.3f  "
                  "trainable=%d%s  err68[%s: %s]"
                  % (frac, seed, m["mae"], m["r2"], m["first_loss"], m["last_loss"],
                     m["trainable_params"], gf, eval_names[0], comp))

    print("\n[ft-dro] mean over seeds (err68 on %s, lower=better):" % eval_names[0])
    key = "err68_%s" % eval_names[0]
    for frac in sorted({r["fraction"] for r in results}):
        rs = [r for r in results if r["fraction"] == frac]
        agg = {}
        for k in order:
            vals = [r[key][k] for r in rs if k in r[key]]
            if vals:
                agg[k] = float(np.mean(vals))
        line = "  frac=%5s  mae=%.3f  r2=%.3f  (seeds=%d)  " % (
            frac, np.mean([r["mae"] for r in rs]), np.mean([r["r2"] for r in rs]), len(rs))
        line += " ".join(dro_metric.fmt(k, agg[k]) for k in order if k in agg)
        print(line)
    print("[ft-dro] references: Compressed ML C=%.2f%% S=%.2f%% C/S=%.2f%% "
          "(lower is better; t0 has no benchmark -- reported as ns resolution)"
          % (bench["c"], bench["s"], bench["ratio"]))
    if args.out:
        _write(args.out, results, "dro", tnames, eval_names, bench)
        print("[ft-dro] wrote %s (%d runs total)" % (args.out, len(results)))


if __name__ == "__main__":
    main()
