"""Fine-tune the foundation model on the drift-chamber cluster-counting task.

The model carries two heads: a per-sample head classifying every waveform sample as noise,
primary ionization or secondary ionization, and a pooled head regressing the primary-cluster
count. Separation power is computed from the pooled count; the per-sample head contributes
training signal only.

Runs the full fraction x seed grid and writes one JSON row per (fraction, seed). `--resume`
skips rows already present, and the JSON is rewritten after every cell, so an interrupted run
loses at most one cell. `--ckpt-dir` additionally saves per-epoch weights.

  python scripts/train_teacher_dch.py --config configs/dch_ftpc.yaml --root $DCH \\
      --device cuda --seeds 0 --out runs/dch_ftpc_s0.json

  # No GPU and no data: fabricates a fixture matching the real file schema.
  python scripts/train_teacher_dch.py --config configs/dch_ftpc.yaml --stub --out /tmp/ftpc.json

Pure ASCII.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data_loader                 # noqa: E402
from src import teacher_dch as tdch         # noqa: E402
from src import separation as sep_mod       # noqa: E402


def _fabricate_npz(path, config, n, seed, mean_k, mom, wide=3000):
    """One fabricated DCH-schema .npz with primary + nearby secondary peaks, so
    --stub exercises the real loader/3-class-target/metric path with no real data."""
    d = config["data"]
    rng = np.random.default_rng(seed)
    P = 300
    t = np.arange(wide)
    wf = np.zeros((n, wide), np.float32)
    tt = np.full((n, P), wide + 10, np.int64)
    tv = np.zeros((n, P), np.int64)
    grid = np.arange(5, wide - 5)
    for i in range(n):
        k = max(0, min(int(rng.poisson(mean_k)), 120))
        pos = np.sort(rng.choice(grid, size=k, replace=False)) if k > 0 else np.array([], int)
        j = 0
        for c in pos:
            wf[i] += rng.uniform(1.5, 2.5) * np.exp(-0.5 * ((t - c) / 2.0) ** 2)
            tt[i, j] = int(c); tv[i, j] = 1; j += 1
            if rng.random() < 0.5 and j < P:                 # a secondary just after
                q = min(wide - 1, c + rng.integers(1, 4))
                wf[i] += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((t - q) / 2.0) ** 2)
                tt[i, j] = int(q); tv[i, j] = 2; j += 1
        wf[i] += 0.02 * rng.standard_normal(wide).astype(np.float32)
    np.savez(path, **{d["wf_key"]: wf, d["tag_times_key"]: tt,
                      d["tag_values_key"]: tv, d["mom_key"]: np.full(n, mom, np.float32)})


def _make_stub_root(config, seed=0):
    root = tempfile.mkdtemp(prefix="ftpc_stub_")
    d = config["data"]; ev = config["eval"]; mom = float(ev["momentum"])
    for glob, nk, sd in [(d["train_glob"], 10.0, seed),
                         (ev["pion_glob"], 11.0, seed + 1),
                         (ev["kaon_glob"], 8.0, seed + 2)]:
        sub = os.path.join(root, os.path.dirname(glob))
        os.makedirs(sub, exist_ok=True)
        _fabricate_npz(os.path.join(sub, "stub0.npz"), config, 200, sd, nk, mom, wide=256)
    print("[ftpc] --stub: fabricated DCH-schema fixture under %s" % root)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--fractions", default=None, help="override, e.g. 0.1,1,10,100")
    ap.add_argument("--seeds", default=None, help="override, e.g. 0,1,2")
    ap.add_argument("--batch-size", type=int, default=None, help="override ftpc.batch_size")
    ap.add_argument("--epochs", type=int, default=None, help="override ftpc.epochs")
    ap.add_argument("--save-weights", default=None)
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir for per-(fraction,seed) checkpoints; doubles as saved "
                         "weights and enables mid-fine-tune resume")
    ap.add_argument("--resume", action="store_true",
                    help="skip (fraction,seed) cells already present in --out")
    ap.add_argument("--plot-dir", default=None,
                    help="dir for per-cell training-loss PNGs (default: alongside "
                         "--out or --ckpt-dir under plots/)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    fcfg = config["ftpc"]
    canonical_frac = max(fcfg["fractions"])   # global max, before any --fractions override
    if args.stub:
        fcfg["stub"] = True
        args.root = _make_stub_root(config)
    if args.fractions:
        fcfg["fractions"] = [float(x) if "." in x else int(x) for x in args.fractions.split(",")]
    if args.seeds:
        fcfg["seeds"] = [int(x) for x in args.seeds.split(",")]
    if args.batch_size:
        fcfg["batch_size"] = args.batch_size
    if args.epochs:
        fcfg["epochs"] = args.epochs
    length_scale = config["eval"]["length_scale"]

    print("[ftpc] loading FULL waveforms + 3-class per-sample targets ...")
    X, ps, _, cnt = data_loader.load_dch_peakcount_concat(
        config, args.root, split="train",
        max_files=args.max_files, max_events=args.max_events)
    cnt = cnt.reshape(-1)
    L = X.shape[1]
    print("[ftpc] train: X=%s  ps=%s  mean primaries/event=%.2f  positive rate=%.4f"
          % (X.shape, ps.shape, cnt.mean(), (ps > 0).mean()))

    # eval_max_events (config eval section) caps the kaon/pion load for the separation
    # eval. Default None = load all (unchanged behavior). Separation power is intrinsic,
    # so a capped sample gives a statistically-consistent number far faster.
    eval_max = config["eval"].get("eval_max_events")
    Xp, _, _, cp = data_loader.load_dch_peakcount_concat(config, args.root, eval_set="pion", max_events=eval_max)
    Xk, _, _, ck = data_loader.load_dch_peakcount_concat(config, args.root, eval_set="kaon", max_events=eval_max)
    truth = sep_mod.separation_power(cp.ravel(), ck.ravel(), length_scale)["separation"]
    print("[ftpc] TRUTH-count separation (ceiling): %.3f sigma (1 m track, L=%.3f)"
          % (truth, length_scale))

    ds = config.get("dataset", "dch")
    ck_dir = args.ckpt_dir or args.save_weights
    # loss-curve PNGs land here by default (every run) -- next to --out, else --ckpt-dir
    plot_dir = args.plot_dir or (os.path.join(os.path.dirname(args.out) or ".", "plots")
                                 if args.out else (os.path.join(ck_dir, "plots") if ck_dir else "plots"))
    out = {"scheme": "ftpc", "dataset": ds, "length_scale": length_scale,
           "truth_separation": truth, "runs": []}

    # resume: reload prior results and skip completed (fraction, seed) cells
    done = set()
    if args.resume and args.out and os.path.exists(args.out):
        prev = json.load(open(args.out))
        out["runs"] = prev.get("runs", [])
        done = {(float(r["fraction"]), int(r["seed"])) for r in out["runs"]}
        print("[ftpc] resume: %d cell(s) already done -> skipping" % len(done))

    def _write():
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=2)

    for frac in fcfg["fractions"]:
        for seed in fcfg["seeds"]:
            if (float(frac), int(seed)) in done:
                print("[ftpc] skip done  frac=%5s seed=%d" % (frac, seed))
                continue
            ckpt = None
            if ck_dir:
                os.makedirs(ck_dir, exist_ok=True)
                ckpt = os.path.join(ck_dir, "%s_ftpc_f%s_s%d.pt" % (ds, frac, seed))
            plot_path = os.path.join(plot_dir, "%s_ftpc_f%s_s%d_loss.png" % (ds, frac, seed))
            model, m = tdch.train_peakcount(X, ps, cnt, fcfg, seed=seed, fraction=frac,
                                           device=args.device, stub=fcfg.get("stub", False),
                                           input_len=L, verbose=args.verbose,
                                           ckpt_path=ckpt, ckpt_every=int(fcfg.get("ckpt_every", 1)),
                                           plot_path=plot_path)
            # canonical teacher = fixed seed 0 at max fraction (not min(seeds): in a
            # sharded per-seed launch every process has seeds=[s] so min==s, which
            # would race all seeds onto the shared teacher file -- gate on seed 0).
            if ck_dir and frac == canonical_frac and seed == 0:   # global max, not the shard's
                tdch.save_peakcount(os.path.join(ck_dir, "%s_ftpc_teacher.pt" % ds), model, fcfg, L)
            # one batched forward per eval set (count head + peak-map count together)
            ch_p, pk_p = tdch.eval_counts(model, Xp, device=args.device)
            ch_k, pk_k = tdch.eval_counts(model, Xk, device=args.device)
            sep_head = sep_mod.separation_power(ch_p, ch_k, length_scale)["separation"]
            sep_peak = sep_mod.separation_power(pk_p, pk_k, length_scale)["separation"]
            row = {"fraction": frac, "seed": seed, "separation": sep_head,
                   "separation_peakmap": sep_peak, **m}
            out["runs"].append(row)
            _write()                        # persist after every cell (crash-safe)
            print("[ftpc] frac=%5s seed=%d  recall=%.3f pred_rate=%.4f count_mae=%.3f "
                  "sep(head)=%.3f sep(peaks)=%.3f"
                  % (frac, seed, m["primary_recall"], m["pred_primary_rate"],
                     m["count_mae_head"], sep_head, sep_peak))
    results = out["runs"]

    print("\n[ftpc] mean over seeds:")
    for frac in fcfg["fractions"]:
        rs = [r for r in results if r["fraction"] == frac]
        print("  frac=%5s  recall=%.3f  sep(head)=%.3f  sep(peaks)=%.3f sigma"
              % (frac, np.mean([r["primary_recall"] for r in rs]),
                 np.mean([r["separation"] for r in rs]),
                 np.mean([r["separation_peakmap"] for r in rs])))
    print("[ftpc] references (1 m track): truth ceiling=%.3f | st 5.49 (OLD 2m footing)"
          % truth)
    if args.out:
        print("[ftpc] wrote", args.out)


if __name__ == "__main__":
    main()
