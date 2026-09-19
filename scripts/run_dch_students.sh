#!/bin/bash
# Semi-supervised distillation sweep for the drift chamber: all fractions, both arms, 4 GPUs.
# The dual-readout sweep is a separate driver, scripts/run_dro_students.sh.
#
# What this runs. For each fraction f, the student gets truth labels on f% of the pool and the
# teacher trained on that same f% supervises the rest. both arms run: distilled and its
# no-teacher control.
#
# Both arms must share one --label-seed. It pins the labeled subset to the teacher's
# draw. If the control instead draws from its own run seed, the two arms train on different
# events and the seed-paired comparison between them is meaningless. Controls are cheap: they
# see f% of the data, not the full pool.
#
# The cache must be FRACTION-MATCHED. The f% student must use the cache built from the
# f% teacher. Passing the 100%-label cache to every fraction leaks label information into the
# low-fraction points, which is exactly where the measurement matters. This script refuses to
# start when a per-fraction cache is missing rather than silently falling back.
#
# training budget is FRACTION-INDEPENDENT. The student trains over the full pool at every
# fraction, so updates per epoch do not depend on the label budget and one budget works
# everywhere. epochs below is a starting point, not a validated choice: the report at the end
# flags any cell whose best epoch lands on the last epoch, which is the signature of
# undertraining. If cells are flagged, raise epochs and re-run.
#
# Usage:  bash scripts/run_dch_students.sh
#         bash scripts/run_dch_students.sh report
set -u -o pipefail
MODE="${1:-run}"

cd "$(cd "$(dirname "$0")/.." && pwd)" || exit 1
module load python 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate nexuswf 2>/dev/null || true
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
# Required before anything loads TimesFM on a parallel filesystem that does not support
# flock (the HuggingFace cache lock otherwise fails with OSError 524).
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY="$(command -v python || command -v python3)"
mkdir -p logs runs

# Point these at the waveform directories for each detector.
export DCH="${DCH:?set DCH to the drift-chamber data directory}"
export DRO="${DRO:?set DRO to the dual-readout data directory}"
CDIR="${CDIR:-${SCRATCH:-/tmp}}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7}"
# alpha is the truth/teacher mix. 0.5 weights them equally; 0 is pure teacher mimicry, with
# labels used only to select the epoch.
ALPHA="${ALPHA:-0.5}"
# The LABEL-ACCOUNTING fix. 0 = the teacher's own seed, so the student reuses the exact
# events the teacher trained on. Without it the two draws are disjoint and a point labeled f%
# was informed by ~2f% of the truth. It also makes the labeled sets nested across fractions
# (0.1% a subset of 1%, and so on) and identical across seeds.
LABEL_SEED="${LABEL_SEED:-0}"
# Output suffix. tag=ratio writes runs/dro_semi_f10_ratio.json instead of overwriting
# runs/dro_semi_f10.json, so a recipe change can be compared against the previous one rather
# than replacing it. always set tag when the recipe changes -- resume keys on (fraction, seed)
# and would otherwise treat two different recipes as one interrupted run.
TAG="${TAG:-}"
FEATW="${FEATW:-0.001}"
EPOCHS="${EPOCHS:-300}"
# Prefer A fixed step budget over an epoch count. An epoch is one pass over the pool,
# so --epochs means a different amount of training per detector and per pool size. --steps
# overrides it inside train_student_semi and gives every cell the same optimizer budget, which
# is what makes the two detectors comparable. Empty = use epochs. Applies to the --semi arm
# only; the control is sized in epochs.
STEPS="${STEPS:-}"
BATCH="${BATCH:-256}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# tag              det  frac  gpu  arm
JOBS=(
  "dch_p1      dch  0.1   0  distill"
  "dch_1       dch  1     1  distill"
  "dch_10      dch  10    2  distill"
  "dch_100     dch  100   3  distill"
  "dch_p1_lab  dch  0.1   0  labels"
  "dch_1_lab   dch  1     1  labels"
  "dch_10_lab  dch  10    2  labels"
  "dch_100_lab dch  100   3  labels"
)

if [ "$MODE" != "report" ]; then
  miss=0
  for F in 0.1 1 10 100; do
    for D in dch; do
      [ -s "$CDIR/tcache_${D}_f${F}_s0.npz" ] || { echo "MISSING cache: tcache_${D}_f${F}_s0.npz"; miss=1; }
    done
  done
  [ "$miss" = 0 ] || { echo "FATAL: build the per-fraction teacher caches first "
                            "(scripts/cache_teacher_preds.py, one per fraction)."; exit 1; }
  for f in configs/dch_ftpc.yaml; do
    [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
  done
  ngpu=$("$PY" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  [ "$ngpu" -ge 4 ] || echo "WARNING: only $ngpu GPU(s) visible; this sweep expects 4"
  [ "$ngpu" -ge 1 ] || { echo "FATAL: no CUDA devices visible"; exit 1; }
  say "preflight OK (8 caches, $ngpu GPUs, alpha=$ALPHA steps=${STEPS:-none} label_seed=$LABEL_SEED tag=${TAG:-none} only=${ONLY:-both} arm=${ARM:-both})"
fi

one() {
  local tag="$1" det="$2" frac="$3" gpu="$4" arm="$5" cfg root out cache
  # Config + student decimation, per detector.
  # The student scheme block lives inside each detector config; there is no separate
  # per-student config file.
  # --downsample sets the student input width and must match the model being reported:
  #   DCH 3008 / 5  -> 602
  #   DRO 6272 / 10 -> 640
  # It does not touch the teacher: the cache was built at full resolution and is row-aligned.
  # Detector protocol flags -- these decide whether the numbers are comparable at all.
  # DCH, from the published sweep (run_dch_little_sweep.sh:145-152 and 278-279):
  #   --sel-val-frac 0.3   hold back a disjoint selection split; the metric is quoted on the
  #                        remaining test events. without it the metric is computed over all
  #                        events, no validation metric is recorded, and the number is not
  #                        comparable to one produced with a selection split.
  #   --count-divisor 53   the count-target scaling the reported models were trained with.
  #   no --max-events.     train_distill requires the cache
  #                        row count to equal the loaded X exactly -- it does not accept a
  #                        prefix: "cached preds (500000 rows) not aligned to train X (120000
  #                        rows)". The caches are 500k, so the student loads 500k. This does not
  #                        affect comparability: sigma is measured on the separate pi/ka eval
  #                        files, and --max-events only changes how much unlabeled pool the
  #                        student gets -- more of which is the entire point of --semi.
  # DRO deliberately gets none of these -- run_dro_little_scan.sh:132: "no --count-divisor
  # (DCH-only) and no --sel-val-frac (DRO uses a random holdout)".
  local ds extra=()
  cfg=configs/dch_ftpc.yaml; root="$DCH"; ds=5
  extra=(--count-divisor 53 --sel-val-frac 0.3)
  cache="$CDIR/tcache_${det}_f${frac}_s0.npz"
  out="runs/${det}_semi_f${frac}${TAG:+_$TAG}.json"
  # The control differs only in the loss: --distill-mode labels, no --semi. Same labeled events
  # (--label-seed), same epochs, same batch. Anything else would make the gap uninterpretable.
  local armflags=(--teacher-preds "$cache" --semi --distill-mode both
                  --alpha "$ALPHA" --feat-weight "$FEATW" --feat-center mean)
  if [ "$arm" = "labels" ]; then
    armflags=(--teacher-preds "$cache" --distill-mode labels)
    out="runs/${det}_semi_f${frac}_lab${TAG:+_$TAG}.json"
  fi
  # --steps is a train_student_semi parameter; the control arm goes through fs.train_student,
  # which has no such knob, so it keeps using --epochs.
  # The control needs A matched update budget, not A fixed epoch count.
  # fs.train_student has no --steps, so the control is sized in epochs -- but its labeled set
  # varies 1000x across fractions, so any single epoch count is wrong at one end:
  #     --epochs 300 -> DCH 100% control got 527k updates (5x the distilled arm)
  #     --epochs  60 -> DRO 0.1% control got 120 updates. It scored 28.5 err68 against ~7.1
  #                     for the properly-trained control, which inflated the distilled-vs-control
  #                     gap to a meaningless +19.5.
  # So compute epochs per cell to hit the same update count the distilled arm gets (steps), with
  # a batch small enough to give >=30 batches/epoch -- the rule that fixed the low-fraction
  # collapse this morning. Both arms then see the same number of optimizer steps and the gap
  # measures the teacher, not the schedule.
  local cb ce
  if [ "$arm" = "labels" ]; then
    read -r cb ce < <("$PY" - "$det" "$frac" "${STEPS:-140000}" <<'PYEOF'
import math, sys
det, frac, target = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
# labeled pool the control can see: the pi/ka evaluation sets are separate files, so the
# whole training pool is available before the fraction is taken.
pool = 500000
n_lab = max(int(round(pool * frac / 100.0)), 32)
n_tr = max(n_lab - max(int(round(n_lab * 0.1)), 8), 8)
batch = 256
while batch > 2 and n_tr / batch < 30:      # >=30 batches/epoch or the epoch means nothing
    batch //= 2
bpe = max(1, math.ceil(n_tr / batch))
print(batch, max(1, math.ceil(target / bpe)))
PYEOF
)
    say "  control budget: batch=$cb epochs=$ce (~${STEPS:-140000} updates on $frac% labels)"
  fi

  # Budget flags are PER-ARM.
  #   distilled: --steps overrides epochs inside train_student_semi, so every cell gets the same
  #              optimizer budget regardless of pool size.
  #   control:   goes through fs.train_student, which has no --steps and reads `epochs` from the
  #              config (fc_little: 60). Forcing --epochs 300 on it made the 100% control 527k
  #              steps/seed against the distilled arm's 140k -- 5x the published protocol, which
  #              never overrode epochs at all. So the control gets neither flag and uses the
  #              config, exactly as every shipped DCH/DRO control did.
  # Save the students. Without --save-students every trained model is discarded and the
  # 100% cell cannot be compressed or synthesised later -- the deploy candidate would have to be
  # retrained from scratch. One directory per (detector, fraction, arm): the checkpoint filename
  # does not encode the recipe, so two arms sharing a directory overwrite each other silently.
  local sdir="${SDIR:-runs/students}/${det}_f${frac}_${arm}"
  mkdir -p "$sdir"
  local stepflag=() epochflag=(--epochs "$EPOCHS") bflag=(--batch-size "$BATCH")
  if [ "$arm" = "labels" ]; then
    epochflag=(--epochs "$ce")
    bflag=(--batch-size "$cb")
  elif [ -n "$STEPS" ]; then
    stepflag=(--steps "$STEPS")
  fi
  # do not skip on file existence. train_distill now flushes after every seed, so a file
  # exists as soon as seed 0 finishes. Skipping on existence would abandon a cell that is 1/8
  # done. Instead always launch it: train_distill._resume_state reads the file, skips the
  # (fraction, seed) cells already in it, and finishes the rest. A fully-done cell costs only the
  # data load.
  if [ -s "$out" ]; then
    n=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('runs',[])))" "$out" 2>/dev/null || echo 0)
    say "RESUME $tag ($n/8 seeds already in $out)"
  fi
  say "START $tag gpu=$gpu frac=$frac arm=$arm label_seed=$LABEL_SEED"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/train_distill.py \
    --config "$cfg" --root "$root" --scheme fc_little --downsample "$ds" \
    "${armflags[@]}" ${extra[@]+"${extra[@]}"} --label-seed "$LABEL_SEED" \
    --select-best-val --fractions "$frac" --seeds "$SEEDS" \
    ${epochflag[@]+"${epochflag[@]}"} "${bflag[@]}" ${stepflag[@]+"${stepflag[@]}"} \
    --save-students "$sdir" \
    --device cuda --out "$out" > "logs/semi_${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { say "FAILED $tag (rc=$rc) -- logs/semi_${tag}.log"; rm -f "$out"; return $rc; }
  say "DONE  $tag -> $out"
}

if [ "$MODE" != "report" ]; then
  for g in 0 1 2 3; do
    ( for j in "${JOBS[@]}"; do
        # shellcheck disable=SC2086
        set -- $j
        [ "$4" = "$g" ] || continue
        [ -n "${ONLY:-}" ] && [ "$2" != "$ONLY" ] && continue
        [ -n "${ARM:-}" ] && [ "$5" != "$ARM" ] && continue
        one "$1" "$2" "$3" "$4" "$5"
      done ) &
  done
  say "all lanes launched"
  wait
  say "ALL DONE"
fi

say "REPORT"
"$PY" - <<'PYEOF'
import glob, json, os, statistics as st
print("\n%-26s %6s %4s %10s %9s  %s" % ("file", "frac", "n", "metric", "SD", "best_epoch (of last)"))
print("-" * 92)
for p in sorted(glob.glob("runs/*_semi_f*.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print("%-26s UNREADABLE %s" % (os.path.basename(p), e)); continue
    rows = d.get("runs", [])
    for f in sorted({r["fraction"] for r in rows}, key=float):
        at = [r for r in rows if r["fraction"] == f]
        if "err68_holdout" in at[0]:
            v = [r["err68_holdout"]["ratio"] for r in at if "ratio" in r.get("err68_holdout", {})]
            lab = "err68 ratio"
        else:
            v = [r.get("separation") or r.get("test_sep") for r in at]
            v = [x for x in v if x is not None]
            lab = "sep sigma"
        if not v:
            continue
        be = sorted(r.get("best_epoch") for r in at if r.get("best_epoch") is not None)
        last = max((r.get("last_epoch") or 0) for r in at)
        # A budget is adequate only if the model stopped improving before it ran out. This is the
        # exact check that caught the 0.1% undertraining (16.8 -> 5.5).
        stuck = sum(1 for e in be if e >= 0.97 * last)
        flag = "  <-- %d seed(s) at the LAST epoch: Budget too short" % stuck if stuck else ""
        print("%-26s %6s %4d %10.4f %9.4f  %s%s"
              % (os.path.basename(p), f, len(v), st.median(v),
                 st.stdev(v) if len(v) > 1 else 0.0, be, flag))
print("\nDCH separation: HIGHER better. DRO err68: LOWER better.")
print("Any Budget too short flag means raise EPOCHS and re-run that cell before believing it.")
PYEOF
