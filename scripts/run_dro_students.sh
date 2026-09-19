#!/bin/bash
# DRO student sweep: both species, both arms, 8 seeds, 4 fractions.
#
# What this runs. For each fraction f: the student gets truth on f% of the pool and the
# f%-teacher labels the rest (distilled arm), plus an otherwise-identical from-scratch control
# that sees only the same f% of truth. 4 fractions x 2 arms = 8 cells, 8 seeds each.
#
# Why the control budget is computed, not hardcoded.
# The distilled arm takes --steps 140000 directly (train_student_semi accepts steps). The
# control goes through fs.train_student, which has no step knob -- it takes batch and epochs and
# multiplies them by whatever the data gives. Its labeled set varies 1000x across fractions, so
# any single epoch count is badly wrong at one end. Worse, the correct numbers depend on the pool
# size: the same batch and epoch count give twice the updates on a pool twice as large. Reusing
# an epoch count computed for a different pool hands one arm a budget advantage and makes the
# distilled-vs-control difference meaningless. Both arms get the same 140,000 optimizer
# steps, so the difference between them measures the teacher and not the schedule.
#
# Prerequisites -- both are hard failures if missed.
#   1. Teacher caches, one per fraction, from the FRACTION-MATCHED seed-1 teacher:
#        bash scripts/run_dro_students.sh cache
#      The f1 student must use the f1 teacher. Passing the 100%-label cache to every fraction is
#      Passing the 100%-label cache to every fraction leaks label information into the
#      low-fraction points.
#   2. --label-seed matching the seed of the teacher the caches were built from. A teacher's
#      labeled subset comes from its run seed, so different teacher seeds trained on different
#      events. A mismatched label-seed trains the student on events its teacher never saw, which
#      reopens the ~2f% label double-count.
#
# Usage:
#   bash scripts/run_dro_students.sh cache     # step 1: build the 4 teacher caches
#   bash scripts/run_dro_students.sh budget    # print the computed control budgets, no run
#   bash scripts/run_dro_students.sh           # step 2: the sweep (4 GPUs)
#   bash scripts/run_dro_students.sh report    # summarize whatever exists
#
# Run it in the foreground inside tmux or screen. Backgrounding returns the shell to its
# prompt, and a login-timeout on a batch system will then release the allocation and kill the
# whole sweep.
set -u -o pipefail
MODE="${1:-run}"

module load python 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate nexuswf 2>/dev/null || true
command -v python >/dev/null || { echo "FATAL: no python on PATH"; exit 1; }
python -c "import torch" 2>/dev/null || { echo "FATAL: wrong env (no torch)"; exit 1; }

cd "$(cd "$(dirname "$0")/.." && pwd)" || exit 1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp}/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DRO="${DRO:?set DRO to the dual-readout data directory}"
mkdir -p logs runs

CDIR="${CDIR:-${SCRATCH:-/tmp}}"
TEACHER_DIR="${TEACHER_DIR:?set TEACHER_DIR to the directory holding the teacher checkpoints}"
TSEED="${TSEED:-1}"                 # teacher seed, selected by validation MAE
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7}"   # STUDENT seeds
STEPS="${STEPS:-140000}"
POOL="${POOL:-160000}"              # 200,000 events - 40,000 holdout
CTX="${CTX:-628}"                   # deployed student width; 640 gives 10,891 not 10,699
ALPHA="${ALPHA:-0.5}"
FEATW="${FEATW:-0.001}"
CFG=configs/dro_bso_2species.yaml

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- the control budget, solved per cell -------------------------------------------------
ctrl_budget() {   # $1 = fraction -> echoes "batch epochs"
  python - "$1" "$POOL" "$STEPS" <<'PYEOF'
import math, sys
frac, pool, target = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n_lab = max(int(round(pool * frac / 100.0)), 32)
n_tr  = max(n_lab - max(int(round(n_lab * 0.1)), 8), 8)
# The batch shrink is deliberate -- this is a good-faith baseline, not a matched arm.
# Considered and rejected: fixing batch at 256 to match the distill arm exactly.
# It sounds like the cleaner experiment, but at 0.1% there are only 144 training events, so
# batch 256 is the whole set -- one batch per epoch, deterministic full-batch gd, zero
# gradient noise, while the distill arm still draws stochastic batches from the 200k
# unlabeled pool. That does not remove the asymmetry, it flips it: the control gets no sgd
# regularisation and its best-val checkpoint is chosen from 140,000 candidates on 16 val
# events, so it would be handicapped twice and distillation would look better than it is.
#
# The claim this sweep supports is "our tiny student is the best model at f% labels", and a
# superiority claim requires a baseline trained in good faith -- the batch a practitioner
# would actually pick for that data size. Matching 140,000 optimizer steps (below) already
# guarantees the control is never UNDER-trained; the batch is then free to suit the data.
batch = 256
while batch > 2 and n_tr / batch < 30:     # >=30 batches/epoch or an epoch means nothing
    batch //= 2
bpe = max(1, math.ceil(n_tr / batch))
print(batch, max(1, math.ceil(target / bpe)))
PYEOF
}

if [ "$MODE" = "budget" ]; then
  echo "control budgets for pool=$POOL, target=$STEPS updates:"
  for F in 0.1 1 10 100; do
    read -r B E < <(ctrl_budget "$F")
    printf "  frac %6s  batch %3d x %5d ep = %d updates\n" "$F" "$B" "$E" "$((B>0 ? E*0+E : 0))"
  done
  exit 0
fi

# ---- step 1: teacher caches ---------------------------------------------------------------
if [ "$MODE" = "cache" ]; then
  for F in 0.1 1 10 100; do
    CK="$TEACHER_DIR/ftdro_f${F}_s${TSEED}.pt"
    OUT="$CDIR/tcache_dro2sp_f${F}_s${TSEED}.npz"
    [ -s "$CK" ] || { echo "FATAL: missing teacher $CK"; exit 1; }
    if [ -s "$OUT" ]; then say "SKIP cache f=$F (exists: $OUT)"; continue; fi
    say "CACHE f=$F from $(basename "$CK")"
    # The cache is built with the TEACHER's config (context_len 6272). The student loads the
    # student config (628/640). Only the Row count and the targets must match, and both configs
    # carry divisors [500,220,2.5], so train_distill's check_aligned passes.
    python -u scripts/cache_teacher_preds.py \
      --config configs/dro_bso_ft_2species.yaml \
      --root "$DRO" --scheme ft --device cuda --batch 16 \
      --teacher "$CK" --out "$OUT" 2>&1 | tee "logs/cache_dro2sp_f${F}.log"
    [ -s "$OUT" ] || { echo "FATAL: cache not written for f=$F"; exit 1; }
  done
  say "All caches done"
  ls -lh "$CDIR"/tcache_dro2sp_f*_s${TSEED}.npz
  exit 0
fi

# ---- preflight ----------------------------------------------------------------------------
if [ "$MODE" != "report" ]; then
  miss=0
  for F in 0.1 1 10 100; do
    [ -s "$CDIR/tcache_dro2sp_f${F}_s${TSEED}.npz" ] || {
      echo "MISSING cache: tcache_dro2sp_f${F}_s${TSEED}.npz"; miss=1; }
  done
  [ "$miss" = 0 ] || { echo "FATAL: run '$0 cache' first."; exit 1; }
  [ -f "$CFG" ] || { echo "FATAL: missing $CFG"; exit 1; }
  ngpu=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  [ "$ngpu" -ge 1 ] || { echo "FATAL: no CUDA devices visible"; exit 1; }
  say "preflight OK ($ngpu GPU(s), teacher seed $TSEED, label-seed $TSEED, ctx $CTX, pool $POOL)"
fi

# tag  frac  gpu  arm
JOBS=(
  "f0.1_d   0.1   0  distill"  "f1_d     1     1  distill"
  "f10_d    10    2  distill"  "f100_d   100   3  distill"
  "f0.1_c   0.1   0  labels"   "f1_c     1     1  labels"
  "f10_c    10    2  labels"   "f100_c   100   3  labels"
)

one() {
  local tag="$1" frac="$2" gpu="$3" arm="$4"
  local cache="$CDIR/tcache_dro2sp_f${frac}_s${TSEED}.npz"
  local out sdir armflags budget=()
  if [ "$arm" = "labels" ]; then
    # control: identical except the loss sees only truth. Same --label-seed, so both arms train
    # on the same labeled events and the comparison stays seed-paired. No --semi, no --steps.
    out="runs/dro2sp_semi_f${frac}_lab.json"
    armflags=(--teacher-preds "$cache" --distill-mode labels)
    read -r B E < <(ctrl_budget "$frac")
    budget=(--batch-size "$B" --epochs "$E")
    say "  control budget: batch=$B epochs=$E (~$STEPS updates on ${frac}% of $POOL)"
  else
    out="runs/dro2sp_semi_f${frac}.json"
    armflags=(--teacher-preds "$cache" --semi --distill-mode both
              --alpha "$ALPHA" --feat-weight "$FEATW" --feat-center mean)
    budget=(--batch-size 256 --steps "$STEPS")
  fi
  sdir="runs/students/dro2sp_f${frac}_${arm}"      # One dir per (frac,arm): the checkpoint name
  mkdir -p "$sdir"                                 # does not encode the recipe, so sharing one
                                                   # dir silently overwrites the other arm.
  if [ -s "$out" ]; then
    n=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('runs',[])))" "$out" 2>/dev/null || echo 0)
    say "RESUME $tag ($n seed(s) already in $out)"
  fi
  say "START $tag gpu=$gpu frac=$frac arm=$arm"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/train_distill.py \
    --config "$CFG" --root "$DRO" --scheme fc_little \
    --downsample 10 --context-len "$CTX" \
    "${armflags[@]}" \
    --label-seed "$TSEED" --select-best-val \
    --fractions "$frac" --seeds "$SEEDS" \
    "${budget[@]}" \
    --save-students "$sdir" \
    --device cuda --out "$out" > "logs/dro2sp_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    say "FAILED $tag (rc=$rc) -- logs/dro2sp_${tag}.log"; return $rc
  fi
  # Verify the students actually landed.
  # --save-students is a silent no-op when unset (_save_student returns None), and the saved
  # filename is dro_<scheme>_h<hidden>_f<frac>_s<seed>.pt -- it encodes frac and seed but not
  # the arm. Two arms sharing one directory therefore overwrite each other with no warning,
  # which is how a previous sweep ended with metrics but no weights. This sweep gives each
  # (frac, arm) its own directory; this check proves it worked rather than assuming it.
  local want got
  want=$(echo "$SEEDS" | tr ',' '\n' | grep -c .)
  got=$(ls -1 "$sdir"/*_f${frac}_s*.pt 2>/dev/null | wc -l)
  if [ "$got" -lt "$want" ]; then
    say "Warning $tag: only $got/$want student .pt in $sdir -- weights are missing"
  else
    say "DONE  $tag -> $out  ($got/$want students in $sdir)"
  fi
  return 0
}

if [ "$MODE" != "report" ]; then
  for g in 0 1 2 3; do
    ( for j in "${JOBS[@]}"; do
        # shellcheck disable=SC2086
        set -- $j
        [ "$3" = "$g" ] || continue
        [ -n "${ONLY:-}" ] && [ "$4" != "$ONLY" ] && continue
        one "$1" "$2" "$3" "$4"
      done ) &
  done
  say "all 4 lanes launched (distill then control per lane)"
  wait
  say "ALL DONE"
fi

# ---- saved-student audit ------------------------------------------------------------------
say "SAVED-STUDENT AUDIT"
python - <<'PYEOF'
import glob, os, re
want = 8
tot = missing = 0
print("\n%-40s %6s  %s" % ("directory", "count", "seeds present"))
print("-"*78)
for frac in ("0.1","1","10","100"):
    for arm in ("distill","labels"):
        d = "runs/students/dro2sp_f%s_%s" % (frac, arm)
        ps = sorted(glob.glob(os.path.join(d, "*_f%s_s*.pt" % frac)))
        seeds = sorted(int(re.search(r"_s(\d+)\.pt$", p).group(1)) for p in ps
                       if re.search(r"_s(\d+)\.pt$", p))
        tot += len(ps)
        flag = "" if len(ps) >= want else "   <-- MISSING %d" % (want - len(ps))
        if len(ps) < want: missing += 1
        print("%-40s %3d/%-2d  %s%s" % (os.path.basename(d), len(ps), want, seeds, flag))
print("-"*78)
print("total student .pt: %d  (expect %d = 4 fractions x 2 arms x %d seeds)" % (tot, 8*want, want))
if missing:
    print("\n%d cell(s) are missing weights. Metrics without weights cannot be compressed" % missing)
    print("    or re-scored later. Check logs/dro2sp_*.log for those cells before proceeding.")
else:
    print("\nAll cells have their weights. Distinct directory per (fraction, arm), so the")
    print("arm-blind filename cannot cause a silent overwrite.")
PYEOF

# ---- report -------------------------------------------------------------------------------
say "REPORT"
python - <<'PYEOF'
import glob, json, os, statistics as st
print("\n%-34s %6s %4s %10s %9s  %s" % ("file","frac","n","ratio med","SD","best_epoch"))
print("-"*94)
for p in sorted(glob.glob("runs/dro2sp_semi_f*.json")):
    try: d=json.load(open(p))
    except Exception as e: print("%-34s UNREADABLE %s"%(os.path.basename(p),e)); continue
    rows=d.get("runs",[])
    for f in sorted({r["fraction"] for r in rows}, key=float):
        at=[r for r in rows if r["fraction"]==f]
        v=[r["err68_holdout"]["ratio"] for r in at
           if "ratio" in (r.get("err68_holdout") or {})]
        if not v: continue
        be=sorted(r.get("best_epoch") for r in at if r.get("best_epoch") is not None)
        last=max((r.get("last_epoch") or 0) for r in at)
        stuck=sum(1 for e in be if e>=0.97*last)
        flag="  <-- %d seed(s) at the LAST epoch: Budget too short"%stuck if stuck else ""
        print("%-34s %6s %4d %10.4f %9.4f  %s%s"
              % (os.path.basename(p), f, len(v), st.median(v),
                 st.stdev(v) if len(v)>1 else 0.0, be[:8], flag))
print("\nDRO err68 C/S ratio: LOWER is better.")
print("teacher (2-species, seed 1): 13.490 / 10.129 / 4.430 / 2.806 at 0.1/1/10/100%.")
PYEOF
