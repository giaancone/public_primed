# Waveform feature extraction: fine-tuning, distillation, compression

Code for fine-tuning a pre-trained time-series foundation model on two detector
waveform-regression tasks, distilling it into a small fully-connected student, and
compressing that student for FPGA deployment.

Two tasks are supported:

- **DCH** — drift chamber. Regress the number of primary ionization clusters in a
  waveform. Scored by kaon/pion separation power (higher is better).
- **DRO** — dual-readout calorimeter. Regress the Cherenkov and scintillation
  components and the pulse arrival time. Scored by err68 on the C/S ratio (lower is
  better).

## Layout

```
src/         library code
  data_loader.py    read waveforms, downsample, apply target divisors
  teacher.py        foundation-model backbone and DRO teacher training
  teacher_dch.py    DCH teacher (per-sample head plus a pooled count head)
  student.py        student training: from-scratch and distilled arms
  semi_distill.py   distillation over the unlabeled pool
  metrics.py        shared regression metrics and the weighted loss
  separation.py     DCH metric
  dro_metric.py     DRO metrics
  student_prep.py   student input decimation and scaling

scripts/     entry points, one per pipeline stage

configs/
  dch_ftpc.yaml             DCH teacher and student
  dro_bso_ft_2species.yaml  DRO teacher
  dro_bso_2species.yaml     DRO student
```

## Requirements

Python 3.9+, and two separate environments. Compression needs TensorFlow, which does
not install cleanly alongside the PyTorch environment used for training.

```
pip install -r requirements-train.txt       # teacher and student training
pip install -r requirements-compress.txt    # pruning, QAT, synthesis
```

Synthesis also needs a Xilinx Vitis HLS installation on `PATH`.

The backbone checkpoint is pinned in the configs (`google/timesfm-2.5-200m-pytorch`)
and the `timesfm` package is pinned in `requirements-train.txt`. Both matter:
`teacher.py` calls internals of `timesfm.torch.util`, so another release may
change the preprocessing or fail outright.

## License

Released under the Apache License 2.0; see `LICENSE`.

## Data

Not included. The DCH inputs are `.npz` files holding a waveform array plus
per-sample ionization tags; the DRO inputs are `.h5` files holding a summed waveform
and a label array. The glob patterns and key names are in the `data` block of each
config. Point `--root` at the directory containing them.

Both training entry points accept `--stub`, which swaps in a small backbone so the
loader, model and metric paths run without a GPU. `train_teacher_dch.py --stub` also
fabricates a DCH-schema fixture and so needs no data at all; `train_teacher_dro.py --stub`
still reads from `--root`.

## Running the pipeline

The four stages run in order. Substitute the detector's data directory for `$DCH` or
`$DRO` throughout.

**1. Fine-tune the teacher.** One run per label fraction and seed; the JSON is
rewritten after each cell, and `--resume` skips completed ones.

```
python scripts/train_teacher_dch.py   --config configs/dch_ftpc.yaml \
    --root $DCH --device cuda --out runs/dch_teacher.json

python scripts/train_teacher_dro.py --config configs/dro_bso_ft_2species.yaml \
    --root $DRO --device cuda --out runs/dro_teacher.json
```

**2. Cache the teacher's predictions and embeddings.** Build one cache per label
fraction, from the teacher trained at that same fraction. A student must not be given
a cache from a teacher trained on more labels than it has.

```
python scripts/cache_teacher_preds.py --config configs/dro_bso_ft_2species.yaml \
    --root $DRO --scheme ft --device cuda \
    --teacher <checkpoint>.pt --out <cache>.npz
```

**3. Train the students.** The distilled arm and the from-scratch control differ only
in the loss; both must be given the same `--label-seed` so they train on the same
labeled events and can be compared seed by seed.

```
# distilled
python scripts/train_distill.py --config configs/dro_bso_2species.yaml \
    --root $DRO --scheme fc_little --downsample 10 --context-len 628 \
    --teacher-preds <cache>.npz --semi --distill-mode both \
    --alpha 0.5 --feat-weight 0.001 --feat-center mean \
    --label-seed 1 --select-best-val --steps 140000 --batch-size 256 \
    --device cuda --out runs/dro_student.json

# from-scratch control: same command with --distill-mode labels, no --semi
```

DCH students take `--downsample 5` instead (3008 to 602), plus two flags that decide
whether the number is comparable at all: `--sel-val-frac 0.3`, which holds back a
disjoint selection split, and `--count-divisor 53`. Both are DCH-only; DRO uses a
random holdout and takes neither.

`scripts/run_dch_students.sh` (DCH) and `scripts/run_dro_students.sh` (DRO) run
the full fraction-by-seed sweep for both arms, apply each detector's flags, and
compute the control's batch size and epoch count so that both arms get the same
number of optimizer steps.

**4. Compress and synthesize.** One command per detector. `--prune-to` prunes the
float student here; pass an already-pruned model and omit it to go straight to QAT.
`--po2-layers` selects which dense layers get power-of-two kernels — omitted means
all of them.

```
python scripts/qat_po2.py --detector dch --cache <cache>.npz --meta <cache_meta>.json \
    --pruned-model <student>.h5 --prune-to 0.6 \
    --kernel-quantizer po2 --bits 10 --target-divisor 53 \
    --out runs/qat_dch.json

python scripts/qat_po2.py --detector dro --cache <cache>.npz --meta <cache_meta>.json \
    --pruned-model <student>.h5 --prune-to 0.1 \
    --kernel-quantizer po2 --bits 11 --po2-layers 0 \
    --target-weights 4,1,0.3 --epochs 150 \
    --out runs/qat_dro.json

python scripts/hls4ml_sweep.py --models-dir <dir> --out hls4ml_results.json
```

`--target-divisor 53` is required for DCH and `--target-weights` for DRO; both are
checked, and the DRO weights have to pair with the divisors in the config. Run-to-run
spread at a fixed operating point is larger than the gap between bit widths, so
compress several times and select on validation loss rather than reading one run.

## Notes

The student input width comes from `--downsample` and `--context-len`, which override
the config. `dch_ftpc.yaml` sets neither, so the DCH flag is required; the DRO config
carries 10 and 640 while the reported student uses `--context-len 628`. Getting this
wrong trains a model with a different number of parameters than the one reported, and
nothing fails.

Target divisors and per-target loss weights are a single decision. The divisor enters
the loss squared, and err68 is invariant to it, so changing one without the other
skews training while every reported metric still looks correct.

Validation is taken from inside the labeled budget rather than in addition to it, so
a stated label fraction is the total amount of truth consumed.

DCH training and evaluation are separate productions. Training reads
`processed_data_train/` with no momentum filter; `pion/` and `kaon/` are filtered to
`eval.momentum` +/- `eval.mom_tol` when loaded, so the filter applies to evaluation
only. DRO has no equivalent split -- its holdout is carved from the same pool the
fractions are drawn from.
