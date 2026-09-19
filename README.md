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
  ft_finetune.py    foundation-model backbone and DRO teacher training
  ft_peakcount.py   DCH teacher (per-sample head plus a pooled count head)
  fs_baseline.py    student training: from-scratch and distilled arms
  semi_distill.py   distillation over the unlabeled pool
  metrics.py        shared regression metrics and the weighted loss
  separation.py     DCH metric
  dro_metric.py     DRO metrics
  liangyu_prep.py   student input decimation and scaling

scripts/     entry points, one per pipeline stage
configs/     one config per detector, holding both teacher and student settings
```

## Requirements

Python 3.9+, and two separate environments. Compression needs TensorFlow, which does
not install cleanly alongside the PyTorch environment used for training.

```
pip install -r requirements-train.txt       # teacher and student training
pip install -r requirements-compress.txt    # pruning, QAT, synthesis
```

Synthesis also needs a Xilinx Vitis HLS installation on `PATH`.

The backbone checkpoint is pinned in the configs (`google/timesfm-2.5-200m-pytorch`),
but the `timesfm` package version is not: `ft_finetune.py` uses internals of
`timesfm.torch.util`, so a different release may change preprocessing or fail
outright. Pin it to the version you used.

## Data

Not included. The DCH inputs are `.npz` files holding a waveform array plus
per-sample ionization tags; the DRO inputs are `.h5` files holding a summed waveform
and a label array. The glob patterns and key names are in the `data` block of each
config. Point `--root` at the directory containing them.

Both training entry points accept `--stub`, which substitutes a small backbone so the
loader, model and metric paths run without a GPU.

## Running the pipeline

The four stages run in order. Substitute the detector's data directory for `$DCH` or
`$DRO` throughout.

**1. Fine-tune the teacher.** One run per label fraction and seed; the JSON is
rewritten after each cell, and `--resume` skips completed ones.

```
python scripts/train_ftpc.py   --config configs/dch_ftpc.yaml \
    --root $DCH --device cuda --out runs/dch_teacher.json

python scripts/train_ft_dro.py --config configs/dro_bso_ft_2species.yaml \
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
    --label-seed 1 --steps 140000 --batch-size 256 \
    --device cuda --out runs/dro_student.json

# from-scratch control: same command with --distill-mode labels, no --semi
```

`scripts/run_semi_sweep.sh` (DCH) and `scripts/run_2species_students.sh` (DRO) run
the full fraction-by-seed sweep for both arms, and compute the control's batch size
and epoch count so that both arms get the same number of optimizer steps.

**4. Compress and synthesize.**

```
python scripts/qat_po2.py --detector dro --cache <cache>.npz --meta <cache_meta>.json \
    --pruned-model <student>.h5 --bits 11 --po2-layers 0 --out runs/qat.json

python scripts/hls4ml_sweep.py --models-dir <dir> --out hls4ml_results.json
```

## Notes

The student input width is set by `--downsample` and `--context-len` on the command
line, not by the config. Omitting them trains a model with a different number of
parameters than the one reported, and nothing fails.

Target divisors and per-target loss weights are a single decision. The divisor enters
the loss squared, and err68 is invariant to it, so changing one without the other
skews training while every reported metric still looks correct.

Validation is taken from inside the labeled budget rather than in addition to it, so
a stated label fraction is the total amount of truth consumed.
