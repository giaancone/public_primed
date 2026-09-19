"""Power-of-two quantization-aware training: the route to DSP = 0.

Why POWER-OF-TWO and not a narrower fixed point. This is a structural property of
hls4ml, not a tuning outcome. In hls4ml 1.3.0:

    hls4ml/backends/fpga/fpga_backend.py:504
        if isinstance(weight_T, ExponentPrecisionType):
            product = 'weight_exponential'     # nnet_mult.h: a << w.weight  -- A shift
        else:
            product = 'mult'                   # nnet_mult.h: a * w          -- A multiplier

    hls4ml/model/quantizers.py:194        QKerasPO2Quantizer -> ExponentPrecisionType
    hls4ml/converters/keras/qkeras.py:17  'quantized_po2' -> QKerasPO2Quantizer

So when the kernel quantizer is `quantized_po2`, the emitted C++ contains no multiply for
those weights and DSP = 0 follows by construction. Bit width does not enter this decision:
`quantized_bits` at 8, 10, 12 or 16 bits all take the `'mult'` branch, so narrowing a
fixed-point kernel cannot move the DSP column.

Three things that must not be changed, each for a specific reason:

  * biases stay `quantized_bits`. Biases are added, never multiplied, so making them powers
    of two buys nothing in hardware and only costs accuracy.
  * activations stay `quantized_relu`. hls4ml asserts on power-of-two data --
    fpga_backend.py:500: "Only ExponentPrecisionType (aka 'power of 2') weights are currently
    supported, not data." A po2 activation quantizer aborts the conversion.
  * Bits >= 9. `quantized_po2` does not map 0.0 to 0.0 at low bit widths: at 4 bits a
    pruned-away weight reappears as 0.0625, at 8 bits as 5.4e-20, and only from 9 bits up does
    it underflow to exact zero. Below 9 the pruning is silently undone and any nonzero-weight
    count becomes fiction. `_assert_zero_preserving` enforces this.

run (in an isolated TensorFlow/QKeras interpreter, never the torch environment):

    .venv-compress/bin/python scripts/qat_po2.py --detector dch \\
        --cache  <cache>.npz --meta <cache_meta>.json \\
        --pruned-model <pruned>.h5 --bits 10 --out runs/qat_po2_dch.json

Wiring can be checked without any data before committing a GPU to a run.
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# From the reference QAT configuration. Kept identical so that power-of-two and fixed-point
# runs differ in exactly one thing: the kernel quantizer.
QAT_EPOCHS, QAT_BATCH, QAT_PATIENCE, QAT_LR = 50, 512, 10, 1e-4
MIN_PO2_BITS = 9


def _expected_dsp(audit, po2_layers, kernel_quantizer):
    """Nonzero weights that still need a multiplier, i.e. the DSP exposure before --force-lut.

    A po2 layer emits a shift and needs none. Every other layer keeps a quantized_bits kernel
    and takes hls4ml's 'mult' branch, so its nonzeros are candidate multipliers. Reporting a
    flat 0 for any po2 run (the previous behavior) hid exactly the case this sweep created:
    --po2-layers 0 leaves ~541 nonzero multiplies in layers 1-3.

    not a synthesis result. --force-lut can push these to fabric -- it did for the deployed
    model's final 8x3 layer (22 multiplies, ~1k LUT). Whether it absorbs 25x that many at an
    acceptable LUT cost is a question only synthesis answers.
    """
    if kernel_quantizer != "po2":
        return int(audit.get("nonzero", 0))
    sel = set(po2_layers or [])
    layers = audit.get("layers") or []
    if not po2_layers:                       # empty/None -> all layers are po2
        return 0
    return int(sum(L.get("nonzero", 0) for i, L in enumerate(layers) if i not in sel))


def _assert_zero_preserving(bits):
    """Refuse a bit width that would resurrect pruned weights.

    Measured on qkeras 0.9.x: q(0.0) is 0.25 at 3 bits, 0.0625 at 4, 3.9e-3 at 5, 1.5e-5 at 6,
    2.3e-10 at 7, 5.4e-20 at 8, and exactly 0.0 from 9 up. This is checked live rather than
    trusted from that table, because it is a property of the installed qkeras.
    """
    import tensorflow as tf
    from qkeras.quantizers import quantized_po2
    got = np.array(quantized_po2(bits=bits)(tf.constant(np.zeros(4, np.float32))))
    if not np.all(got == 0):
        raise SystemExit(
            "quantized_po2(bits=%d) maps 0.0 -> %g, so every PRUNED weight comes back "
            "nonzero and the sparsity is silently undone. Use --bits %d or more."
            % (bits, float(got[0]), MIN_PO2_BITS))


def wmse(weights, ratio_weight=0.0, ratio_cols=(0, 1)):
    """Per-target weighted MSE, optionally plus RATIO-CONSISTENCY.

    The weighted part matches the closure the
    deployed DRO artifacts were fine-tuned with.

    Why the ratio term is here. The DRO student is trained with ratio-consistency
    (the student scheme's `ratio_weight`). Fine-tuning it for pruning and QAT under a loss
    without that term actively undoes it: the ratio degrades by several-fold, driven almost
    entirely by the scintillation component while the Cherenkov one barely moves. Per-target
    weights cannot substitute -- upweighting s eightfold still
    lost it, because the ratio is a relationship between c and s, not a per-channel quantity.
    The compression loss must match the training loss.

    Same stable cross-product as student: (pc*ts - ps*tc) is zero exactly when the c and s
    relative errors agree (and so cancel in R = c/s), normalized by the batch-mean tc*ts so a
    near-zero true c or s cannot blow it up. ratio_weight=0 -> byte-identical to before."""
    import tensorflow as tf
    w = tf.constant(weights, dtype=tf.float32)
    rw = float(ratio_weight)
    ci, si = int(ratio_cols[0]), int(ratio_cols[1])

    @tf.autograph.experimental.do_not_convert
    def weighted_mse(y_true, y_pred):
        base = tf.reduce_mean(tf.square(y_pred - y_true) * w)
        if rw <= 0.0:
            return base
        tc, ts = y_true[:, ci], y_true[:, si]
        pc, ps = y_pred[:, ci], y_pred[:, si]
        m = tf.cast(tf.logical_and(tc > 1e-6, ts > 1e-6), tf.float32)
        n = tf.reduce_sum(m) + 1e-6
        cross = (pc * ts - ps * tc) * m
        denom = tf.reduce_sum(tc * ts * m) / n
        return base + rw * (tf.reduce_sum(tf.square(cross)) / n) / (denom + 1e-6)
    return weighted_mse


# ---------------------------------------------------------------------------------------
# Pruning and quantization-aware training.
#
# Both take the loss as an argument rather than compiling one internally, because a
# multi-target regression needs per-target weights and a hard-coded `loss='mse'` cannot
# express them. With no weights supplied the caller passes unit weights, and
# reduce_mean(square(err) * 1.0) is exactly Keras 'mse', so a single-target run behaves the
# same as compiling with loss='mse' directly.
#
# The surrounding recipe -- clone, ConstantSparsity schedule, end_step, UpdatePruningStep,
# EarlyStopping with restore_best_weights and patience, the mask-enforcement callback, and
# the return value -- follows the published compression procedure.
def prune_and_strip_weighted(base_model, sparsity, datasets, lr, epochs, batch_size,
                             patience, loss):
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    if sparsity == 0.0:
        return base_model, 0.0
    X_tr, Y_tr = datasets["X_train"], datasets["Y_train"]
    X_va, Y_va = datasets["X_val"], datasets["Y_val"]
    end_step = (len(X_tr) // batch_size) * epochs
    m = tf.keras.models.clone_model(base_model)
    m.set_weights(base_model.get_weights())
    m = tfmot.sparsity.keras.prune_low_magnitude(
        m, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=sparsity, begin_step=0, end_step=end_step))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["mae"])
    h = m.fit(X_tr, Y_tr, validation_data=(X_va, Y_va), epochs=epochs,
              batch_size=batch_size,
              callbacks=[tfmot.sparsity.keras.UpdatePruningStep(),
                         tf.keras.callbacks.EarlyStopping(
                             monitor="val_loss", patience=patience,
                             restore_best_weights=True, verbose=0)],
              verbose=0)
    _vl = [float(v) for v in h.history["val_loss"]]
    print("[prune] ran %d/%d epochs; best at epoch %d; val_loss %.6g -> %.6g%s"
          % (len(_vl), epochs, int(min(range(len(_vl)), key=lambda i: _vl[i])) + 1,
             _vl[0], _vl[-1], "   <-- EARLY-STOPPED" if len(_vl) < epochs else ""))
    return tfmot.sparsity.keras.strip_pruning(m), float(min(_vl))


def train_qat_weighted(qat_model, masks, lr, datasets, epochs, batch_size, patience, loss):
    import tensorflow as tf
    qat_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["mae"])
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                            restore_best_weights=True, verbose=0)]
    if any(m is not None for m in masks):
        _m = masks

        class _EnforceSp(tf.keras.callbacks.Callback):
            """Re-apply pruning masks after each training batch."""

            def on_train_batch_end(self, batch, logs=None):
                for layer, mask in zip(self.model.layers, _m):
                    if mask is not None and hasattr(layer, "kernel"):
                        w = layer.kernel.numpy()
                        if w.shape == mask.shape:
                            layer.kernel.assign(w * mask)
        cbs.append(_EnforceSp())
    h = qat_model.fit(datasets["X_train"], datasets["Y_train"],
                      validation_data=(datasets["X_val"], datasets["Y_val"]),
                      epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=0)
    # Report the epoch count. The trainer runs verbose=0 with EarlyStopping(patience),
    # so a run that quits after 5 of 150 epochs is indistinguishable from one that used all 150 --
    # same log, same single best_val_loss, and the only symptom is an accuracy shortfall that
    # looks like the quantizer's fault. Printing the curve separates "undertrained" from
    # "this operating point is wrong" without re-running anything.
    vl = [float(v) for v in h.history["val_loss"]]
    ran, best_ep = len(vl), int(min(range(len(vl)), key=lambda i: vl[i])) + 1
    print("[qat] ran %d/%d epochs; best at epoch %d; val_loss %.6g -> %.6g (best %.6g)%s"
          % (ran, epochs, best_ep, vl[0], vl[-1], min(vl),
             "   <-- EARLY-STOPPED" if ran < epochs else ""))
    if ran < epochs and best_ep <= max(3, ran // 4):
        print("[qat] warning: best epoch is in the first quarter of a short run -- this is an "
              "UNDERTRAINED model, not a quantization limit. Raise --patience/--lr before "
              "concluding anything about bits or sparsity.")
    train_qat_weighted.last_history = vl
    return qat_model, float(min(vl))


def build_po2_qat(fp_model, po2_bits, bias_bits=(12, 2), kernel="po2", po2_layers=None):
    """Mirrors the reference QAT model builder, with a PO2 kernel quantizer.

    Structure (layer order, bn placement, weight transfer, mask capture) follows the reference
    function so the only difference from the <12,2> arm is `kernel_quantizer`. Returns
    (qat_model, masks) with masks in the reference format: one entry per layer, None for non-Dense.
    """
    import tensorflow as tf
    from qkeras import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras.quantizers import quantized_po2

    # kernel="bits" reproduces the <12,2> arm through this exact code path. That matters more
    # than it sounds: the archived <12,2> DCH artifacts turned out to be from the superseded
    # RAW-COUNT run and score NaN, and the cdiv53 sp50 model was lost with /tmp. Regenerating
    # both arms here means the comparison depends on no missing file and differs in exactly
    # one line -- this one.
    if kernel not in ("po2", "bits"):
        raise SystemExit("kernel must be 'po2' or 'bits', got %r" % kernel)
    kq_po2 = quantized_po2(bits=po2_bits)                  # -> ExponentPrecisionType -> shift
    kq_bits = quantized_bits(bias_bits[0], bias_bits[1], symmetric=1)  # -> 'mult' -> DSP

    def _kq(idx):
        """Per-layer kernel quantizer. hls4ml calls product_type() per layer
        (fpga_backend.py:496), so a mixed model is legal: po2 layers emit `a << w` and the
        rest emit `a * w`. That matters because the multiplies are not spread evenly --
        layer 0 alone is 94.4% of them (628x16 of 10,648 on DRO, 602x16 of 10,216 on DCH).
        Putting po2 only there removes ~94% of the multipliers while the small output layers,
        where a 3-target regression needs its precision, keep full fixed point."""
        if kernel == "bits":
            return kq_bits
        if po2_layers is None:
            return kq_po2
        return kq_po2 if idx in po2_layers else kq_bits
    bq = quantized_bits(bias_bits[0], bias_bits[1], symmetric=1)   # biases are added
    aq = quantized_relu(bias_bits[0], bias_bits[1])                # data must not be po2

    dense_fp = [l for l in fp_model.layers if isinstance(l, tf.keras.layers.Dense)]
    bn_fp = [l for l in fp_model.layers
             if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not dense_fp:
        raise SystemExit("no Dense layers found in the starting model")

    qat = tf.keras.Sequential([tf.keras.layers.Input(shape=fp_model.input_shape[1:])])
    for i, size in enumerate([l.units for l in dense_fp[:-1]]):
        qat.add(QDense(size, kernel_quantizer=_kq(i), bias_quantizer=bq))
        if bn_fp:
            qat.add(tf.keras.layers.BatchNormalization())
        qat.add(QActivation(aq))
    qat.add(QDense(dense_fp[-1].units, kernel_quantizer=_kq(len(dense_fp) - 1),
                   bias_quantizer=bq))
    qat.add(QActivation(aq))
    _ = qat(tf.random.normal((1,) + fp_model.input_shape[1:]))

    masks, d_idx, bn_idx = [], 0, 0
    for layer in qat.layers:
        if isinstance(layer, QDense) and d_idx < len(dense_fp):
            fp_w = dense_fp[d_idx].get_weights()
            layer.set_weights(fp_w)
            masks.append((np.abs(fp_w[0]) > 1e-7).astype(np.float32))
            d_idx += 1
        elif isinstance(layer, tf.keras.layers.BatchNormalization) and bn_idx < len(bn_fp):
            layer.set_weights(bn_fp[bn_idx].get_weights())
            bn_idx += 1
            masks.append(None)
        else:
            masks.append(None)
    return qat, masks


def audit_po2(model, masks=None):
    """Prove the artifact is what we claim before any number leaves this script.

    Checks the two properties the DSP=0 claim rests on: every effective kernel weight is an
    exact power of two (so hls4ml emits a shift), and the pruning survived quantization (so
    the nonzero count we quote is real).
    """
    from qkeras import QDense
    out = {"layers": [], "nonzero": 0, "all_po2": True, "sparsity_preserved": True}
    mi = 0
    for layer in model.layers:
        if not isinstance(layer, QDense):
            mi += 1
            continue
        # the effective kernel -- what inference and hls4ml actually see, not the stored float
        qk = np.array(layer.kernel_quantizer_internal(layer.kernel))
        nz = qk[qk != 0]
        lg = np.log2(np.abs(nz)) if nz.size else np.array([])
        is_po2 = bool(np.allclose(lg, np.round(lg))) if nz.size else True
        rec = {"layer": layer.name, "shape": list(qk.shape),
               "nonzero": int(nz.size), "total": int(qk.size),
               "all_po2": is_po2,
               "distinct_exponents": int(np.unique(np.round(lg)).size) if nz.size else 0}
        if masks is not None and mi < len(masks) and masks[mi] is not None:
            m = masks[mi]
            if m.shape == qk.shape:
                # a weight the mask killed must still be zero after quantization
                resurrected = int(((m == 0) & (qk != 0)).sum())
                rec["resurrected_by_quantizer"] = resurrected
                if resurrected:
                    out["sparsity_preserved"] = False
        out["all_po2"] &= is_po2
        out["nonzero"] += int(nz.size)
        out["layers"].append(rec)
        mi += 1
    return out


def score(detector, model, cache, meta):
    """Same metric, same arrays, and same code path as the training-time evaluation."""
    from src import dro_metric, separation as sepmod
    if detector == "dch":
        ls = float(meta["length_scale"])
        cp = model.predict(cache["Xpi_test"], verbose=0).ravel()
        ck = model.predict(cache["Xka_test"], verbose=0).ravel()
        # separation power is a ratio in matched units, so the count divisor cancels exactly
        return {"separation": float(sepmod.separation_power(cp, ck, ls)["separation"])}
    div = list(meta["divisors"])
    P = model.predict(cache["X_test"], verbose=0).copy()
    for j, dv in enumerate(div):
        P[:, j] *= dv
    # Divisors=None is correct and LOAD-BEARING. P was just scaled to physical units by
    # the loop above, and dro_metrics multiplies the t0 resolution by divisors[2] again when it
    # is given them -- so passing `div` here scaled t0 twice. It shipped t0=2.2611 to the
    # hls4ml handoff; the true value is 0.9130 ns. c/s/ratio are relative errors and therefore
    # divisor-invariant, which is exactly why this hid. Proven by
    # equivalent to the training-time evaluation.
    r = dro_metric.dro_metrics(P, cache["Y_test"], ["c", "s", "t0"], None)
    e = r["err68"] if "err68" in r else r
    return {"err68": {k: float(v) for k, v in e.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["dch", "dro"])
    ap.add_argument("--cache", required=True, help="data npz (reference key names)")
    ap.add_argument("--meta", required=True, help="json: length_scale (dch) or divisors (dro)")
    ap.add_argument("--pruned-model", required=True,
                    help="the PRUNED, stripped float model the <12,2> arm also started from. "
                         "With --prune-to this may instead be the UNPRUNED float student, "
                         "which is then pruned here with the reference _prune_and_strip.")
    ap.add_argument("--prune-to", type=float, default=None,
                    help="target weight sparsity, e.g. 0.5. Use when no pruned artifact "
                         "survives: runs the reference _prune_and_strip (30 epochs, lr 5e-5, "
                         "batch 512, patience 8) on --pruned-model "
                         "first. No DCH pruned float model was archived, only QAT'd ones.")
    ap.add_argument("--target-divisor", type=float, default=1.0,
                    help="divide Y_train/Y_val by this. Required = 53 for DCH: the "
                         "cache holds RAW cluster counts (5-53) and the deployed <12,2> "
                         "student was trained on count/53. Leaving it at 1.0 trains po2 on a "
                         "different target than the baseline, making the comparison "
                         "meaningless -- and quantized_bits(12,2) spans only +/-4, which is "
                         "why the raw-count arm returned NaN.")
    ap.add_argument("--kernel-quantizer", default="po2", choices=["po2", "bits"],
                    help="po2 -> quantized_po2 kernels (shift, DSP=0). bits -> quantized_bits "
                         "at --bias-bits, i.e. the <12,2> BASELINE arm through identical code.")
    ap.add_argument("--bits", type=int, default=10,
                    help="po2 exponent bits (>= %d, see _assert_zero_preserving)" % MIN_PO2_BITS)
    ap.add_argument("--bias-bits", default="12,2")
    ap.add_argument("--po2-layers", default="",
                    help="HYBRID: comma list of dense-layer indices to make po2, e.g. '0'. "
                         "Empty = all layers (with --kernel-quantizer po2). Layer 0 carries "
                         "94.4%% of the multiplies on DRO and 94.3%% on DCH, so '0' removes "
                         "~94%% of the multipliers while the small output layers keep full "
                         "fixed-point precision. hls4ml resolves the product type per layer, "
                         "so mixed models are legal.")
    ap.add_argument("--ratio-weight", type=float, default=0.0,
                    help="DRO: weight of the ratio-consistency term in the prune/QAT loss. "
                         "MUST match the value the student was TRAINED with "
                         "(the student scheme's ratio_weight). Without it the fine-tune "
                         "destroys ratio performance -- 2.046 -> 6.507 measured. Requires "
                         "--target-weights (it shares that loss closure).")
    ap.add_argument("--target-weights", default="",
                    help="per-target loss weights, e.g. '1,8,0.3'. Required for DRO: "
                         "the deployed DRO artifacts were prune- AND QAT-fine-tuned under our "
                         "weighted MSE, not the reference plain mse. "
                         "Omit for DCH -- a single scalar target, where plain MSE is correct "
                         "and reproduces the recorded 3.3192.")
    ap.add_argument("--prune-epochs", type=int, default=30,
                    help="reference prune-scan value")
    ap.add_argument("--epochs", type=int, default=QAT_EPOCHS)
    ap.add_argument("--lr", type=float, default=QAT_LR)
    ap.add_argument("--batch", type=int, default=QAT_BATCH)
    ap.add_argument("--patience", type=int, default=QAT_PATIENCE)
    ap.add_argument("--baseline", default="",
                    help="the <12,2> .h5 to score alongside, for a same-run comparison")
    ap.add_argument("--save-model", default="")
    ap.add_argument("--out", default="runs/qat_po2.json")
    a = ap.parse_args()

    if a.kernel_quantizer == "po2":
        if a.bits < MIN_PO2_BITS:
            raise SystemExit("--bits %d < %d; see _assert_zero_preserving"
                             % (a.bits, MIN_PO2_BITS))
        _assert_zero_preserving(a.bits)

    from qkeras.utils import load_qmodel      # noqa: F401  (baseline may be a QAT model)
    import tensorflow as tf

    bb = tuple(int(v) for v in a.bias_bits.split(","))
    tw = [float(v) for v in a.target_weights.split(",")] if a.target_weights else None

    def _loss_for(y):
        """The training loss for a target array `y`.

        With --target-weights this is the per-target weighted MSE (plus the ratio term when
        --ratio-weight is set). Without it, the weights are all ones, and
        reduce_mean(square(err) * 1.0) is exactly Keras 'mse' -- so a single-target run is
        numerically identical to compiling with loss='mse'.
        """
        n_t = int(y.shape[1]) if getattr(y, "ndim", 1) > 1 else 1
        return wmse(tw if tw is not None else [1.0] * n_t, a.ratio_weight)
    if a.detector == "dro" and tw is None:
        raise SystemExit(
            "DRO needs --target-weights, and the RIGHT value depends on the cache's divisors:\n"
            "    divisors [565, 1755, 2.4766] (old, single-species)  -> --target-weights 1,8,0.3\n"
            "    divisors [500, 220,  2.5]    (corrected, 2-species) -> --target-weights 4,1,0.3\n"
            "The divisor and the weight are ONE decision -- [1,8,0.3] existed only to compensate\n"
            "for a div_s that was ~6x too large (a p99 pooled over BGO+BSO+PWO). Pairing it with\n"
            "the corrected divisors weights s 14x MORE than c. See the check below and\n"
            "configs/dro_bso_2species.yaml CHANGE 3.")
    if tw is not None:
        print("[loss] weighted MSE %s (both prune and QAT)" % tw)
    cache = np.load(a.cache)
    meta = json.load(open(a.meta))

    # ---- species + LOSS-BALANCE guards (DRO) --------------------------------------------
    if a.detector == "dro":
        # 1. Which species is this cache? Written by the cache exporter. Older
        #    caches predate the key; absent is not proof of anything, so say so and move on.
        nsp = meta.get("n_species")
        if nsp is None:
            print("[species] cache meta has no `species` key -- cannot verify which species "
                  "it holds. A single-species cache is not comparable to a merged-species "
                  "benchmark.")
        else:
            print("[species] cache holds %d species: %s" % (nsp, meta.get("species")))
            if nsp < 2:
                print("[species] warning: single-species cache. The published benchmark is "
                      "measured on merged species; do not quote this result against it.")

        # 2. the divisor/weight trap, Computed rather than assumed.
        #    Effective emphasis on relative error is w_i * (mean_i / D_i)^2, so the divisor
        #    enters squared and silently reweights the loss. Measured from Y_test, which the
        #    exporter stores in original units.
        #        565/1755 + [4,1,0.3] -> c:s = 115   (teacher trained to ignore s)
        #        565/1755 + [1,8,0.3] -> c:s = 3.6   (the old compensation -- what shipped)
        #        500/220  + [4,1,0.3] -> c:s = 2.1   (correct)
        #        500/220  + [1,8,0.3] -> c:s = 0.07  (14x s-biased -- the trap)
        #    Both failure modes land far outside [0.3, 30]; both working recipes sit inside it.
        try:
            _div = [float(v) for v in meta["divisors"]]
            _yt = np.asarray(cache["Y_test"], dtype=np.float64)
            _mc, _ms = float(_yt[:, 0].mean()), float(_yt[:, 1].mean())
            _eff = (tw[0] * (_mc / _div[0]) ** 2) / (tw[1] * (_ms / _div[1]) ** 2)
            print("[loss] divisors=%s  mean c=%.1f s=%.1f  ->  effective c:s = %.3g"
                  % (_div[:2], _mc, _ms, _eff))
            if not (0.3 <= _eff <= 30.0):
                raise SystemExit(
                    "[loss] effective c:s = %.3g is outside [0.3, 30] -- the weights and the\n"
                    "       divisors disagree. With divisors %s use --target-weights %s.\n"
                    "       (err68 is divisor-INVARIANT for c/s/ratio, which is exactly why this\n"
                    "       kind of mismatch hid for months: every reported number looks fine\n"
                    "       while the objective is badly skewed.)"
                    % (_eff, _div[:2],
                       "4,1,0.3" if _div[1] < 1000 else "1,8,0.3"))
        except (KeyError, IndexError, ValueError, ZeroDivisionError) as _e:
            print("[loss] could not compute the effective c:s (%s) -- check it by hand" % _e)
    fp = tf.keras.models.load_model(a.pruned_model, compile=False)

    # target scale. Applied to train and val, exactly as the cdiv53 run did. Guard rather
    # than trust: a DCH run left at divisor 1.0 would silently train against a target the
    # baseline never saw.
    Ytr = cache["Y_train"].astype(np.float32) / a.target_divisor
    Yv = cache["Y_val"].astype(np.float32) / a.target_divisor
    if a.detector == "dch" and Ytr.max() > 4.0:
        raise SystemExit(
            "DCH target max is %.2f after dividing by %g. The deployed <12,2> student was "
            "trained on count/53 (max ~1.0); anything above 4.0 is also outside "
            "quantized_bits(12,2). Pass --target-divisor 53." % (Ytr.max(), a.target_divisor))
    print("[target] divisor=%g -> train max %.4f, mean %.4f"
          % (a.target_divisor, float(Ytr.max()), float(Ytr.mean())))

    if a.prune_to is not None:
        # Reference recipe and constants: 30 epochs, lr 5e-5, batch 512, patience 8.
        print("[prune] %s -> %.0f%% sparsity"
              % (os.path.basename(a.pruned_model), 100 * a.prune_to))
        fp.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss="mse", metrics=["mae"])
        ds_p = {"X_train": cache["X_train"], "Y_train": Ytr,
                "X_val": cache["X_val"], "Y_val": Yv}
        fp, _ = prune_and_strip_weighted(fp, a.prune_to, ds_p, 5e-5, a.prune_epochs,
                                         512, 8, _loss_for(Ytr))
        got = 1.0 - (sum(int((np.abs(w) > 1e-7).sum()) for w in fp.get_weights() if w.ndim == 2)
                     / sum(int(w.size) for w in fp.get_weights() if w.ndim == 2))
        print("[prune] achieved sparsity %.3f (target %.3f)" % (got, a.prune_to))
        if abs(got - a.prune_to) > 0.05:
            raise SystemExit("pruning missed its target -- tfmot's ConstantSparsity applies "
                             "the mask only every `frequency` steps; too few steps yields a "
                             "DENSE model that still reports success "
                             "in too few steps to apply the mask.")

    po2_layers = ([int(v) for v in a.po2_layers.split(",")] if a.po2_layers else None)
    if po2_layers is not None and a.kernel_quantizer != "po2":
        raise SystemExit("--po2-layers only applies with --kernel-quantizer po2")
    qat, masks = build_po2_qat(fp, a.bits, bb, kernel=a.kernel_quantizer,
                               po2_layers=po2_layers)
    pre = audit_po2(qat, masks)
    print("[pre-train] nonzero=%d  all_po2=%s" % (pre["nonzero"], pre["all_po2"]))

    datasets = {"X_train": cache["X_train"], "Y_train": Ytr,
                "X_val": cache["X_val"], "Y_val": Yv}
    qat, best_val = train_qat_weighted(qat, masks, a.lr, datasets,
                                       a.epochs, a.batch, a.patience, _loss_for(Ytr))

    post = audit_po2(qat, masks)
    if a.kernel_quantizer == "po2" and po2_layers is None and not post["all_po2"]:
        raise SystemExit("AUDIT FAILED: kernel weights are not all powers of two -- hls4ml "
                         "would emit `mult`, not `weight_exponential`, and DSP would not be 0")
    if not post["sparsity_preserved"]:
        raise SystemExit("AUDIT FAILED: the quantizer resurrected pruned weights -- the "
                         "nonzero count would be a fiction. Raise --bits.")

    kq_desc = ("quantized_po2(bits=%d)" % a.bits if a.kernel_quantizer == "po2"
               else "quantized_bits(%d,%d)" % bb)
    res = {"detector": a.detector, "kernel_quantizer": kq_desc,
           "arm": a.kernel_quantizer,
           "bias_quantizer": "quantized_bits(%d,%d)" % bb,
           "activation_quantizer": "quantized_relu(%d,%d)" % bb,
           "pruned_model": a.pruned_model, "best_val_loss": best_val,
           "target_divisor": a.target_divisor, "pruned_here_to": a.prune_to,
           "target_weights": tw, "loss": ("weighted_mse" if tw else "mse"),
           "po2_layers": po2_layers,
           "prune_epochs": a.prune_epochs, "qat_epochs": a.epochs,
           "nonzero": post["nonzero"],
           "kb": post["nonzero"] * (a.bits if a.kernel_quantizer == "po2" else bb[0])
                 / 8 / 1024,
           "audit": post,
           # Only the po2 layers lose their multipliers.
           # This used to read `0 if kernel_quantizer == "po2"`, which reported DSP=0 for a
           # partially quantized model too -- with --po2-layers 0 the other three layers keep
           # quantized_bits kernels and take hls4ml's 'mult' branch. Count the nonzeros that
           # are still multiplies instead of asserting zero.
           "expected_dsp": _expected_dsp(post, po2_layers, a.kernel_quantizer),
           "why_dsp_zero": ("ExponentPrecisionType weights select nnet::product::"
                            "weight_exponential (a << w), so no multiplier is instantiated "
                            "-- hls4ml fpga_backend.py:504. Layers NOT in --po2-layers keep "
                            "quantized_bits kernels and DO instantiate multipliers; whether "
                            "those land in DSP or fabric is a synthesis question settled by "
                            "--force-lut, NOT by this field."),
           "metrics": score(a.detector, qat, cache, meta)}

    if a.baseline and os.path.exists(a.baseline):
        b = load_qmodel(a.baseline, compile=False)
        bnz = sum(int((np.abs(w) > 1e-7).sum()) for w in b.get_weights() if w.ndim == 2)
        res["baseline"] = {"model": a.baseline, "nonzero": bnz,
                           "kb": bnz * 12 / 8 / 1024,
                           "metrics": score(a.detector, b, cache, meta)}

    if a.save_model:
        qat.save(a.save_model)
        res["saved_model"] = a.save_model
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print("\n" + "=" * 70)
    # Label the arm, not a hardcoded "po2". This banner used to print `po2 bits=10` on every
    # run including the <12,2> control, which makes the two arms indistinguishable in the logs
    # afterwards -- exactly when you are trying to tell them apart.
    if a.kernel_quantizer == "po2":
        _arm = "po2 bits=%d%s" % (a.bits, (" layers=%s" % a.po2_layers) if po2_layers else
                                  " (all layers)")
    else:
        _arm = "<%d,%d> bits" % bb
    print("%-26s nonzero=%d  %.2f KB   DSP=%s   %s"
          % (_arm, res["nonzero"], res["kb"], res["expected_dsp"], res["metrics"]))
    if "baseline" in res:
        print("<12,2>          nonzero=%d  %.2f KB   %s"
              % (res["baseline"]["nonzero"], res["baseline"]["kb"],
                 res["baseline"]["metrics"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
