#!/usr/bin/env python
"""Synthesize the compressed students with hls4ml and report FPGA resources.

    python hls4ml_sweep.py --models-dir <dir> --out hls4ml_results.json

What this answers. Not "how many LUTs" in absolute terms -- that is device- and
config-dependent and means little on its own. The comparisons are the point, and they hold at
any fixed configuration:

  * pruned vs unpruned at identical settings -- what pruning bought in silicon rather than in
    stored bytes. Hls4ml does not skip zero weights by default. If a sparse and a dense
    model come out identical, that is expected behavior and is itself the finding: pruning
    saved storage but not logic unless the tool is told to exploit sparsity.
  * narrower vs wider fixed point at identical sparsity -- the bit-width trade in LUTs rather
    than in kilobytes.
  * Measured latency against the target clock period.

The three settings that decide every number. Set them deliberately and quote them beside any
figure taken from this script:

  --part        which FPGA. Changes DSP availability, and therefore how many multiplies spill
                into LUT fabric. Can move LUT counts by an order of magnitude.
  --clock       target period in ns. A tighter clock means more pipelining, so more FFs and LUTs.
  --reuse       multiplies sharing one multiplier. ReuseFactor 1 is fastest and largest.
  --force-lut   put all multiplies in LUT fabric instead of DSP blocks.

--Force-lut changes what the DSP column means. With it, DSP = 0 holds by construction
for any model, whatever its quantization. Without it, hls4ml places multiplies in DSP blocks by
default. A DSP count is only evidence about the quantizer if this flag is off.

Start with --dry-run to see the plan and the estimated wall time before committing.
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=os.path.expanduser("~/hls4ml_models"))
    p.add_argument("--out", default="hls4ml_results.json")
    p.add_argument("--work-dir", default="/tmp/hls4ml_sweep")
    # --- the settings that determine every number ---
    p.add_argument("--part", default="xcvu13p-flga2577-2-e",
                   help="FPGA part. State this ON any slide.")
    p.add_argument("--clock", type=float, default=5.0,
                   help="clock period in ns (5.0 = 200 MHz)")
    p.add_argument("--reuse", type=int, default=1,
                   help="ReuseFactor: multiplies sharing one multiplier")
    p.add_argument("--force-lut", action="store_true",
                   help="multipliers in LUT fabric, not DSPs. Injects "
                        "`config_op mul -impl fabric` into the generated build_prj.tcl -- "
                        "the hls4ml config key alone does NOT work, see inject_fabric_directive().")
    p.add_argument("--precision-override", default="",
                   help="DIAGNOSTIC ONLY, e.g. '10,4'. Overrides every layer's precision to "
                        "ap_fixed<W,I>, DIVORCING the synthesised precision from the "
                        "precision the model was QAT-trained at. The physics numbers are then "
                        "MEANINGLESS -- use this only to answer resource questions such as "
                        "'does a narrower operand stop Vitis reaching for a DSP?'. Every "
                        "affected row is stamped diagnostic_only=true in the output JSON.")
    p.add_argument("--strategy", default="Latency", choices=["Latency", "Resource"])
    p.add_argument("--reference-preset", action="store_true",
                   help="published-baseline operating point: 25 ns clock, LUT-fabric "
                        "multiplies (DSP=0), ReuseFactor 1. See docstring.")
    # --- which models ---
    p.add_argument("--only", default="", help="substring filter, e.g. 'dch' or 'sp50'")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def describe(name):
    """dch_sp50_12b2i.h5 -> ('dch', 50, 12, 2)

    Tokens are matched, not positional. The old version indexed parts[1]/parts[2], which
    crashed outright on our own archived filenames (`dch_model_sp50_12b2i.h5` -- one extra
    token) and would silently mis-parse anything else with a prefix. Unrecognised fields come
    back None rather than raising: the sparsity/bits here are only labels for the output
    table, and losing a label is not a reason to abandon an hour of synthesis.
    """
    import re
    b = os.path.basename(name)
    if b.endswith(".h5"):
        b = b[:-3]
    parts = b.split("_")
    det = parts[0]
    sp = bits = intb = None
    for tok in parts[1:]:
        m = re.fullmatch(r"sp(\d+)", tok)
        if m and sp is None:
            sp = int(m.group(1))
            continue
        m = re.fullmatch(r"(\d+)b(\d+)i", tok)
        if m and bits is None:
            bits, intb = int(m.group(1)), int(m.group(2))
    return det, sp, bits, intb


# Vitis hls directive that binds every multiply to LUT fabric instead of a DSP48.
# UG1399: `config_op <op> -impl <auto|fabric|dsp|meddsp|fulldsp|maxdsp>`.
FABRIC_DIRECTIVE = "config_op mul -impl fabric"
# Anchor line in hls4ml/templates/vitis/build_prj.tcl (verified against 1.3.0). Everything
# from `open_solution` to `csynth_design` is solution configuration, and the directive has to
# land inside that window. `create_clock` is the last config line before the csim/csynth
# blocks, so inserting immediately after it is both inside the window and stable across the
# reset/no-reset branches above it.
FABRIC_ANCHOR = "create_clock -period $clock_period -name default"


def inject_fabric_directive(odir):
    """Force LUT-fabric multipliers by editing the generated build_prj.tcl.

    The hls4ml config key alone does not work. Setting cfg["Model"]["DSPUsage"] = 0
    is silently ignored by the Vitis backend, and synthesis still returns thousands of DSPs.
    The binding is a synthesis decision, so it has to be made in Tcl rather than in the hls4ml
    config dict.

    safe to do here: ModelGraph.compile() -> write() copies the template to <odir>, and
    VitisBackend.build() writes only build_opt.tcl before running
    `vitis-run --tcl build_prj.tcl`. It never regenerates build_prj.tcl, so an edit made
    between compile() and build() survives. Verified against hls4ml 1.3.0
    (vitis_writer.py:62-64, vitis_backend.py:139-144).

    Raises rather than warns: a silently-unpatched project would synthesise with DSPs and
    report a number that looks like a valid DSP=0 attempt. That failure mode already cost us
    one 66-minute run.
    """
    tcl = os.path.join(odir, "build_prj.tcl")
    if not os.path.exists(tcl):
        raise RuntimeError("no build_prj.tcl in %s -- call compile() before patching" % odir)
    with open(tcl) as fh:
        txt = fh.read()
    if FABRIC_DIRECTIVE in txt:
        return "already-present"
    if FABRIC_ANCHOR not in txt:
        raise RuntimeError(
            "anchor %r not found in %s -- hls4ml changed its Vitis template; re-check where "
            "solution config ends and csynth_design begins before trusting any DSP number."
            % (FABRIC_ANCHOR, tcl))
    txt = txt.replace(FABRIC_ANCHOR,
                      FABRIC_ANCHOR + "\n# injected by scripts/hls4ml_sweep.py --force-lut\n"
                      + FABRIC_DIRECTIVE, 1)
    with open(tcl, "w") as fh:
        fh.write(txt)
    # read back: the whole point is that we must not proceed on an assumption
    with open(tcl) as fh:
        if FABRIC_DIRECTIVE not in fh.read():
            raise RuntimeError("patch of %s did not stick" % tcl)
    return "injected"


def apply_precision_override(cfg, w_bits, i_bits):
    """diagnostic: rewrite every precision string to ap_fixed<w,i>. Returns n layers touched.

    This deliberately breaks the invariant the rest of the script protects -- that the
    synthesised precision equals the precision the accuracy was validated at. It exists for
    one question: does narrowing the operands change whether Vitis binds a multiply to a DSP?
    Answering that with a real <10,x> model costs a QAT retrain; answering it this way costs
    one synthesis run. Any row produced this way is stamped diagnostic_only and its
    LUT/DSP counts describe a model whose physics numbers we have not measured.
    """
    typ = "ap_fixed<%d,%d>" % (w_bits, i_bits)
    n = 0
    for scope in ("Model", ):
        prec = cfg.get(scope, {}).get("Precision")
        if isinstance(prec, str):
            cfg[scope]["Precision"] = typ
            n += 1
        elif isinstance(prec, dict):
            for k in prec:
                prec[k] = typ
                n += 1
    for lname, lcfg in (cfg.get("LayerName") or {}).items():
        prec = lcfg.get("Precision")
        if isinstance(prec, str):
            lcfg["Precision"] = typ
            n += 1
        elif isinstance(prec, dict):
            for k in prec:
                prec[k] = typ
                n += 1
    if n == 0:
        raise RuntimeError("precision override touched 0 fields -- config layout changed; "
                           "do not trust the resulting numbers")
    return n


def main():
    a = parse_args()
    if a.reference_preset:
        # Operating point taken from the published baseline this work compares against.
        #
        # clock. That baseline reports a fixed 25 ns latency across operating points while the
        # achieved frequency varies. A fixed latency with a varying achieved clock means N
        # cycles at a fixed target period, and only 25 ns x 1 cycle is self-consistent: a
        # 12.5 ns x 2-cycle target would fail timing at the reported frequencies.
        #
        # DSP = 0 by forcing fabric multiplies. This also makes the chosen part largely
        # irrelevant, since device choice mattered mainly by setting how many multiplies spill
        # from DSPs into fabric; here everything is fabric by construction. The part need only
        # be large enough to hold the design.
        a.clock, a.force_lut, a.reuse = 25.0, True, 1
        print("[reference preset] 25 ns clock, DSP=0 (LUT fabric), ReuseFactor 1")
        print("[reference preset] NOTE the bit widths here need not match the published "
              "baseline's. Check before quoting the two side by side.")
    prec_override = None
    if a.precision_override:
        try:
            _w, _i = [int(v) for v in a.precision_override.split(",")]
        except ValueError:
            sys.exit("--precision-override wants 'W,I', e.g. '10,4'")
        if _i >= _w:
            sys.exit("--precision-override: integer bits (%d) must be < total bits (%d)"
                     % (_i, _w))
        prec_override = (_w, _i)

    files = sorted(f for f in os.listdir(a.models_dir) if f.endswith(".h5"))
    if a.only:
        files = [f for f in files if a.only in f]
    if not files:
        sys.exit("no .h5 models matched in %s" % a.models_dir)

    print("=" * 78)
    print("hls4ml sweep -- %d model(s)" % len(files))
    print("=" * 78)
    print("  part      : %s" % a.part)
    print("  clock     : %.2f ns  (%.0f MHz)" % (a.clock, 1000.0 / a.clock))
    print("  strategy  : %s   ReuseFactor %d" % (a.strategy, a.reuse))
    print("  multiplies: %s" % ("LUT FABRIC via `%s`" % FABRIC_DIRECTIVE
                                if a.force_lut else "DSP blocks (hls4ml default)"))
    if prec_override:
        print("  precision : Diagnostic override ap_fixed<%d,%d> -- NOT the QAT "
              "precision; resource numbers only, physics numbers INVALID" % prec_override)
    else:
        print("  precision : from the QKeras quantizers (matches the validated accuracy)")
    print("  est. wall : ~%d-%d min" % (len(files) * 2, len(files) * 6))
    for f in files:
        det, sp, bits, ib = describe(f)
        print("    %-26s %-4s sparsity %-4s <%s,%s>" % (f, det, sp, bits, ib))
    if a.dry_run:
        print("\n--dry-run: nothing synthesised.")
        return

    import hls4ml
    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects

    co = {}
    _add_supported_quantized_objects(co)
    os.makedirs(a.work_dir, exist_ok=True)
    results, t_all = [], time.time()

    for i, f in enumerate(files, 1):
        det, sp, bits, ib = describe(f)
        tag = f.replace(".h5", "")
        print("\n" + "-" * 78)
        print("[%d/%d] %s" % (i, len(files), tag), flush=True)
        t0 = time.time()
        try:
            # compile=False: these were fine-tuned with a custom loss closure that does not
            # survive deserialisation, and we only need the graph + weights.
            model = load_model(os.path.join(a.models_dir, f), custom_objects=co, compile=False)

            cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
            cfg["Model"]["Strategy"] = a.strategy
            cfg["Model"]["ReuseFactor"] = a.reuse
            for lname in cfg.get("LayerName", {}):
                cfg["LayerName"][lname]["ReuseFactor"] = a.reuse
                if a.force_lut:
                    # keep every multiply out of the DSP blocks
                    cfg["LayerName"][lname]["Strategy"] = a.strategy
            if a.force_lut:
                cfg["Model"]["DSPUsage"] = 0          # ignored by Vitis; the Tcl does the work
            # nb: precision is normally taken from the qkeras quantizers by
            # config_from_keras_model -- do not override it, or the synthesised precision
            # stops matching the precision the accuracy numbers were validated at. The one
            # exception is the explicitly-labeled --precision-override diagnostic below.
            if prec_override:
                nfields = apply_precision_override(cfg, *prec_override)
                print("   [DIAGNOSTIC] precision forced to ap_fixed<%d,%d> across %d fields "
                      "-- physics numbers for this row are INVALID"
                      % (prec_override[0], prec_override[1], nfields), flush=True)

            odir = os.path.join(a.work_dir, tag)
            hm = hls4ml.converters.convert_from_keras_model(
                model, hls_config=cfg, output_dir=odir,
                part=a.part, clock_period=a.clock, backend="Vitis")
            hm.compile()          # writes the project, incl. build_prj.tcl from the template
            if a.force_lut:
                # after compile(), before build() -- see inject_fabric_directive() docstring.
                print("   [force-lut] %s -> %s" % (inject_fabric_directive(odir),
                                                   FABRIC_DIRECTIVE), flush=True)
            rep = hm.build(csim=False, synth=True, vsynth=False)
            cs = (rep or {}).get("CSynthesisReport", {}) or {}

            def num(k):
                v = cs.get(k)
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None

            lat = num("WorstLatency")
            row = {"model": tag, "detector": det, "sparsity": sp,
                   "bits": bits, "int_bits": ib,
                   "LUT": num("LUT"), "FF": num("FF"), "DSP": num("DSP"),
                   "BRAM_18K": num("BRAM_18K"),
                   "latency_cycles": lat,
                   "latency_ns": (lat * a.clock if lat is not None else None),
                   "II": cs.get("IntervalMax"),
                   "est_clock_ns": cs.get("EstimatedClockPeriod"),
                   "secs": round(time.time() - t0, 1)}
            # Stamp provenance on the row, not only in the file header -- rows get copied
            # into tables and docs one at a time, and a diagnostic row that loses its label
            # becomes a fabricated result.
            row["multiplies_in"] = "LUT" if a.force_lut else "DSP"
            if prec_override:
                row["diagnostic_only"] = True
                row["synth_precision"] = "ap_fixed<%d,%d>" % prec_override
                row["qat_precision"] = ("<%d,%d>" % (bits, ib)
                                        if bits is not None else "unknown")
            results.append(row)
            print("   LUT %-8s FF %-7s DSP %-6s BRAM %-4s  latency %s cyc (%.1f ns)  [%.0fs]"
                  % (row["LUT"], row["FF"], row["DSP"], row["BRAM_18K"],
                     row["latency_cycles"],
                     row["latency_ns"] if row["latency_ns"] is not None else float("nan"),
                     row["secs"]), flush=True)
        except Exception as e:                                    # noqa: BLE001
            print("   FAILED: %s" % e, flush=True)
            results.append({"model": tag, "detector": det, "sparsity": sp,
                            "bits": bits, "int_bits": ib, "error": str(e)})

        json.dump({"part": a.part, "clock_ns": a.clock, "reuse_factor": a.reuse,
                   "strategy": a.strategy, "multiplies_in": ("LUT" if a.force_lut else "DSP"),
                   "fabric_directive": (FABRIC_DIRECTIVE if a.force_lut else None),
                   "precision_override": ("ap_fixed<%d,%d>" % prec_override
                                          if prec_override else None),
                   "diagnostic_only": bool(prec_override),
                   "results": results}, open(a.out, "w"), indent=2)

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print("SUMMARY   part=%s  %.0f MHz  RF=%d  mult=%s"
          % (a.part, 1000.0 / a.clock, a.reuse, "LUT" if a.force_lut else "DSP"))
    print("=" * 78)
    print("%-24s %-9s %-8s %-7s %-7s %-10s" % ("model", "LUT", "FF", "DSP", "BRAM", "latency"))
    for r in results:
        if "error" in r:
            print("%-24s FAILED" % r["model"]); continue
        print("%-24s %-9s %-8s %-7s %-7s %s ns"
              % (r["model"], r["LUT"], r["FF"], r["DSP"], r["BRAM_18K"],
                 ("%.1f" % r["latency_ns"]) if r["latency_ns"] is not None else "?"))

    # the comparisons that are actually interesting
    ok = [r for r in results if "error" not in r and r.get("LUT")]
    for det in sorted({r["detector"] for r in ok}):
        d = {(r["sparsity"], r["bits"]): r for r in ok if r["detector"] == det}
        base = d.get((0, 12))
        if base:
            print("\n%s -- what PRUNING bought at <12,2> (vs 0%% sparse):" % det.upper())
            for sp in sorted({k[0] for k in d if k[1] == 12}):
                r = d.get((sp, 12))
                if r:
                    print("   %3d%%  LUT %-8s (%+.1f%%)   DSP %-6s   %.1f ns"
                          % (sp, r["LUT"], 100.0 * (r["LUT"] - base["LUT"]) / base["LUT"],
                             r["DSP"], r["latency_ns"] or float("nan")))
        pairs = [(sp, d.get((sp, 12)), d.get((sp, 16)))
                 for sp in sorted({k[0] for k in d}) if d.get((sp, 12)) and d.get((sp, 16))]
        if pairs:
            print("\n%s -- 12-bit vs 16-bit at equal sparsity:" % det.upper())
            for sp, a12, a16 in pairs:
                print("   %3d%%  LUT %-8s vs %-8s (%+.1f%%)   DSP %-5s vs %-5s"
                      % (sp, a12["LUT"], a16["LUT"],
                         100.0 * (a12["LUT"] - a16["LUT"]) / a16["LUT"],
                         a12["DSP"], a16["DSP"]))

    print("\nwrote %s   (total %.1f min)" % (a.out, (time.time() - t_all) / 60))
    print("\nREMINDER: quote LUT/DSP numbers ONLY alongside the part, clock and reuse factor "
          "printed above. Without them the figures are not interpretable.")


if __name__ == "__main__":
    main()
