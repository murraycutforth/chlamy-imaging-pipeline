"""
compare_masking_methods.py
==========================
Compare multiple well-masking strategies across all NPY plates.

Usage
-----
    python scripts/compare_masking_methods.py

Outputs
-------
    output/mask_comparison/report.md        — human-readable summary table + notes
    output/mask_comparison/per_plate.csv    — per-plate × per-method statistics
    output/mask_comparison/global_stats.csv — one row per method, global aggregates

Methods compared
----------------
  global_min_3s   : baseline — global threshold (dark/light frames pooled), np.min
                    across time, 3σ above blank mean
  global_mean_3s  : same thresholds, np.mean across time (larger masks)
  global_min_5s   : global threshold, np.min across time, 5σ (tighter)
  per_ts_3s       : per-frame threshold, intersection across all timesteps, 3σ
  per_ts_5s       : per-frame threshold, intersection across all timesteps, 5σ
"""

import csv
import logging
import warnings
from pathlib import Path

import numpy as np

from chlamy_impi.lib.mask_functions import (
    compute_threshold_mask_global,
    compute_threshold_mask_per_timestep,
    MIN_MASK_PIXELS,
)
from chlamy_impi.lib.npq_functions import compute_all_ynpq_averaged
from chlamy_impi.paths import WELL_SEGMENTATION_DIR

logging.basicConfig(level=logging.WARNING)

OUTPUT_DIR = Path("output/mask_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "global_min_3s": {
        "fn": lambda arr: compute_threshold_mask_global(arr, num_std=3, time_reduction_fn=np.min),
        "label": "Global min 3σ",
        "description": (
            "Baseline. One dark and one light threshold are derived from the blank "
            "top-left well across all pooled dark/light frames. Each pixel's *minimum* "
            "value across all dark (resp. light) frames must exceed its threshold."
        ),
    },
    "global_mean_3s": {
        "fn": lambda arr: compute_threshold_mask_global(arr, num_std=3, time_reduction_fn=np.mean),
        "label": "Global mean 3σ",
        "description": (
            "Same global thresholds as baseline, but uses the *mean* across time "
            "instead of the minimum. Less aggressive: a pixel that exceeds the "
            "threshold on average is included even if it dips below at some timepoints."
        ),
    },
    "global_min_5s": {
        "fn": lambda arr: compute_threshold_mask_global(arr, num_std=5, time_reduction_fn=np.min),
        "label": "Global min 5σ",
        "description": (
            "Global threshold raised to 5σ above the blank mean. Otherwise identical "
            "to the baseline. Isolates the effect of σ from the per-frame calibration."
        ),
    },
    "per_ts_3s": {
        "fn": lambda arr: compute_threshold_mask_per_timestep(arr, num_std=3),
        "label": "Per-timestep 3σ",
        "description": (
            "Per-frame threshold: for each (dark, light) frame pair a threshold is "
            "computed from the blank well for *that specific frame*. A pixel must "
            "exceed the threshold at every single timestep (intersection). 3σ."
        ),
    },
    "per_ts_5s": {
        "fn": lambda arr: compute_threshold_mask_per_timestep(arr, num_std=5),
        "label": "Per-timestep 5σ",
        "description": (
            "Same per-frame intersection logic, threshold raised to 5σ. "
            "Motivated by the observation that the initial Fm frame has a higher "
            "blank mean (~24 ADU) than subsequent Fm' frames (~14 ADU); the higher σ "
            "compensates for per-frame calibration yielding lower absolute thresholds "
            "on the initial Fm frame."
        ),
    },
}

# Plate used to verify that the Y(NPQ) blow-up is fixed
PROBE_PLATE = "20240520_99-M2_20h_HL"
YNPQ_ASSERT_BOUND = -2.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mask_stats(mask: np.ndarray) -> dict:
    """Return a dict of statistics for a single plate's mask array."""
    Ni, Nj = mask.shape[:2]
    sizes = mask.reshape(Ni, Nj, -1).sum(axis=-1).ravel()

    n_total = len(sizes)
    n_empty = int((sizes == 0).sum())
    n_small = int(((sizes > 0) & (sizes < MIN_MASK_PIXELS)).sum())
    nonempty = sizes[sizes > 0]

    if len(nonempty) == 0:
        return dict(n_total=n_total, n_empty=n_empty, n_small=n_small,
                    n_nonempty=0, min=0, p5=0, p25=0, p50=0, p75=0, p95=0,
                    max=0, mean=0.0)

    return dict(
        n_total=n_total,
        n_empty=n_empty,
        n_small=n_small,
        n_nonempty=len(nonempty),
        min=int(nonempty.min()),
        p5=float(np.percentile(nonempty, 5)),
        p25=float(np.percentile(nonempty, 25)),
        p50=float(np.percentile(nonempty, 50)),
        p75=float(np.percentile(nonempty, 75)),
        p95=float(np.percentile(nonempty, 95)),
        max=int(nonempty.max()),
        mean=float(nonempty.mean()),
    )


def ynpq_min_for_plate(arr: np.ndarray, mask: np.ndarray) -> float | None:
    """Return the minimum masked-mean Y(NPQ) across all wells and timesteps."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ynpq = compute_all_ynpq_averaged(arr.copy(), mask.copy())
            return float(np.nanmin(ynpq))
        except AssertionError:
            return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    npy_files = sorted(WELL_SEGMENTATION_DIR.glob("*.npy"))
    print(f"Found {len(npy_files)} plates.\n")

    # Accumulate per-plate rows
    per_plate_rows = []

    # Accumulate global sizes per method
    global_sizes = {m: [] for m in METHODS}
    global_fails = {m: 0 for m in METHODS}   # plates that raised an exception

    probe_ynpq = {}  # method -> min Y(NPQ) on probe plate

    for path in npy_files:
        arr = np.load(path)
        stem = path.stem
        is_probe = stem == PROBE_PLATE

        for method_key, method in METHODS.items():
            try:
                mask = method["fn"](arr)
            except Exception as e:
                print(f"  ERROR  {stem}  [{method_key}]: {e}")
                global_fails[method_key] += 1
                continue

            stats = mask_stats(mask)
            global_sizes[method_key].extend(
                mask.reshape(arr.shape[0], arr.shape[1], -1).sum(axis=-1).ravel().tolist()
            )

            row = {"plate": stem, "method": method_key}
            row.update(stats)
            per_plate_rows.append(row)

            if is_probe:
                probe_ynpq[method_key] = ynpq_min_for_plate(arr, mask)

        if is_probe:
            print(f"Probe plate ({PROBE_PLATE}) Y(NPQ) min per method:")
            for m, v in probe_ynpq.items():
                status = "PASS" if v is not None and v > YNPQ_ASSERT_BOUND else "FAIL"
                print(f"  {m:<18} {f'{v:.4f}' if v is not None else 'exception':>10}  [{status}]")
            print()

    # -----------------------------------------------------------------------
    # Write per-plate CSV
    # -----------------------------------------------------------------------
    per_plate_path = OUTPUT_DIR / "per_plate.csv"
    fieldnames = ["plate", "method", "n_total", "n_empty", "n_small", "n_nonempty",
                  "min", "p5", "p25", "p50", "p75", "p95", "max", "mean"]
    with open(per_plate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_plate_rows)
    print(f"Per-plate statistics written to {per_plate_path}")

    # -----------------------------------------------------------------------
    # Global statistics per method
    # -----------------------------------------------------------------------
    global_stats = {}
    total_wells = len(npy_files) * 384  # nominal; some plates are 336
    for method_key in METHODS:
        sizes = np.array(global_sizes[method_key])
        n_empty = int((sizes == 0).sum())
        n_small = int(((sizes > 0) & (sizes < MIN_MASK_PIXELS)).sum())
        nonempty = sizes[sizes > 0]
        global_stats[method_key] = dict(
            n_plates_ok=len(npy_files) - global_fails[method_key],
            n_plates_fail=global_fails[method_key],
            n_wells=len(sizes),
            n_empty=n_empty,
            pct_empty=100.0 * n_empty / len(sizes) if len(sizes) else 0,
            n_small=n_small,
            pct_small=100.0 * n_small / len(sizes) if len(sizes) else 0,
            n_nonempty=len(nonempty),
            min=int(nonempty.min()) if len(nonempty) else 0,
            p5=float(np.percentile(nonempty, 5)) if len(nonempty) else 0,
            p25=float(np.percentile(nonempty, 25)) if len(nonempty) else 0,
            p50=float(np.percentile(nonempty, 50)) if len(nonempty) else 0,
            p75=float(np.percentile(nonempty, 75)) if len(nonempty) else 0,
            p95=float(np.percentile(nonempty, 95)) if len(nonempty) else 0,
            max=int(nonempty.max()) if len(nonempty) else 0,
            mean=float(nonempty.mean()) if len(nonempty) else 0,
        )

    global_stats_path = OUTPUT_DIR / "global_stats.csv"
    gs_fields = ["method", "n_plates_ok", "n_plates_fail", "n_wells",
                 "n_empty", "pct_empty", "n_small", "pct_small",
                 "n_nonempty", "min", "p5", "p25", "p50", "p75", "p95", "max", "mean"]
    with open(global_stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gs_fields)
        writer.writeheader()
        for mk, gs in global_stats.items():
            writer.writerow({"method": mk, **gs})
    print(f"Global statistics written to {global_stats_path}")

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    _write_markdown_report(global_stats, probe_ynpq)

    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


def _write_markdown_report(global_stats: dict, probe_ynpq: dict):
    lines = []

    lines += [
        "# Well Masking Method Comparison",
        "",
        "Generated by `scripts/compare_masking_methods.py`.",
        "",
        "Each well is a 21×21 pixel sub-image (441 pixels total).  "
        "The mask selects pixels belonging to the algal colony; only masked pixels "
        "contribute to per-well averages of Fv/Fm, Y(II), and Y(NPQ).",
        "",
        "---",
        "",
        "## Methods",
        "",
    ]
    for mk, m in METHODS.items():
        lines += [f"### `{mk}` — {m['label']}", "", m["description"], ""]

    lines += [
        "---",
        "",
        "## Global mask-size statistics (non-empty wells only)",
        "",
        "Well size: 21 × 21 = 441 pixels.  "
        f"Minimum meaningful mask: {MIN_MASK_PIXELS} px (`MIN_MASK_PIXELS`).",
        "",
        "| Method | Plates OK | Empty wells | < 3 px wells | "
        "Min | p5 | p25 | p50 | p75 | p95 | Max | Mean |",
        "|--------|-----------|-------------|--------------|"
        "-----|-----|-----|-----|-----|-----|-----|------|",
    ]

    for mk, gs in global_stats.items():
        label = METHODS[mk]["label"]
        lines.append(
            f"| {label} "
            f"| {gs['n_plates_ok']} "
            f"| {gs['n_empty']} ({gs['pct_empty']:.1f}%) "
            f"| {gs['n_small']} ({gs['pct_small']:.2f}%) "
            f"| {gs['min']} "
            f"| {gs['p5']:.0f} "
            f"| {gs['p25']:.0f} "
            f"| {gs['p50']:.0f} "
            f"| {gs['p75']:.0f} "
            f"| {gs['p95']:.0f} "
            f"| {gs['max']} "
            f"| {gs['mean']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        f"## Y(NPQ) validation on probe plate `{PROBE_PLATE}`",
        "",
        "This plate triggers a Y(NPQ) < −2 assertion failure under the baseline method "
        "because two edge pixels in well (15,2) have initial Fm ≈ background after "
        "background subtraction (~19 and ~26 ADU), yet have F and Fm' values above "
        "400 ADU at later timepoints.  Dividing by a near-zero Fm produces Y(NPQ) ≈ −21 "
        "for those pixels, dragging the masked well mean to −3.68.",
        "",
        f"| Method | Min Y(NPQ) (all wells, all timesteps) | Passes (> {YNPQ_ASSERT_BOUND}) |",
        "|--------|---------------------------------------|---------|",
    ]
    for mk, v in probe_ynpq.items():
        label = METHODS[mk]["label"]
        if v is None:
            lines.append(f"| {label} | exception during computation | — |")
        else:
            passes = "✓" if v > YNPQ_ASSERT_BOUND else "✗"
            lines.append(f"| {label} | {v:.4f} | {passes} |")

    lines += [
        "",
        "---",
        "",
        "## Notes on method selection",
        "",
        "* **Global mean 3σ** produces the most inclusive masks (largest mean size, "
        "fewest empty wells) but does not fix the Y(NPQ) blow-up: a pixel that is "
        "near-zero at t=0 (Fm) but high at all other timepoints passes easily on its mean.",
        "* **Global min 5σ** fixes the probe-plate failure and reduces mask sizes "
        "substantially.  The threshold is calibrated from all light frames pooled "
        "(blank mean ~14–15 ADU for the many Fm' frames, ~24 ADU for the single Fm "
        "frame), so the resulting threshold is lower than a per-Fm-frame calibration "
        "would yield.  For the probe plate this is sufficient, but in plates with a "
        "lower signal-to-noise ratio the globally diluted threshold may admit more "
        "near-background Fm pixels than per-frame calibration would.",
        "* **Per-timestep 3σ** raises the effective threshold on the initial Fm frame "
        "(calibrated from the blank's Fm-frame intensity, ~24 ADU, vs. ~14 ADU for Fm' "
        "frames), fixing one of the two problematic pixels for the probe plate but not "
        "both.  Not sufficient on its own.",
        "* **Per-timestep 5σ** also fixes the probe plate and is the most principled "
        "choice: each frame's threshold is calibrated to that frame's own blank "
        "statistics, making the mask robust to frame-to-frame variation in background "
        "intensity.  Mask sizes are comparable to global min 5σ (mean ~31 vs ~31).",
        "* The `< 3 px wells` column shows how many non-empty wells fall below the "
        f"minimum meaningful mask size ({MIN_MASK_PIXELS} px).  These wells would "
        "yield statistics dominated by shot noise from 1–2 pixels.  Both 5σ methods "
        "have more such wells (~240–270) than the 3σ methods (~120–160) due to their "
        "tighter thresholds.",
        "",
        "See `per_plate.csv` and `global_stats.csv` for full per-plate breakdowns.",
    ]

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
