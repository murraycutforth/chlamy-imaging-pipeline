"""
compare_masking_methods.py
==========================
Compare multiple well-masking strategies across all NPY plates.

Usage
-----
    python scripts/compare_masking_methods.py

Outputs
-------
    output/mask_comparison/report.md               — human-readable summary table + notes
    output/mask_comparison/per_plate.csv           — per-plate × per-method statistics
                                                     (includes dark_threshold, light_threshold)
    output/mask_comparison/global_stats.csv        — one row per method, global aggregates
    output/mask_comparison/per_plate_thresholds.csv — per-plate blank-well noise statistics

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
# Each "fn" returns (mask, (dark_threshold, light_threshold))
# ---------------------------------------------------------------------------

METHODS = {
    "global_min_3s": {
        "fn": lambda arr: compute_threshold_mask_global(
            arr, num_std=3, time_reduction_fn=np.min, return_thresholds=True
        ),
        "label": "Global min 3σ",
        "description": (
            "Baseline. One dark and one light threshold are derived from the blank "
            "top-left well across all pooled dark/light frames. Each pixel's *minimum* "
            "value across all dark (resp. light) frames must exceed its threshold."
        ),
    },
    "global_mean_3s": {
        "fn": lambda arr: compute_threshold_mask_global(
            arr, num_std=3, time_reduction_fn=np.mean, return_thresholds=True
        ),
        "label": "Global mean 3σ",
        "description": (
            "Same global thresholds as baseline, but uses the *mean* across time "
            "instead of the minimum. Less aggressive: a pixel that exceeds the "
            "threshold on average is included even if it dips below at some timepoints."
        ),
    },
    "global_min_5s": {
        "fn": lambda arr: compute_threshold_mask_global(
            arr, num_std=5, time_reduction_fn=np.min, return_thresholds=True
        ),
        "label": "Global min 5σ",
        "description": (
            "Global threshold raised to 5σ above the blank mean. Otherwise identical "
            "to the baseline. Isolates the effect of σ from the per-frame calibration."
        ),
    },
    "per_ts_3s": {
        "fn": lambda arr: compute_threshold_mask_per_timestep(
            arr, num_std=3, return_thresholds=True
        ),
        "label": "Per-timestep 3σ",
        "description": (
            "Per-frame threshold: for each (dark, light) frame pair a threshold is "
            "computed from the blank well for *that specific frame*. A pixel must "
            "exceed the threshold at every single timestep (intersection). 3σ."
        ),
    },
    "per_ts_5s": {
        "fn": lambda arr: compute_threshold_mask_per_timestep(
            arr, num_std=5, return_thresholds=True
        ),
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


def blank_well_stats(arr: np.ndarray) -> dict:
    """Compute blank-well (top-left, i=0, j=0) noise statistics.

    Collects mean and std of dark and light frames separately from the blank
    well pixels pooled across the spatial dimensions.  These are the raw inputs
    to the threshold formula: threshold = mean + num_std * std.

    Also checks whether the top-left well is genuinely blank using the same
    heuristic as _compute_threshold (|global_mean - topleft_mean| / global_std <= 3).
    """
    NUM_TIMESTEPS = arr.shape[2]
    dark_idxs = list(range(0, NUM_TIMESTEPS, 2))
    light_idxs = list(range(1, NUM_TIMESTEPS, 2))

    blank_dark = arr[0, 0][dark_idxs]   # shape: (n_dark, h, w)
    blank_light = arr[0, 0][light_idxs]  # shape: (n_light, h, w)

    dark_mean = float(np.mean(blank_dark))
    dark_std = float(np.std(blank_dark))
    light_mean = float(np.mean(blank_light))
    light_std = float(np.std(blank_light))

    # Same heuristic used in _compute_threshold to detect non-blank top-left
    global_avg = float(np.mean(arr))
    global_std = float(np.std(arr))
    top_left_avg = float(np.mean(arr[0, 0]))
    is_blank = bool(abs(global_avg - top_left_avg) / global_std <= 3.0)

    return dict(
        blank_is_topleft=is_blank,
        blank_dark_mean=dark_mean,
        blank_dark_std=dark_std,
        blank_light_mean=light_mean,
        blank_light_std=light_std,
        # Coefficient of variation (noise relative to mean) — higher = noisier calibration
        blank_dark_cv=dark_std / dark_mean if dark_mean > 0 else float("nan"),
        blank_light_cv=light_std / light_mean if light_mean > 0 else float("nan"),
        n_dark_frames=len(dark_idxs),
        n_light_frames=len(light_idxs),
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

    # Accumulate per-plate blank-well stats (one row per plate)
    per_plate_threshold_rows = []

    # Accumulate global sizes per method
    global_sizes = {m: [] for m in METHODS}
    global_fails = {m: 0 for m in METHODS}   # plates that raised an exception

    probe_ynpq = {}  # method -> min Y(NPQ) on probe plate

    for path in npy_files:
        arr = np.load(path)
        stem = path.stem
        is_probe = stem == PROBE_PLATE

        # Blank-well stats are independent of masking method
        bws = blank_well_stats(arr)
        threshold_row = {"plate": stem, **bws}

        for method_key, method in METHODS.items():
            try:
                mask, (dark_threshold, light_threshold) = method["fn"](arr)
            except Exception as e:
                print(f"  ERROR  {stem}  [{method_key}]: {e}")
                global_fails[method_key] += 1
                continue

            stats = mask_stats(mask)
            global_sizes[method_key].extend(
                mask.reshape(arr.shape[0], arr.shape[1], -1).sum(axis=-1).ravel().tolist()
            )

            row = {
                "plate": stem,
                "method": method_key,
                "dark_threshold": float(dark_threshold),
                "light_threshold": float(light_threshold),
            }
            row.update(stats)
            per_plate_rows.append(row)

            # Store per-method thresholds in the threshold row too
            threshold_row[f"dark_threshold_{method_key}"] = float(dark_threshold)
            threshold_row[f"light_threshold_{method_key}"] = float(light_threshold)

            if is_probe:
                probe_ynpq[method_key] = ynpq_min_for_plate(arr, mask)

        per_plate_threshold_rows.append(threshold_row)

        if is_probe:
            print(f"Probe plate ({PROBE_PLATE}) Y(NPQ) min per method:")
            for m, v in probe_ynpq.items():
                status = "PASS" if v is not None and v > YNPQ_ASSERT_BOUND else "FAIL"
                print(f"  {m:<18} {f'{v:.4f}' if v is not None else 'exception':>10}  [{status}]")
            print()

    # -----------------------------------------------------------------------
    # Write per-plate CSV (now includes dark_threshold, light_threshold)
    # -----------------------------------------------------------------------
    per_plate_path = OUTPUT_DIR / "per_plate.csv"
    fieldnames = ["plate", "method", "dark_threshold", "light_threshold",
                  "n_total", "n_empty", "n_small", "n_nonempty",
                  "min", "p5", "p25", "p50", "p75", "p95", "max", "mean"]
    with open(per_plate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_plate_rows)
    print(f"Per-plate statistics written to {per_plate_path}")

    # -----------------------------------------------------------------------
    # Write per-plate threshold CSV
    # -----------------------------------------------------------------------
    threshold_fields = (
        ["plate", "blank_is_topleft",
         "blank_dark_mean", "blank_dark_std", "blank_dark_cv",
         "blank_light_mean", "blank_light_std", "blank_light_cv",
         "n_dark_frames", "n_light_frames"]
        + [f"dark_threshold_{m}" for m in METHODS]
        + [f"light_threshold_{m}" for m in METHODS]
    )
    thresholds_path = OUTPUT_DIR / "per_plate_thresholds.csv"
    with open(thresholds_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=threshold_fields)
        writer.writeheader()
        writer.writerows(per_plate_threshold_rows)
    print(f"Per-plate threshold statistics written to {thresholds_path}")

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
    # Build threshold analysis data for the report
    # (uses baseline method global_min_3s for n_small comparison)
    # -----------------------------------------------------------------------
    threshold_analysis = _compute_threshold_analysis(
        per_plate_rows, per_plate_threshold_rows
    )

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    _write_markdown_report(global_stats, probe_ynpq, threshold_analysis)

    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def _compute_threshold_analysis(per_plate_rows, per_plate_threshold_rows) -> dict:
    """Analyse whether blank-well noise or low plate-wide signal predicts small-mask count."""
    # Build lookup: plate -> stats for baseline method
    baseline = {}
    for row in per_plate_rows:
        if row["method"] == "global_min_3s":
            baseline[row["plate"]] = row

    # Collect aligned vectors
    plates = []
    dark_means, dark_stds, dark_cvs = [], [], []
    light_means, light_stds, light_cvs = [], [], []
    dark_thresholds, light_thresholds = [], []
    n_smalls = []
    mask_means, mask_p50s, mask_p25s, mask_p5s = [], [], [], []

    for row in per_plate_threshold_rows:
        p = row["plate"]
        if p not in baseline:
            continue
        b = baseline[p]
        plates.append(p)
        dark_means.append(row["blank_dark_mean"])
        dark_stds.append(row["blank_dark_std"])
        dark_cvs.append(row["blank_dark_cv"])
        light_means.append(row["blank_light_mean"])
        light_stds.append(row["blank_light_std"])
        light_cvs.append(row["blank_light_cv"])
        dark_thresholds.append(row["dark_threshold_global_min_3s"])
        light_thresholds.append(row["light_threshold_global_min_3s"])
        n_smalls.append(b["n_small"])
        mask_means.append(b["mean"])
        mask_p50s.append(b["p50"])
        mask_p25s.append(b["p25"])
        mask_p5s.append(b["p5"])

    dark_stds_arr = np.array(dark_stds)
    light_stds_arr = np.array(light_stds)
    dark_cvs_arr = np.array(dark_cvs)
    light_cvs_arr = np.array(light_cvs)
    dark_thresholds_arr = np.array(dark_thresholds)
    light_thresholds_arr = np.array(light_thresholds)
    n_smalls_arr = np.array(n_smalls)
    dark_means_arr = np.array(dark_means)
    light_means_arr = np.array(light_means)
    mask_means_arr = np.array(mask_means)
    mask_p50s_arr = np.array(mask_p50s)
    mask_p25s_arr = np.array(mask_p25s)
    mask_p5s_arr = np.array(mask_p5s)

    has_small = n_smalls_arr > 0

    def _pearsonr(x, y):
        """Pearson r without scipy."""
        xm = x - x.mean()
        ym = y - y.mean()
        denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
        return float(np.dot(xm, ym) / denom) if denom > 0 else 0.0

    # Top-20 plates by n_small (baseline) with their blank and mask-size stats
    order = np.argsort(n_smalls_arr)[::-1]
    top_n = min(20, len(plates))
    top_plates = [
        {
            "plate": plates[i],
            "n_small": int(n_smalls_arr[i]),
            "mask_p50": mask_p50s_arr[i],
            "mask_mean": mask_means_arr[i],
            "dark_mean": dark_means_arr[i],
            "dark_std": dark_stds_arr[i],
            "dark_cv": dark_cvs_arr[i],
            "dark_threshold": dark_thresholds_arr[i],
            "light_mean": light_means_arr[i],
            "light_std": light_stds_arr[i],
            "light_cv": light_cvs_arr[i],
            "light_threshold": light_thresholds_arr[i],
        }
        for i in order[:top_n]
    ]

    # Threshold distribution statistics
    def _pct_stats(arr):
        return dict(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            max=float(arr.max()),
        )

    return dict(
        n_plates=len(plates),
        n_with_small=int(has_small.sum()),
        n_without_small=int((~has_small).sum()),
        # Mean blank noise in each group
        mean_dark_std_with_small=float(dark_stds_arr[has_small].mean()) if has_small.any() else float("nan"),
        mean_dark_std_without_small=float(dark_stds_arr[~has_small].mean()) if (~has_small).any() else float("nan"),
        mean_light_std_with_small=float(light_stds_arr[has_small].mean()) if has_small.any() else float("nan"),
        mean_light_std_without_small=float(light_stds_arr[~has_small].mean()) if (~has_small).any() else float("nan"),
        mean_dark_cv_with_small=float(dark_cvs_arr[has_small].mean()) if has_small.any() else float("nan"),
        mean_dark_cv_without_small=float(dark_cvs_arr[~has_small].mean()) if (~has_small).any() else float("nan"),
        # Mean mask size in each group (signal proxy)
        mean_mask_p50_with_small=float(mask_p50s_arr[has_small].mean()) if has_small.any() else float("nan"),
        mean_mask_p50_without_small=float(mask_p50s_arr[~has_small].mean()) if (~has_small).any() else float("nan"),
        mean_mask_mean_with_small=float(mask_means_arr[has_small].mean()) if has_small.any() else float("nan"),
        mean_mask_mean_without_small=float(mask_means_arr[~has_small].mean()) if (~has_small).any() else float("nan"),
        # Pearson correlations
        r_dark_std_vs_n_small=_pearsonr(dark_stds_arr, n_smalls_arr),
        r_light_std_vs_n_small=_pearsonr(light_stds_arr, n_smalls_arr),
        r_dark_cv_vs_n_small=_pearsonr(dark_cvs_arr, n_smalls_arr),
        r_dark_threshold_vs_n_small=_pearsonr(dark_thresholds_arr, n_smalls_arr),
        r_light_threshold_vs_n_small=_pearsonr(light_thresholds_arr, n_smalls_arr),
        r_mask_mean_vs_n_small=_pearsonr(mask_means_arr, n_smalls_arr),
        r_mask_p50_vs_n_small=_pearsonr(mask_p50s_arr, n_smalls_arr),
        r_mask_p25_vs_n_small=_pearsonr(mask_p25s_arr, n_smalls_arr),
        r_mask_p5_vs_n_small=_pearsonr(mask_p5s_arr, n_smalls_arr),
        # Threshold distribution
        dark_threshold_stats=_pct_stats(dark_thresholds_arr),
        light_threshold_stats=_pct_stats(light_thresholds_arr),
        dark_std_stats=_pct_stats(dark_stds_arr),
        light_std_stats=_pct_stats(light_stds_arr),
        dark_mean_stats=_pct_stats(dark_means_arr),
        light_mean_stats=_pct_stats(light_means_arr),
        top_plates=top_plates,
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown_report(global_stats: dict, probe_ynpq: dict, threshold_analysis: dict):
    lines = []

    lines += [
        "# Well Masking Method Comparison",
        "",
        "Generated by `scripts/compare_masking_methods.py`.",
        "",
        "Each well is a 21×21 pixel sub-image (441 pixels total).  The mask selects pixels belonging to the algal colony; only masked pixels "
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

    # -----------------------------------------------------------------------
    # Threshold investigation section
    # -----------------------------------------------------------------------
    ta = threshold_analysis
    lines += [
        "",
        "---",
        "",
        "## Threshold investigation: blank-well noise and small masks",
        "",
        "The pipeline threshold formula (for the baseline `global_min_3s` method) is:",
        "",
        "```",
        "threshold = mean(blank_well) + 3 × std(blank_well)",
        "```",
        "",
        "where `blank_well` is the top-left well (i=0, j=0) pooled across all dark "
        "(resp. light) frames.  A well ends up with a small mask (1–2 pixels) when the "
        "threshold is high relative to the fluorescence signal in that well.  Two "
        "mechanisms can raise the threshold: a high blank *mean* (bright background) "
        "or a high blank *std* (noisy calibration).",
        "",
        "The data below examine whether anomalously large blank-well noise is the "
        f"primary driver of small-mask wells.  Analysis covers {ta['n_plates']} plates "
        "using the `global_min_3s` (baseline) method.",
        "",
        "### Threshold distribution across plates",
        "",
        "| | Dark threshold (ADU) | Light threshold (ADU) |",
        "|--|--|--|",
    ]

    def _fmt(s):
        return (
            f"mean {s['mean']:.1f} ± {s['std']:.1f}, "
            f"range [{s['min']:.0f}–{s['max']:.0f}], "
            f"p5/p50/p95 = {s['p5']:.0f}/{s['p50']:.0f}/{s['p95']:.0f}"
        )

    dt = ta["dark_threshold_stats"]
    lt = ta["light_threshold_stats"]
    lines += [
        f"| Mean ± std | {dt['mean']:.1f} ± {dt['std']:.1f} | {lt['mean']:.1f} ± {lt['std']:.1f} |",
        f"| p5 / p50 / p95 | {dt['p5']:.1f} / {dt['p50']:.1f} / {dt['p95']:.1f} | {lt['p5']:.1f} / {lt['p50']:.1f} / {lt['p95']:.1f} |",
        f"| Min / Max | {dt['min']:.1f} / {dt['max']:.1f} | {lt['min']:.1f} / {lt['max']:.1f} |",
        "",
        "Blank-well noise statistics:",
        "",
        "| | Dark blank std (ADU) | Light blank std (ADU) | Dark blank mean (ADU) | Light blank mean (ADU) |",
        "|--|--|--|--|--|",
    ]

    ds = ta["dark_std_stats"]
    ls = ta["light_std_stats"]
    dm = ta["dark_mean_stats"]
    lm = ta["light_mean_stats"]
    lines += [
        f"| Mean ± std | {ds['mean']:.2f} ± {ds['std']:.2f} | {ls['mean']:.2f} ± {ls['std']:.2f} | {dm['mean']:.1f} ± {dm['std']:.1f} | {lm['mean']:.1f} ± {lm['std']:.1f} |",
        f"| p5 / p50 / p95 | {ds['p5']:.2f} / {ds['p50']:.2f} / {ds['p95']:.2f} | {ls['p5']:.2f} / {ls['p50']:.2f} / {ls['p95']:.2f} | {dm['p5']:.1f} / {dm['p50']:.1f} / {dm['p95']:.1f} | {lm['p5']:.1f} / {lm['p50']:.1f} / {lm['p95']:.1f} |",
        f"| Min / Max | {ds['min']:.2f} / {ds['max']:.2f} | {ls['min']:.2f} / {ls['max']:.2f} | {dm['min']:.1f} / {dm['max']:.1f} | {lm['min']:.1f} / {lm['max']:.1f} |",
        "",
        "### Correlation of candidate drivers with small-mask count",
        "",
        f"Of {ta['n_plates']} plates: **{ta['n_with_small']}** have ≥1 small-mask well, "
        f"**{ta['n_without_small']}** have none.",
        "",
        "Two candidate drivers are tested: (1) noisy blank calibration raising the "
        "threshold, and (2) low plate-wide well signal.  The mask size distribution "
        "(mean, median) serves as a proxy for overall plate signal — larger masks "
        "indicate stronger fluorescence relative to background.",
        "",
        "| Variable | Pearson r with n_small (baseline) | Interpretation |",
        "|----------|----------------------------------|----------------|",
        f"| Blank dark std | {ta['r_dark_std_vs_n_small']:+.3f} | calibration noise |",
        f"| Blank light std | {ta['r_light_std_vs_n_small']:+.3f} | calibration noise |",
        f"| Blank dark CV (std/mean) | {ta['r_dark_cv_vs_n_small']:+.3f} | normalised calibration noise |",
        f"| Dark threshold | {ta['r_dark_threshold_vs_n_small']:+.3f} | combined noise+mean effect |",
        f"| Light threshold | {ta['r_light_threshold_vs_n_small']:+.3f} | combined noise+mean effect |",
        f"| Per-plate mean mask size | {ta['r_mask_mean_vs_n_small']:+.3f} | **plate-wide signal proxy** |",
        f"| Per-plate median mask size | {ta['r_mask_p50_vs_n_small']:+.3f} | **plate-wide signal proxy** |",
        f"| Per-plate p25 mask size | {ta['r_mask_p25_vs_n_small']:+.3f} | **plate-wide signal proxy** |",
        f"| Per-plate p5 mask size | {ta['r_mask_p5_vs_n_small']:+.3f} | **plate-wide signal proxy** |",
        "",
        "Mean values by group:",
        "",
        "| Group | n plates | Median mask size | Mean mask size | Dark blank std | Dark CV |",
        "|-------|----------|-----------------|----------------|----------------|---------|",
        f"| n_small > 0 | {ta['n_with_small']} "
        f"| {ta['mean_mask_p50_with_small']:.1f} "
        f"| {ta['mean_mask_mean_with_small']:.1f} "
        f"| {ta['mean_dark_std_with_small']:.2f} "
        f"| {ta['mean_dark_cv_with_small']:.3f} |",
        f"| n_small = 0 | {ta['n_without_small']} "
        f"| {ta['mean_mask_p50_without_small']:.1f} "
        f"| {ta['mean_mask_mean_without_small']:.1f} "
        f"| {ta['mean_dark_std_without_small']:.2f} "
        f"| {ta['mean_dark_cv_without_small']:.3f} |",
        "",
        "### Top plates by small-mask count",
        "",
        "| Plate | n_small | Plate median mask | Plate mean mask | Dark blank std | Dark CV | Dark threshold |",
        "|-------|---------|------------------|-----------------|----------------|---------|----------------|",
    ]
    for p in ta["top_plates"]:
        if p["n_small"] == 0:
            break
        lines.append(
            f"| {p['plate']} "
            f"| {p['n_small']} "
            f"| {p['mask_p50']:.0f} "
            f"| {p['mask_mean']:.1f} "
            f"| {p['dark_std']:.2f} "
            f"| {p['dark_cv']:.3f} "
            f"| {p['dark_threshold']:.1f} |"
        )

    lines += [
        "",
        "### Interpretation",
        "",
        "The blank-well noise correlations (dark std, dark CV) are near zero or "
        "slightly negative, ruling out noisy calibration as the primary driver.  "
        "The plate-wide signal proxies (median and mean mask size) show moderate "
        "negative correlations (r ≈ −0.19 to −0.23): plates where the typical well "
        "mask is smaller also tend to have more small-mask wells.  This is consistent "
        "with **low overall fluorescence signal** being the main cause — when "
        "background-subtracted cell intensities are close to the threshold across the "
        "whole plate, the weakest individual wells fall just below the cut-off.",
        "",
        "Two distinct regimes are visible in the top-plates table:",
        "",
        "* **Plate-level failure** (`31v2-M2_20h_HL`, median mask = 10, n_small = 55): "
        "the entire plate has severely depressed signal (likely failed inoculation or "
        "cell death), not a calibration problem (dark std = 2.47, below average).  "
        "All mask sizes shift downward, and many wells cross the small-mask threshold.",
        "",
        "* **Sparse outliers** (2–6 small masks, median mask ~ 13–35): these plates "
        "have moderately lower-than-average overall signal.  A small number of "
        "individual wells — likely edge wells, sparsely-seeded wells, or wells with "
        "partial cell loss — fall below the threshold while the rest of the plate is "
        "fine.",
        "",
        "The one genuine exception is `99v1-M8_1min-5min` (dark std = 11.83, CV = 1.45, "
        "threshold = 43.6 ADU — far above the p95 of ~21 ADU), where the anomalously "
        "noisy blank drives an unusually high threshold that causes 3 small-mask wells "
        "despite normal plate-wide signal.  This remains an isolated outlier.",
    ]

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
        "See `per_plate.csv`, `global_stats.csv`, and `per_plate_thresholds.csv` for full per-plate breakdowns.",
    ]

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
