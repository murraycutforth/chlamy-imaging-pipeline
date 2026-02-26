"""Visualise Y(II) and Y(NPQ) time series from the Stage 2b database.

Called at the end of Stage 2b as a sanity check.  Produces two mosaic PNG
files in the database output directory, one per metric, with one subplot per
light regime.

Each subplot shows:
  - A random sample of individual well traces (for visual texture)
  - Population percentile bands: 5–95th (light fill) and IQR (darker fill)
  - Median line
"""

import logging
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chlamy_impi.paths import get_database_output_dir

logger = logging.getLogger(__name__)

_RNG = np.random.default_rng(42)
_SAMPLE_N = 300  # max individual well traces per subplot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return columns matching ^{prefix}_\\d+$, sorted numerically by suffix."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matches = [(c, int(m.group(1))) for c in df.columns if (m := pat.match(c))]
    return [c for c, _ in sorted(matches, key=lambda x: x[1])]


def _elapsed_hours(
    subset: pd.DataFrame, mt_cols: list[str], val_cols: list[str]
) -> np.ndarray:
    """Return elapsed-time matrix of shape (n_wells, len(val_cols)) in hours.

    Each y2_k / ynpq_k column maps to measurement_time_k.
    Elapsed = measurement_time_k − measurement_time_0.  NaT → NaN.
    """
    mt_df = subset[mt_cols]
    t0 = mt_df.iloc[:, 0]
    elapsed = mt_df.subtract(t0, axis=0)
    elapsed_h = elapsed.apply(lambda col: col.dt.total_seconds() / 3600)

    mt_idx = {col: i for i, col in enumerate(mt_cols)}
    step_indices = [
        mt_idx[f"measurement_time_{c.rsplit('_', 1)[1]}"] for c in val_cols
    ]
    return elapsed_h.values[:, step_indices]  # (n_wells, n_steps)


def _draw_subplot(
    ax: plt.Axes,
    subset: pd.DataFrame,
    val_cols: list[str],
    mt_cols: list[str],
    color: str,
    metric_label: str,
    first: bool,
) -> None:
    n_wells = len(subset)
    if n_wells == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    try:
        time_h = _elapsed_hours(subset, mt_cols, val_cols)
    except KeyError as exc:
        ax.text(
            0.5, 0.5, f"time error:\n{exc}",
            transform=ax.transAxes, ha="center", fontsize=7,
        )
        return

    vals = subset[val_cols].values.astype(float)  # (n_wells, n_steps)

    # Individual traces from a random sample
    idx = _RNG.choice(n_wells, size=min(_SAMPLE_N, n_wells), replace=False)
    for i in idx:
        t, v = time_h[i], vals[i]
        ok = np.isfinite(t) & np.isfinite(v)
        if ok.sum() > 1:
            ax.plot(t[ok], v[ok], color=color, alpha=0.05,
                    linewidth=0.5, rasterized=True)

    # Percentile bands over the full population
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p05 = np.nanpercentile(vals, 5, axis=0)
        p25 = np.nanpercentile(vals, 25, axis=0)
        p50 = np.nanpercentile(vals, 50, axis=0)
        p75 = np.nanpercentile(vals, 75, axis=0)
        p95 = np.nanpercentile(vals, 95, axis=0)

    t_med = np.nanmedian(time_h, axis=0)
    ok = np.isfinite(t_med)
    ax.fill_between(t_med[ok], p05[ok], p95[ok], color=color, alpha=0.12,
                    label="5–95th pct")
    ax.fill_between(t_med[ok], p25[ok], p75[ok], color=color, alpha=0.25,
                    label="IQR")
    ax.plot(t_med[ok], p50[ok], color=color, linewidth=1.5, label="median")

    ax.set_xlabel("Time (hours)", fontsize=8)
    ax.set_ylabel(metric_label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if first:
        ax.legend(fontsize=7, loc="upper right")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_timeseries_mosaic(df: pd.DataFrame) -> None:
    """Write timeseries_y2.png and timeseries_ynpq.png to the database output dir.

    One subplot per light regime; individual well traces plus population
    percentile bands (5–95th, IQR, median).
    """
    output_dir = get_database_output_dir()

    y2_cols = _sorted_cols(df, "y2")
    ynpq_cols = _sorted_cols(df, "ynpq")
    mt_cols = _sorted_cols(df, "measurement_time")

    if not y2_cols or not mt_cols:
        logger.warning("Missing y2_* or measurement_time_* columns — skipping plots")
        return

    light_regimes = sorted(df["light_regime"].dropna().unique())
    if not light_regimes:
        logger.warning("No light_regime values — skipping plots")
        return

    n = len(light_regimes)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    for metric, val_cols, color, fname in [
        ("Y(II)",  y2_cols,   "steelblue",  "timeseries_y2.png"),
        ("Y(NPQ)", ynpq_cols, "darkorange", "timeseries_ynpq.png"),
    ]:
        if not val_cols:
            logger.warning(f"No {metric} columns found — skipping {fname}")
            continue

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"{metric} time series by light regime  (n = {len(df):,} wells)",
            fontsize=13,
        )

        for idx, regime in enumerate(light_regimes):
            ax = axes[idx // ncols][idx % ncols]
            subset = df[df["light_regime"] == regime].reset_index(drop=True)
            ax.set_title(f"{regime}  (n = {len(subset):,})", fontsize=9)
            _draw_subplot(ax, subset, val_cols, mt_cols, color, metric, first=(idx == 0))

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        out = output_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out}")
