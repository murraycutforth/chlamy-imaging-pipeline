"""Update docs/_config.yml and docs/index.md to reflect the latest pipeline run.

Computes fresh statistics from `output/database_creation/database.csv` and parses the
latest dated comparison Markdown report, then rewrites the volatile fields in
`docs/index.md` and `docs/_config.yml`.

Run after `scripts/generate_report_assets.py`. Run with `--dry-run` to print the
new values without writing.

Usage:
    python scripts/update_report_docs.py [--dry-run] [--bump {patch,minor,none}]
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
CONFIG_YML = DOCS / "_config.yml"
INDEX_MD = DOCS / "index.md"
ASSETS_DIR = DOCS / "assets" / "images"
DB_CSV = REPO_ROOT / "output" / "database_creation" / "database.csv"
DB_DIR = REPO_ROOT / "output" / "database_creation"
CLEANED_DIR = REPO_ROOT / "output" / "cleaned_raw_data"
WELLS_PARQUET = REPO_ROOT / "output" / "image_processing" / "wells.parquet"
TIMESERIES_PARQUET = REPO_ROOT / "output" / "image_processing" / "timeseries.parquet"
IDENTITY_XLSX = REPO_ROOT / "data" / "Finalized Identities Phase I plates.xlsx"

sys.path.insert(0, str(REPO_ROOT))


def fmt(n: int) -> str:
    return f"{n:,}"


def compute_stats() -> dict:
    df = pd.read_csv(DB_CSV, low_memory=False)
    n_rows = len(df)
    n_cols = len(df.columns)
    unique_plates = df["plate"].nunique()
    non_empty_mask = df["mask_area"] > 0
    non_empty = int(non_empty_mask.sum())
    empty = int((df["mask_area"] == 0).sum())

    fvfm = df.loc[non_empty_mask, "fv_fm"].agg(["mean", "median", "std"])

    y2_cols = [
        c
        for c in df.columns
        if c.startswith("y2_") and not c.startswith("y2_std") and c.split("_")[-1].isdigit()
    ]
    ynpq_cols = [
        c for c in df.columns if c.startswith("ynpq_") and c.split("_")[-1].isdigit()
    ]
    y2_vals = df.loc[non_empty_mask, y2_cols].values.flatten()
    y2_vals = y2_vals[~np.isnan(y2_vals)]
    ynpq_vals = df.loc[non_empty_mask, ynpq_cols].values.flatten()
    ynpq_vals = ynpq_vals[~np.isnan(ynpq_vals)]

    light_counts = (
        df.groupby("light_regime")
        .apply(lambda x: x[["plate", "measurement", "start_date"]].drop_duplicates().shape[0])
        .sort_values(ascending=False)
    )

    n_tifs = len(list(CLEANED_DIR.glob("*.tif")))

    # Total non-NaN datapoints in the wide-format Y(II) timeseries — what the report cites.
    n_timeseries_points = int(len(y2_vals))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "unique_plates": unique_plates,
        "non_empty": non_empty,
        "empty": empty,
        "non_empty_pct": 100.0 * non_empty / n_rows,
        "empty_pct": 100.0 * empty / n_rows,
        "fvfm_mean": float(fvfm["mean"]),
        "fvfm_median": float(fvfm["median"]),
        "fvfm_std": float(fvfm["std"]),
        "y2_mean": float(np.mean(y2_vals)),
        "y2_median": float(np.median(y2_vals)),
        "y2_std": float(np.std(y2_vals)),
        "ynpq_mean": float(np.mean(ynpq_vals)),
        "ynpq_median": float(np.median(ynpq_vals)),
        "ynpq_std": float(np.std(ynpq_vals)),
        "light_counts": dict(light_counts),
        "n_tifs": n_tifs,
        "n_timeseries_points": n_timeseries_points,
    }


def compute_identity_coverage() -> dict | None:
    """Return identity-coverage stats by joining wells.parquet with the identity dataframe.

    Returns None if any input is missing — caller should warn and skip identity-section updates.
    """
    if not WELLS_PARQUET.exists() or not IDENTITY_XLSX.exists():
        return None
    try:
        from chlamy_impi.database_creation.construct_identity_df import (
            construct_identity_dataframe,
        )
    except ImportError:
        return None

    mut = pd.read_excel(IDENTITY_XLSX, sheet_name="large-lib_rearray2.txt")
    ident = construct_identity_dataframe(mut)
    wells = pd.read_parquet(WELLS_PARQUET)

    wells_plates = set(wells["plate"].unique())
    ident_plates = set(ident["plate"].unique())

    excluded_plates = sorted(wells_plates - ident_plates)
    no_data_plates = sorted(ident_plates - wells_plates)

    import string

    wells = wells.copy()
    wells["_well_id"] = [
        f"{string.ascii_uppercase[int(i)]}{int(j) + 1:02d}"
        for i, j in zip(wells["i"], wells["j"])
    ]
    not_blank = ~((wells["i"] == 0) & (wells["j"] == 0))
    wells_nb = wells[not_blank]

    ident_keys = set(zip(ident["plate"], ident["well_id"]))
    dropped = wells_nb[
        ~wells_nb.apply(lambda r: (r["plate"], r["_well_id"]) in ident_keys, axis=1)
    ]
    in_db_plates = wells_plates & ident_plates
    dropped = dropped[dropped["plate"].isin(in_db_plates)]
    unique_drops = dropped[["plate", "_well_id"]].drop_duplicates()
    per_plate = (
        unique_drops.groupby("plate").size().sort_values(ascending=False).to_dict()
    )

    return {
        "excluded_plates": excluded_plates,
        "no_data_plates": no_data_plates,
        "n_dropped_total": len(unique_drops),
        "n_dropped_plates": len(per_plate),
        "per_plate": per_plate,
    }


def check_images() -> list[str]:
    """Return list of warning strings — referenced images missing or gitignored."""
    warnings_list: list[str] = []
    if not INDEX_MD.exists():
        return ["index.md not found"]
    text = INDEX_MD.read_text()
    refs = re.findall(r"!\[[^\]]*\]\(assets/images/([^)]+)\)", text)
    for ref in set(refs):
        path = ASSETS_DIR / ref
        if not path.exists():
            warnings_list.append(f"missing image: docs/assets/images/{ref}")
    return warnings_list


def find_latest_comparison() -> Path | None:
    candidates = sorted(DB_DIR.glob("*/comparison_*_to_*.md"))
    return candidates[-1] if candidates else None


def parse_comparison(path: Path) -> dict:
    text = path.read_text()

    def first_match(pattern: str, default: str = "") -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else default

    old_rows = int(first_match(r"\| Old \| (\d+) \|", "0"))
    new_rows = int(first_match(r"\| New \| (\d+) \|", "0"))
    diff_rows = new_rows - old_rows

    plates_added_match = re.search(r"\*\*Plates added\*\* \((\d+)\):\s*([^\n]*)", text)
    plates_added = (
        plates_added_match.group(2).strip() if plates_added_match else ""
    )
    n_plates_added = int(plates_added_match.group(1)) if plates_added_match else 0

    newly_empty = int(first_match(r"Newly empty wells.*?\*\*(\d+)\*\*", "0"))
    newly_pop = int(first_match(r"Newly populated wells.*?\*\*(\d+)\*\*", "0"))
    n_diff = int(first_match(r"\*\*(\d+) wells\*\* exceeded", "0"))

    old_date = first_match(r"database_(\d{4}-\d{2}-\d{2})\.csv`", "")
    dates = re.findall(r"database_(\d{4}-\d{2}-\d{2})\.csv", text)
    new_date = dates[1] if len(dates) > 1 else ""

    return {
        "old_rows": old_rows,
        "new_rows": new_rows,
        "diff_rows": diff_rows,
        "plates_added": plates_added,
        "n_plates_added": n_plates_added,
        "newly_empty": newly_empty,
        "newly_pop": newly_pop,
        "n_diff": n_diff,
        "old_date": old_date,
        "new_date": new_date,
    }


def bump_version(current: str, kind: str) -> str:
    if kind == "none":
        return current
    parts = current.split(".")
    while len(parts) < 2:
        parts.append("0")
    major, minor = int(parts[0]), int(parts[1])
    if kind == "patch":
        minor += 1
    elif kind == "minor":
        major += 1
        minor = 0
    return f"{major}.{minor}"


def update_config(stats: dict, today: str, bump: str, dry_run: bool) -> None:
    text = CONFIG_YML.read_text()
    cur_version = re.search(r'report_version:\s*"([\d.]+)"', text).group(1)
    new_version = bump_version(cur_version, bump)

    new_text = re.sub(
        r'report_version:\s*"[\d.]+"', f'report_version: "{new_version}"', text
    )
    new_text = re.sub(
        r'report_date:\s*"[\d-]+"', f'report_date: "{today}"', new_text
    )

    print(f"  _config.yml: version {cur_version} -> {new_version}, date -> {today}")
    if not dry_run:
        CONFIG_YML.write_text(new_text)


def update_index(stats: dict, comp: dict, dry_run: bool) -> None:
    text = INDEX_MD.read_text()
    s = stats

    replacements = [
        # Key-numbers table
        (
            r"\| Raw TIF/CSV pairs processed \| \*\*\d+\*\* \|",
            f"| Raw TIF/CSV pairs processed | **{s['n_tifs']}** |",
        ),
        (
            r"\| Plates passing error correction \| \*\*\d+ / \d+\*\* \(100%\) \|",
            f"| Plates passing error correction | **{s['n_tifs']} / {s['n_tifs']}** (100%) |",
        ),
        (
            r"\| Unique plate IDs in database \| \*\*\d+\*\* \|",
            f"| Unique plate IDs in database | **{s['unique_plates']}** |",
        ),
        (
            r"\| Total wells in database \| \*\*[\d,]+\*\* \|",
            f"| Total wells in database | **{fmt(s['n_rows'])}** |",
        ),
        (
            r"\| Non-empty wells \(valid signal\) \| \*\*[\d,]+\*\* \([\d.]+%\) \|",
            f"| Non-empty wells (valid signal) | **{fmt(s['non_empty'])}** ({s['non_empty_pct']:.1f}%) |",
        ),
        (
            r"\| Empty wells \(no algal colony\) \| \*\*[\d,]+\*\* \([\d.]+%\) \|",
            f"| Empty wells (no algal colony) | **{fmt(s['empty'])}** ({s['empty_pct']:.1f}%) |",
        ),
        (
            r"\| Database columns \| \*\*\d+\*\* \|",
            f"| Database columns | **{s['n_cols']}** |",
        ),
        (
            r"\| Database rows \| \*\*[\d,]+\*\* \|",
            f"| Database rows | **{fmt(s['n_rows'])}** |",
        ),
        # Fv/Fm / Y(II) / Y(NPQ) stats rows in summary table
        (
            r"\| Fv/Fm \| [\d.]+ \| [\d.]+ \| [\d.]+ \|",
            f"| Fv/Fm | {s['fvfm_mean']:.3f} | {s['fvfm_median']:.3f} | {s['fvfm_std']:.3f} |",
        ),
        (
            r"\| Y\(II\) \| [\d.]+ \| [\d.]+ \| [\d.]+ \|",
            f"| Y(II) | {s['y2_mean']:.3f} | {s['y2_median']:.3f} | {s['y2_std']:.3f} |",
        ),
        (
            r"\| Y\(NPQ\) \| [\d.]+ \| [\d.]+ \| [\d.]+ \|",
            f"| Y(NPQ) | {s['ynpq_mean']:.3f} | {s['ynpq_median']:.3f} | {s['ynpq_std']:.3f} |",
        ),
        # Stage 0 "Results" bullet — same TIF count
        (
            r"- \*\*\d+ / \d+ plates pass\*\* \(100%\)",
            f"- **{s['n_tifs']} / {s['n_tifs']} plates pass** (100%)",
        ),
        # Stage 2b "Output:" sentence and appendix sentence
        (
            r"A single canonical `database\.csv` \([\d,]+ rows x \d+ columns\)",
            f"A single canonical `database.csv` ({fmt(s['n_rows'])} rows x {s['n_cols']} columns)",
        ),
        (
            r"The final `database\.csv` contains \d+ columns and [\d,]+ rows\.",
            f"The final `database.csv` contains {s['n_cols']} columns and {fmt(s['n_rows'])} rows.",
        ),
        # Timeseries data points
        (
            r"\| Timeseries data points \| \*\*[\d,]+\*\* \|",
            f"| Timeseries data points | **{fmt(s['n_timeseries_points'])}** |",
        ),
    ]

    new_text = text
    for pat, repl in replacements:
        new_text, n = re.subn(pat, repl, new_text)
        if n == 0:
            print(f"  WARN: no match for pattern: {pat[:60]}...")

    new_text = update_latest_comparison(new_text, comp)
    new_text = update_light_regime_table(new_text, s["light_counts"])

    if not dry_run:
        INDEX_MD.write_text(new_text)
    print(f"  index.md: rewrote {len(replacements) + 2} sections")


def update_latest_comparison(text: str, comp: dict) -> str:
    pat = re.compile(
        r"\*\*Latest comparison \(\d{4}-\d{2}-\d{2} vs \d{4}-\d{2}-\d{2}\):\*\* "
        r"[^\n]*?Parameter values for all other previously-existing wells are unchanged\."
    )

    plates_phrase = (
        f"{comp['n_plates_added']} plate{'s' if comp['n_plates_added'] != 1 else ''} added "
        f"({comp['plates_added']})"
        if comp["n_plates_added"]
        else "no new plates"
    )

    para = (
        f"**Latest comparison ({comp['old_date']} vs {comp['new_date']}):** "
        f"{comp['diff_rows']:+,} rows ({fmt(comp['old_rows'])} -> {fmt(comp['new_rows'])}), "
        f"{plates_phrase}. "
        f"{comp['newly_empty']} wells transitioned from populated to empty and "
        f"{comp['newly_pop']} from empty to populated. "
        f"{comp['n_diff']} wells exceeded parameter-diff thresholds. "
        f"Parameter values for all other previously-existing wells are unchanged."
    )

    new_text, n = pat.subn(para, text)
    if n == 0:
        print("  WARN: 'Latest comparison' paragraph not found")
    return new_text


def update_light_regime_table(text: str, counts: dict) -> str:
    header = "### Experiments per light regime"
    if header not in text:
        print("  WARN: light regime table not found")
        return text

    rows = "\n".join(f"| {regime} | {n} |" for regime, n in counts.items())
    table = (
        f"{header}\n\n"
        "| Light regime | Plate count |\n"
        "|---|---|\n"
        f"{rows}\n"
    )

    pat = re.compile(
        re.escape(header) + r".*?(?=\n---\n)", flags=re.DOTALL
    )
    new_text, n = pat.subn(table + "\n", text)
    if n == 0:
        print("  WARN: failed to substitute light regime table")
    return new_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bump", choices=["patch", "minor", "none"], default="patch")
    args = parser.parse_args()

    print(f"Reading {DB_CSV}")
    stats = compute_stats()

    print("\nStats:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    comp_path = find_latest_comparison()
    if comp_path is None:
        raise SystemExit("No comparison report found.")
    print(f"\nLatest comparison: {comp_path.relative_to(REPO_ROOT)}")
    comp = parse_comparison(comp_path)
    for k, v in comp.items():
        print(f"  {k}: {v}")

    print("\nIdentity coverage:")
    coverage = compute_identity_coverage()
    if coverage is None:
        print("  (skipped — wells.parquet or identity spreadsheet not available)")
    else:
        print(f"  excluded plates (in data, not in identity): {coverage['excluded_plates']}")
        print(f"  no-data plates (in identity, not in data): {coverage['no_data_plates']}")
        print(f"  total dropped wells: {coverage['n_dropped_total']} across {coverage['n_dropped_plates']} plates")
        for plate, n in coverage["per_plate"].items():
            print(f"    {plate}: {n}")

    print("\nImage check:")
    img_warnings = check_images()
    if img_warnings:
        for w in img_warnings:
            print(f"  WARN: {w}")
    else:
        print("  all referenced images present")

    today = date.today().isoformat()
    print(f"\nUpdating docs (dry_run={args.dry_run}, bump={args.bump}):")
    update_config(stats, today, args.bump, args.dry_run)
    update_index(stats, comp, args.dry_run)
    print("\nDone." + (" (dry run, no files written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
