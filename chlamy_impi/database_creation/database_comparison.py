"""Regression comparison between two dated database CSV files.

Public API:
    compare_databases(old_path, new_path, ...) -> dict
    generate_comparison_report(result, old_path, new_path) -> str
    write_comparison_report(result, old_path, new_path, output_path)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_WELL_KEY = ["plate", "measurement", "well_id"]
_PARAM_COLS = ["fv_fm", "y2_1", "ynpq_1"]


def compare_databases(
    old_path: Path,
    new_path: Path,
    fv_fm_threshold: float = 0.05,
    y2_threshold: float = 0.10,
) -> dict:
    """Load two database CSVs and produce a structured comparison dict.

    Keys in the returned dict:
        old_rows, new_rows          — row counts
        old_plates, new_plates      — sets of plate names
        added_plates, removed_plates
        schema_added, schema_removed — column name sets
        newly_empty                 — list of (plate, well_id) tuples that became empty
        newly_populated             — list of (plate, well_id) tuples that became populated
        param_diffs                 — DataFrame of wells exceeding diff thresholds
    """
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    result = {}

    # --- Row / plate counts ---
    result["old_rows"] = len(old_df)
    result["new_rows"] = len(new_df)
    result["old_plates"] = set(old_df["plate"].unique()) if "plate" in old_df.columns else set()
    result["new_plates"] = set(new_df["plate"].unique()) if "plate" in new_df.columns else set()
    result["added_plates"] = result["new_plates"] - result["old_plates"]
    result["removed_plates"] = result["old_plates"] - result["new_plates"]

    # --- Schema diff ---
    result["schema_added"] = set(new_df.columns) - set(old_df.columns)
    result["schema_removed"] = set(old_df.columns) - set(new_df.columns)

    # --- Empty-well changes ---
    result["newly_empty"] = []
    result["newly_populated"] = []

    if "mask_area" in old_df.columns and "mask_area" in new_df.columns:
        key_cols = [c for c in _WELL_KEY if c in old_df.columns and c in new_df.columns]
        old_empty = set(map(tuple, old_df.loc[old_df["mask_area"] == 0, key_cols].values))
        new_empty = set(map(tuple, new_df.loc[new_df["mask_area"] == 0, key_cols].values))
        result["newly_empty"] = sorted(new_empty - old_empty)
        result["newly_populated"] = sorted(old_empty - new_empty)

    # --- Well-level parameter diffs ---
    common_cols = [c for c in _PARAM_COLS if c in old_df.columns and c in new_df.columns]
    join_cols = [c for c in _WELL_KEY if c in old_df.columns and c in new_df.columns]

    if join_cols and common_cols:
        merged = pd.merge(
            old_df[join_cols + common_cols].rename(columns={c: f"{c}_old" for c in common_cols}),
            new_df[join_cols + common_cols].rename(columns={c: f"{c}_new" for c in common_cols}),
            on=join_cols,
            how="inner",
        )

        diff_rows_mask = pd.Series(False, index=merged.index)
        for col in common_cols:
            old_col = merged[f"{col}_old"]
            new_col = merged[f"{col}_new"]
            abs_diff = (new_col - old_col).abs()
            threshold = fv_fm_threshold if col == "fv_fm" else y2_threshold
            merged[f"{col}_diff"] = abs_diff
            diff_rows_mask |= abs_diff > threshold

        result["param_diffs"] = merged[diff_rows_mask].reset_index(drop=True)
    else:
        result["param_diffs"] = pd.DataFrame()

    return result


def generate_comparison_report(result: dict, old_path: Path, new_path: Path) -> str:
    """Return a markdown string summarising the comparison result."""
    lines = []
    lines.append(f"# Database Comparison Report")
    lines.append(f"")
    lines.append(f"- **Old**: `{old_path.name}`")
    lines.append(f"- **New**: `{new_path.name}`")
    lines.append(f"")

    # Row counts
    lines.append(f"## Row Counts")
    lines.append(f"")
    lines.append(f"| | Rows |")
    lines.append(f"|---|---|")
    lines.append(f"| Old | {result['old_rows']} |")
    lines.append(f"| New | {result['new_rows']} |")
    lines.append(f"| Diff | {result['new_rows'] - result['old_rows']:+d} |")
    lines.append(f"")

    # Schema
    lines.append(f"## Schema Changes")
    lines.append(f"")
    if result["schema_added"]:
        lines.append(f"**Columns added**: {', '.join(sorted(result['schema_added']))}")
    else:
        lines.append(f"**Columns added**: *(none)*")
    if result["schema_removed"]:
        lines.append(f"**Columns removed**: {', '.join(sorted(result['schema_removed']))}")
    else:
        lines.append(f"**Columns removed**: *(none)*")
    lines.append(f"")

    # Plate changes
    lines.append(f"## Plate Changes")
    lines.append(f"")
    if result["added_plates"]:
        lines.append(f"**Plates added** ({len(result['added_plates'])}): {', '.join(sorted(str(p) for p in result['added_plates']))}")
    else:
        lines.append(f"**Plates added**: *(none)*")
    if result["removed_plates"]:
        lines.append(f"**Plates removed** ({len(result['removed_plates'])}): {', '.join(sorted(str(p) for p in result['removed_plates']))}")
    else:
        lines.append(f"**Plates removed**: *(none)*")
    lines.append(f"")

    # Empty well changes
    lines.append(f"## Empty-Well Changes")
    lines.append(f"")
    lines.append(f"Newly empty wells (mask_area 0→0): **{len(result['newly_empty'])}**")
    if result["newly_empty"]:
        for item in result["newly_empty"][:20]:
            lines.append(f"  - {' / '.join(str(x) for x in item)}")
        if len(result["newly_empty"]) > 20:
            lines.append(f"  - *(and {len(result['newly_empty']) - 20} more)*")
    lines.append(f"")
    lines.append(f"Newly populated wells (mask_area 0→>0): **{len(result['newly_populated'])}**")
    if result["newly_populated"]:
        for item in result["newly_populated"][:20]:
            lines.append(f"  - {' / '.join(str(x) for x in item)}")
        if len(result["newly_populated"]) > 20:
            lines.append(f"  - *(and {len(result['newly_populated']) - 20} more)*")
    lines.append(f"")

    # Parameter diffs
    lines.append(f"## Well-Level Parameter Diffs")
    lines.append(f"")
    diffs = result["param_diffs"]
    if diffs.empty:
        lines.append(f"No wells exceeded the diff thresholds.")
    else:
        lines.append(f"**{len(diffs)} wells** exceeded at least one threshold:")
        lines.append(f"")
        # Build a compact table
        diff_cols = [c for c in diffs.columns if c.endswith("_diff")]
        header_cols = ["plate", "well_id"] + diff_cols
        header_cols = [c for c in header_cols if c in diffs.columns]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join("---" for _ in header_cols) + " |")
        for _, row in diffs[header_cols].head(50).iterrows():
            lines.append("| " + " | ".join(str(round(row[c], 4)) if isinstance(row[c], float) else str(row[c]) for c in header_cols) + " |")
        if len(diffs) > 50:
            lines.append(f"")
            lines.append(f"*(showing 50 of {len(diffs)} rows)*")
    lines.append(f"")

    return "\n".join(lines)


def write_comparison_report(result: dict, old_path: Path, new_path: Path, output_path: Path):
    """Write the markdown report to output_path and print a short console summary."""
    report = generate_comparison_report(result, old_path, new_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Comparison report written to: {output_path}")

    # Console summary
    diffs = result["param_diffs"]
    print(
        f"\n=== Database Comparison Summary ===\n"
        f"  Old: {old_path.name} ({result['old_rows']} rows)\n"
        f"  New: {new_path.name} ({result['new_rows']} rows)\n"
        f"  Plates added/removed: {len(result['added_plates'])}/{len(result['removed_plates'])}\n"
        f"  Schema cols added/removed: {len(result['schema_added'])}/{len(result['schema_removed'])}\n"
        f"  Newly empty/populated wells: {len(result['newly_empty'])}/{len(result['newly_populated'])}\n"
        f"  Wells exceeding param thresholds: {len(diffs)}\n"
        f"  Full report: {output_path}\n"
        f"==================================="
    )
