"""Build a per-well contamination dataframe from the Daily Checklist Log File.

The checklist xlsx has two sheets ("Phase I", "Phase II"). Each row of the camera
section records a plate-measurement that went under the camera, with a free-text
"Contaminated colonies" cell listing well IDs (e.g. ``L1``, ``M8,M9,L9(almost)``,
``N12-N16, O12-O16``).

This module parses those cells into a long-form ``(plate, measurement, well_id)``
DataFrame. Wells that appear in this DataFrame are flagged as contaminated when
merged into the final database in Stage 2b.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from chlamy_impi.paths import get_daily_checklist_path

logger = logging.getLogger(__name__)


_PLATE_NAME_RE = re.compile(r"^(.+)-([Mm]\d+)$")

# Single well token: row letter A-P, then 1-2 digit column 1-24.
# Optionally followed by a hyphen and a second column for a same-row range.
# Trailing chars (parentheses, punctuation) are tolerated by the surrounding tokenizer.
_WELL_TOKEN_RE = re.compile(r"\b([A-Pa-p])\s*(\d{1,2})(?:\s*-\s*([A-Pa-p]?)\s*(\d{1,2}))?")


def _normalize_well_id(row_letter: str, col: int) -> str | None:
    """Return ``A01``..``P24`` if valid, else None."""
    if not (1 <= col <= 24):
        return None
    return f"{row_letter.upper()}{col:02d}"


def parse_colony_codes(text: str) -> list[str]:
    """Parse a free-text contamination cell into a sorted list of well_ids.

    Supports single wells (``L1``), comma-separated lists (``M8,M9,L9``), ranges
    along a row (``N12-N16``), and tolerates trailing notes (``L9(almost)``,
    ``"and potentilly N3, N2"``). Tokens that don't match a valid well (e.g.
    ``"potentilly"``, out-of-range columns) are silently skipped.

    Returns an empty list for NaN / empty inputs.
    """
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    found: set[str] = set()
    for m in _WELL_TOKEN_RE.finditer(text):
        row_letter, col_str, end_row, end_col_str = m.groups()
        col = int(col_str)
        well = _normalize_well_id(row_letter, col)
        if well is None:
            continue

        if end_col_str:
            # Range: same row required (end_row may be omitted "N12-16" or specified "N12-N16").
            end_row_letter = end_row if end_row else row_letter
            if end_row_letter.upper() != row_letter.upper():
                # Cross-row ranges are ambiguous; only emit the start well.
                found.add(well)
                continue
            end_col = int(end_col_str)
            if not (1 <= end_col <= 24) or end_col < col:
                found.add(well)
                continue
            for c in range(col, end_col + 1):
                w = _normalize_well_id(row_letter, c)
                if w is not None:
                    found.add(w)
        else:
            found.add(well)

    return sorted(found)


def _split_plate_name(plate_name: str) -> tuple[str, str] | None:
    """Split ``"32v1-M3"`` into ``("32v1", "M3")``. Returns None if unparseable."""
    if not isinstance(plate_name, str):
        return None
    m = _PLATE_NAME_RE.match(plate_name.strip())
    if m is None:
        return None
    plate = m.group(1).strip().lower()
    measurement = m.group(2).strip().upper()
    return plate, measurement


# Per-sheet column names for the camera section. Whitespace/newlines as
# they appear in the source file.
_SHEET_COLUMNS = {
    "Phase II": {
        "date": "Date",
        "plate_name": "Plate Name",
        "colonies": "Contaminated colonies",
    },
    "Phase I": {
        "date": "Date",
        "plate_name": "Plate Name",
        "colonies": "Contaminated colonies",
    },
}


def _read_sheet(sheet_name: str, cols: dict, path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=1, engine="openpyxl")
    sub = df[[cols["date"], cols["plate_name"], cols["colonies"]]].copy()
    sub.columns = ["date", "plate_name", "colonies"]
    sub = sub.dropna(subset=["plate_name"])
    sub["sheet"] = sheet_name
    return sub


def _normalize_date(value) -> str | None:
    """Convert a checklist Date cell to ``"YYYY-MM-DD"`` to match DB ``start_date``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def construct_contamination_dataframe() -> pd.DataFrame:
    """Read the daily checklist xlsx and return a (plate, measurement, start_date, well_id) DataFrame.

    One row per (plate, measurement, start_date, well_id) flagged as contaminated.
    The ``start_date`` column is the checklist row's ``Date`` formatted as
    ``"YYYY-MM-DD"`` to match the DB schema. Rows missing a parseable plate name,
    date, or any recognisable well IDs are dropped.
    """
    path = get_daily_checklist_path()
    cols = ["plate", "measurement", "start_date", "well_id"]
    if not path.exists():
        logger.warning(f"Daily checklist file not found at {path}; contamination column will be all 0")
        return pd.DataFrame(columns=cols)

    frames = []
    for sheet, sheet_cols in _SHEET_COLUMNS.items():
        try:
            frames.append(_read_sheet(sheet, sheet_cols, path))
        except Exception as exc:
            logger.warning(f"Failed to read sheet {sheet!r}: {exc}")

    if not frames:
        return pd.DataFrame(columns=cols)

    raw = pd.concat(frames, ignore_index=True)

    rows = []
    skipped_plates = 0
    skipped_dates = 0
    for _, r in raw.iterrows():
        parsed = _split_plate_name(r["plate_name"])
        if parsed is None:
            skipped_plates += 1
            continue
        plate, measurement = parsed
        start_date = _normalize_date(r["date"])
        if start_date is None:
            skipped_dates += 1
            continue
        wells = parse_colony_codes(r["colonies"])
        for w in wells:
            rows.append(
                {"plate": plate, "measurement": measurement, "start_date": start_date, "well_id": w}
            )

    if skipped_plates:
        logger.info(f"Skipped {skipped_plates} checklist rows with unparseable plate names")
    if skipped_dates:
        logger.info(f"Skipped {skipped_dates} checklist rows with unparseable dates")

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    logger.info(
        f"Constructed contamination dataframe: {len(out)} (plate,measurement,start_date,well) entries "
        f"covering {out[['plate','measurement','start_date']].drop_duplicates().shape[0]} plate-measurement-dates"
    )
    return out
