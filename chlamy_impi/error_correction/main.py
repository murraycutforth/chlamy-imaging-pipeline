"""Stage 0: Raw TIF + CSV error correction.

Reads raw TIF/CSV pairs from INPUT_DIR, applies all corrections in order,
validates the result, and writes cleaned files to CLEANED_RAW_DATA_DIR.

Correction order
----------------
1. remove_warmup_pair         (TIF only: strips first 2 frames)
2. remove_all_black_frame_pairs
3. remove_duplicate_initial_frame_pair
4. remove_spurious_frames     (automated timestamp-based strategy)
5. validate_tif_csv_pair      (fail-fast assertions)
6. save cleaned TIF + CSV

Run as::

    python -m chlamy_impi.error_correction.main
"""
import logging
from pathlib import Path

from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.error_correction.corrections import (
    remove_all_black_frame_pairs,
    remove_duplicate_initial_frame_pair,
    remove_warmup_pair,
)
from chlamy_impi.error_correction.spurious_frame_strategy import remove_spurious_frames
from chlamy_impi.error_correction.tif_io import load_csv, load_tif, save_csv, save_tif
from chlamy_impi.error_correction.validation import validate_tif_csv_pair
from chlamy_impi.paths import find_all_raw_tif_and_csv, get_cleaned_raw_data_dir

logger = logging.getLogger(__name__)


def correct_plate(tif_path: Path, meta_csv_path: Path, output_dir: Path) -> None:
    """Apply all corrections to one TIF/CSV pair and write cleaned outputs.

    Parameters
    ----------
    tif_path:      Path to the raw ``.tif`` file.
    meta_csv_path: Path to the matching ``.csv`` metadata file.
    output_dir:    Directory where the cleaned ``.tif`` and ``.csv`` are saved.
    """
    basename = tif_path.stem
    _, _, time_regime = parse_name(basename + ".tif")

    logger.info(f"Processing {basename} (time_regime={time_regime})")

    tif = load_tif(tif_path)
    meta_df = load_csv(meta_csv_path)

    logger.debug(f"  raw TIF shape: {tif.shape}, CSV rows: {len(meta_df)}")

    # 1. Remove warmup pair (always present, not in CSV)
    tif = remove_warmup_pair(tif)

    # 2. Remove black frame pairs: full-black (TIF only) then half-black (TIF+CSV)
    tif, meta_df = remove_all_black_frame_pairs(tif, meta_df)

    # 3. Remove duplicate initial frame pair (safety net, rarely fires after warmup removal)
    tif, meta_df = remove_duplicate_initial_frame_pair(tif, meta_df)

    # 4. Automated spurious-frame removal based on timestamp analysis
    tif, meta_df = remove_spurious_frames(tif, meta_df, time_regime)

    # 5. Validate — raises immediately if any invariant is violated
    validate_tif_csv_pair(tif, meta_df, basename, time_regime)

    # 6. Save
    out_tif = output_dir / tif_path.name
    out_csv = output_dir / meta_csv_path.name
    save_tif(tif, out_tif)
    save_csv(meta_df, out_csv)

    logger.info(f"  saved cleaned TIF ({tif.shape[0]} frames) and CSV ({len(meta_df)} rows)")


def _print_summary(total: int, errors: dict[str, str]) -> None:
    n_ok = total - len(errors)
    logger.info("=" * 48)
    logger.info(f"Error correction complete: {n_ok}/{total} plates OK")
    if errors:
        logger.error(f"{len(errors)} plate(s) FAILED:")
        for name, msg in errors.items():
            logger.error(f"  {name}: {msg}")
    logger.info("=" * 48)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    logger.info("=" * 48)
    logger.info("Stage 0: Raw TIF/CSV error correction")
    logger.info("=" * 48)

    tif_csv_pairs = find_all_raw_tif_and_csv()
    if not tif_csv_pairs:
        logger.error("No TIF/CSV pairs found in INPUT_DIR")
        raise SystemExit("ERROR: No input files found")

    logger.info(f"Found {len(tif_csv_pairs)} TIF/CSV pair(s)")

    output_dir = get_cleaned_raw_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}

    for tif_path, csv_path in tif_csv_pairs:
        try:
            correct_plate(tif_path, csv_path, output_dir)
        except Exception as exc:
            logger.error(f"FAILED {tif_path.stem}: {exc}")
            errors[tif_path.stem] = str(exc)

    _print_summary(len(tif_csv_pairs), errors)

    if errors:
        raise SystemExit(f"ERROR: {len(errors)} plate(s) failed correction")


if __name__ == "__main__":
    main()
