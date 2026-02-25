"""Load and save raw TIF and CSV files for Stage 0 error correction."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

logger = logging.getLogger(__name__)


def load_tif(path: Path) -> np.ndarray:
    """Return (n_frames, H, W) uint16 array from a TIF file."""
    tif = tifffile.imread(str(path))
    if tif.ndim == 2:
        tif = tif[np.newaxis, ...]
    assert tif.ndim == 3, f"Expected 3D array, got shape {tif.shape} from {path}"
    logger.debug(f"Loaded TIF {path.name}: shape={tif.shape}, dtype={tif.dtype}")
    return tif


def save_tif(tif: np.ndarray, path: Path) -> None:
    """Save (n_frames, H, W) array as a lossless TIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), tif, photometric="minisblack")
    logger.debug(f"Saved TIF {path}: shape={tif.shape}")


def load_csv(path: Path) -> pd.DataFrame:
    """Load semicolon-delimited metadata CSV, stripping the trailing empty column."""
    df = pd.read_csv(path, header=0, delimiter=";")
    # The camera software appends a trailing semicolon producing an unnamed last column
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]
    logger.debug(f"Loaded CSV {path.name}: {len(df)} rows")
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save metadata DataFrame with semicolon delimiter and trailing semicolons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append a trailing semicolon column to match the original camera format
    df_out = df.copy()
    df_out[""] = ""
    df_out.to_csv(path, index=False, sep=";")
    logger.debug(f"Saved CSV {path}: {len(df)} rows")
