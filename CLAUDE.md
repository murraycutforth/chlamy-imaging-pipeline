# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chlamy-IMPI is an image processing and data analysis pipeline for studying photosynthesis in *Chlamydomonas reinhardtii* mutants. It processes fluorescence imaging data from 384-well plates (16×24) to extract photosynthetic parameters (Fv/Fm, Y2, NPQ).

## Environment Setup

```bash
pip install -r requirements.txt
```

[machine dependent] environment may already exist at: /Users/murraycutforth/miniconda3/envs/chlamy/bin/python

## Commands

```bash
# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_fv_fm.py

# Run the full pipeline (4 stages in order)
python -m chlamy_impi.error_correction.main
python -m chlamy_impi.well_segmentation_preprocessing.main
python -m chlamy_impi.image_processing.main
python -m chlamy_impi.database_creation.main_v2

# Interactive Streamlit demo
streamlit run chlamy_impi/interactive.demo.py
```

## Pipeline Architecture

The pipeline has 4 sequential stages. Each stage reads from a specific input directory and writes to an output directory defined in `chlamy_impi/paths.py`.

**Stage 0 – Error Correction** (`error_correction/main.py`)
- Input: raw `.tif` + `.csv` pairs in `data/chlamy/`
- Strips warmup frames, removes black frame pairs, detects spurious frames via timestamps
- Validates frame count, TIF/CSV alignment, monotone timestamps, and interval consistency
- Output: cleaned `.tif` + `.csv` in `output/cleaned_raw_data/`

**Stage 1 – Well Segmentation** (`well_segmentation_preprocessing/main.py`)
- Input: cleaned `.tif` + `.csv` from `output/cleaned_raw_data/`
- Uses the `segment-multiwell-plate` library to extract individual wells from 384-well plates
- Output: `.npy` arrays of shape `(16, 24, num_frames, well_height, well_width)` in `output/well_segmentation_cache/`

**Stage 2a – Image Processing** (`image_processing/main.py`)
- Input: `.npy` arrays from Stage 1 + cleaned `.csv` from Stage 0
- Computes per-well photosynthetic parameters (Fv/Fm, Y(II), Y(NPQ)) using `lib/` modules
- Output: `plates.parquet`, `wells.parquet`, `timeseries.parquet` in `output/image_processing/`

**Stage 2b – Database Creation** (`database_creation/main_v2.py`)
- Input: parquets from Stage 2a + identity spreadsheet `.xlsx` mapping wells to mutant strains
- Pivots timeseries long→wide, merges with identity, runs sanity checks
- Output: `database.parquet` + `database.csv` in `output/database_creation/`

## Key Files

- **`chlamy_impi/paths.py`**: Central path configuration — all input/output directories. Edit here when changing data locations.
- **`chlamy_impi/database_creation/constants.py`**: Hardcoded experimental parameters: valid frame counts per time regime, expected time intervals.
- **`chlamy_impi/lib/`**: Core computation modules — `fv_fm_functions.py`, `y2_functions.py`, `npq_functions.py`, `mask_functions.py`.
- **`chlamy_impi/image_processing/main.py`**: Stage 2a entry point — vectorised per-plate processing, writes parquets.
- **`chlamy_impi/database_creation/main_v2.py`**: Stage 2b entry point — reads parquets, merges identity, writes database.csv.

## Photosynthetic Parameters

- **Fv/Fm** = (Fm − F0) / Fm — maximum photosynthetic yield (single measurement per plate)
- **Y2** — effective quantum yield of PSII at each timepoint (up to 81 timepoints, stored as `y2_1`…`y2_81`)
- **NPQ** — non-photochemical quenching at each timepoint (stored as `ynpq_1`…`ynpq_81`)

Each TIF image has 2 channels (F0 and Fm) per timepoint. Arrays use pixel-level values; masks (`mask_functions.py`) filter valid pixels per well before averaging.

## Database Schema

One row per plate × well × measurement set. Key columns:
- `plate`, `i`, `j`, `well_id` — plate/well location
- `fv_fm`, `fv_fm_std` — max yield
- `y2_1`…`y2_81`, `y2_std_1`…`y2_std_81` — time-series quantum yield
- `ynpq_1`…`ynpq_81` — time-series NPQ
- `measurement_time_0`…`measurement_time_81` — Unix timestamps
- `mutant_ID`, `gene`, `confidence_level`, `description` — genetic identity from identity spreadsheet

## Error Handling Patterns

- Stage 2a collects per-plate failures in a list; continues on error when `IGNORE_ERRORS=True`
- Final sanity checks are in `database_creation/database_sanity_checks.py`
- Logging uses Python's `logging` module at DEBUG/INFO levels
