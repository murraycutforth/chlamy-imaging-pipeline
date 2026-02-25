# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chlamy-IMPI is an image processing and data analysis pipeline for studying photosynthesis in *Chlamydomonas reinhardtii* mutants. It processes fluorescence imaging data from 384-well plates (16×24) to extract photosynthetic parameters (Fv/Fm, Y2, NPQ).

## Environment Setup

```bash
pip install -r requirements.txt
# or with Poetry:
poetry install && poetry shell
```

[machine dependent] environment may already exist at: /Users/murraycutforth/miniconda3/envs/chlamy/bin/python

## Commands

```bash
# Run tests
python -m unittest discover tests

# Run a single test file
python -m unittest tests/test_fv_fm.py

# Run the full pipeline (3 stages in order)
python -m chlamy_impi.well_segmentation_preprocessing.main
python -m chlamy_impi.error_correction.main
python -m chlamy_impi.database_creation.main

# Interactive Streamlit demo
streamlit run chlamy_impi/interactive.demo.py

# Investigate metadata for manual error correction
python -m chlamy_impi.database_creation.investigate_meta_df
```

## Pipeline Architecture

The pipeline has 3 sequential stages. Each stage reads from a specific input directory and writes to an output directory defined in `chlamy_impi/paths.py`.

**Stage 1 – Well Segmentation** (`well_segmentation_preprocessing/main.py`)
- Input: `.tif` fluorescence images + `.csv` metadata in `data/chlamy/`
- Uses the `segment-multiwell-plate` library to extract individual wells from 384-well plates
- Output: `.npy` arrays of shape `(16, 24, num_frames, well_height, well_width)` in `output/well_segmentation_cache/`

**Stage 1.5 – Error Correction** (`error_correction/main.py`)
- Input: `.npy` arrays from Stage 1
- Detects and removes spurious/duplicate frames using timestamp metadata
- Output: corrected `.npy` arrays in `output/corrected_well_segmentation_cache/`
- Note: Some plates require manual correction via `database_creation/manual_error_correction.py`

**Stage 2 – Database Creation** (`database_creation/main.py`)
- Input: corrected `.npy` arrays + identity spreadsheet `.xlsx` mapping wells to mutant strains
- Computes per-well photosynthetic parameters using `lib/` modules
- Output: `database.parquet` + `database.csv` in `output/database_creation/`

## Key Files

- **`chlamy_impi/paths.py`**: Central path configuration — all input/output directories. Edit here when changing data locations.
- **`chlamy_impi/database_creation/constants.py`**: Hardcoded experimental parameters: valid frame counts `{84, 92, 100, 164, 172, 180}`, time regime intervals.
- **`chlamy_impi/lib/`**: Core computation modules — `fv_fm_functions.py`, `y2_functions.py`, `npq_functions.py`, `mask_functions.py`.

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

- All stages write failed-plate info to `failed_files.csv` and per-plate error `.txt` files
- `IGNORE_ERRORS` flag in pipeline scripts allows skipping problematic plates
- Sanity checks are in `database_creation/database_sanity_checks.py`
- Logging uses Python's `logging` module at DEBUG/INFO levels
