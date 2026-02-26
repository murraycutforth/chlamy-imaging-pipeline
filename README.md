# Chlamy-IMPI

This repository contains the image processing and data analysis pipeline used to study photosynthesis in a large
number of *Chlamydomonas reinhardtii* mutants under various growth conditions.

Note: this repository is still under development, so things might break and these instructions might be incomplete.


## Environment setup

```bash
pip install -r requirements.txt
```

The conda environment on the lab Mac is at `/Users/murraycutforth/miniconda3/envs/chlamy/bin/python`.


## Pipeline overview

The pipeline runs in four sequential stages. Place all raw `.tif` and `.csv` files in `data/chlamy/` before starting.

```
data/chlamy/          (raw TIF + CSV)
      │
      ▼
[Stage 0]  python -m chlamy_impi.error_correction.main
      │    output/cleaned_raw_data/   (cleaned TIF + CSV)
      │
      ▼
[Stage 1]  python -m chlamy_impi.well_segmentation_preprocessing.main
      │    output/well_segmentation_cache/   (.npy arrays, shape 16×24×frames×H×W)
      │
      ▼
[Stage 2a] python -m chlamy_impi.image_processing.main
      │    output/image_processing/   (plates.parquet, wells.parquet, timeseries.parquet)
      │
      ▼
[Stage 2b] python -m chlamy_impi.database_creation.main_v2
           output/database_creation/database.parquet + database.csv
```

All input/output paths are configured in `chlamy_impi/paths.py`.


## Stage 0 — Raw TIF/CSV error correction

Reads every `.tif` / `.csv` pair from `data/chlamy/`, applies a sequence of automated corrections,
validates the result, and writes cleaned copies to `output/cleaned_raw_data/`.

```bash
python -m chlamy_impi.error_correction.main
```

### What it corrects

Every raw TIF has a fixed header structure that must be stripped before the frames align with the CSV:

```
frames [0, 1]   warmup duplicate        — no CSV row  (always present)
frames [2, 3]   pre-measurement trigger — no CSV row  (full-black or half-black)
frames [4, 5]   measurement 0           — CSV row 0
frames [6, 7]   measurement 1           — CSV row 1
...
```

Corrections are applied in this order:

1. **Remove warmup pair** — strips frames 0–1 (TIF only).
2. **Remove black frame pairs** — removes pre-measurement trigger pairs from TIF only; removes mid-experiment half-black pairs from both TIF and CSV.
3. **Remove duplicate initial frame pair** — safety net for a rare double-warmup edge case.
4. **Remove spurious frames** — automated timestamp-based detection removes any remaining extra frame pairs.
5. **Validate** — asserts frame count, TIF/CSV alignment, no black frames, monotone timestamps, and interval consistency. Raises immediately on failure.
6. **Save** cleaned TIF + CSV to `output/cleaned_raw_data/`.

### Valid post-correction frame counts

| Time regime | Valid frame counts |
|---|---|
| `30s-30s` | 160, 162, 164, 172, 178, 180 |
| `1min-1min` | 160, 162, 164, 172, 180 |
| `10min-10min` | 160, 162, 164, 172, 180 |
| `1min-5min` | 180 |
| `5min-5min` | 180 |
| `2h-2h` | 82, 84, 98, 100 |
| `20h_ML` | 82, 84, 92 |
| `20h_HL` | 82, 84, 90, 92 |

Lower counts (82, 160, 162) are truncated experiments. Higher counts (90, 98, 178) are Phase II variants with additional measurements.

### Known unfixable plates

Three plates are exempt from timestamp validation (frame count is still checked) due to confirmed data-collection faults. They are listed in `get_timestamp_check_exempt_plates()` in `error_correction/validation.py`:

| Plate | Reason |
|---|---|
| `20231102_4-M4_20h_ML` | 24-hour gap — experiment interrupted overnight |
| `20231104_4-M6_10min-10min` | DST clock rollback (US, Nov 2023) |
| `20241102_33v3-M6_10min-10min` | DST clock rollback (US, Nov 2024) |

Any future plate with similar issues must be added to this list explicitly; it will otherwise fail validation.

## Stage 1 — Well segmentation

Reads cleaned TIFs from `output/cleaned_raw_data/` and segments each frame into individual wells using the `segment-multiwell-plate` library.

```bash
python -m chlamy_impi.well_segmentation_preprocessing.main
```

Output: `.npy` arrays of shape `(16, 24, num_frames, H, W)` in `output/well_segmentation_cache/`.


## Stage 2a — Image processing

Computes per-well photosynthetic parameters from the segmented `.npy` arrays and writes three normalised parquet files.

```bash
python -m chlamy_impi.image_processing.main
```

Output in `output/image_processing/`:
- `plates.parquet` — one row per experiment (plate × measurement)
- `wells.parquet` — one row per (plate × well), including Fv/Fm and mask area
- `timeseries.parquet` — one row per (plate × well × time step), long format, containing Y(II) and Y(NPQ)


## Stage 2b — Database creation

Joins the Stage 2a parquets with the identity spreadsheet and writes the final database.

```bash
python -m chlamy_impi.database_creation.main_v2
```

Output: `output/database_creation/database.parquet` and `database.csv`.

Key columns: `plate`, `i`, `j`, `well_id`, `fv_fm`, `y2_1`…`y2_81`, `ynpq_1`…`ynpq_81`,
`measurement_time_0`…`measurement_time_81`, `mutant_ID`, `gene`, `confidence_level`.


## Data analysis

Coming soon.


## Streamlit interactive demo

```bash
streamlit run chlamy_impi/interactive.demo.py
```


## Running tests

Unit tests (fast, no real data required):

```bash
python -m pytest tests/ --ignore=tests/test_well_segmentation_integration.py --ignore=tests/test_rotation_correction.py
```

Integration tests (require Stage 0 output in `output/cleaned_raw_data/`):

```bash
python -m pytest tests/test_well_segmentation_integration.py tests/test_rotation_correction.py
```
