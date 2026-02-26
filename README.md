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

The pipeline runs in four sequential stages. Place all raw `.tif` and `.csv` files in `data/` before starting.

```
data/                 (raw TIF + CSV)
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
           output/database_creation/database.csv            (canonical, fixed name)
           output/database_creation/YYYY-MM-DD/             (dated run directory)
               database_YYYY-MM-DD.csv
               comparison_<old>_to_<new>.md   (if a previous database exists)
               timeseries_y2.png + timeseries_ynpq.png
```

All input/output paths are configured in `chlamy_impi/paths.py`.


## Stage 0 — Raw TIF/CSV error correction

Reads every `.tif` / `.csv` pair from `data/`, applies a sequence of automated corrections,
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

Each run creates a dated subdirectory `output/database_creation/YYYY-MM-DD/` containing:

- `database_YYYY-MM-DD.csv` — snapshot for this run
- `comparison_<old>_to_<new>.md` — regression report against the previous run (if one exists)
- `timeseries_y2.png`, `timeseries_ynpq.png` — per-light-regime time series mosaics

A canonical `output/database_creation/database.csv` (fixed name) is also written for backwards compatibility.

Key columns: `plate`, `measurement`, `start_date`, `i`, `j`, `well_id`, `fv_fm`,
`y2_1`…`y2_177`, `ynpq_1`…`ynpq_177`,
`measurement_time_0`…`measurement_time_177`, `mutant_ID`, `gene`, `confidence_level`.


## Pipeline report (GitHub Pages)

A human-readable report of the pipeline results is maintained in `docs/` and published via GitHub Pages. It covers the full pipeline architecture, sample visualisations (well mosaics, mask mosaics, timeseries), Fv/Fm distributions, and data quality notes.

### Viewing the report

The report is published at:
```
https://<org>.github.io/chlamy-imaging-pipeline/
```
(Enable GitHub Pages under repo Settings → Pages → source: `main` branch, `/docs` folder.)

### Generating a new report version

Run the full pipeline first (Stages 0–2b), then regenerate the report assets:

```bash
# 1. Run the full pipeline to update output/ with new data
python -m chlamy_impi.error_correction.main
python -m chlamy_impi.well_segmentation_preprocessing.main
python -m chlamy_impi.image_processing.main
python -m chlamy_impi.database_creation.main_v2

# 2. Regenerate report images from the new pipeline outputs
python scripts/generate_report_assets.py

# 3. Commit the updated docs/ directory
git add docs/
git commit -m "Update pipeline report - YYYY-MM-DD"
git push
```

The key files to update manually when significant pipeline changes are made:

| File | What to update |
|---|---|
| `docs/_config.yml` | `report_version`, `report_date`, `pipeline_version` |
| `docs/index.md` | Summary statistics tables, known issues, schema appendix |

The timeseries PNGs (`timeseries_y2.png`, `timeseries_ynpq.png`) and the Fv/Fm plots are regenerated automatically by `scripts/generate_report_assets.py`. Sample well mosaics and mask images are copied from the latest pipeline output.

### Report versioning convention

- Increment `report_version` in `docs/_config.yml` when the report content changes significantly (new sections, new analysis, schema changes).
- The `report_date` should always reflect the date of the most recent pipeline run used to generate the report assets.


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
