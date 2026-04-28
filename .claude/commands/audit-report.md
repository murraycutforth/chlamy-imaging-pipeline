Audit and refresh the GitHub Pages report at `docs/index.md` so its statistics match the latest pipeline outputs.

The audit workflow is encapsulated in `scripts/update_report_docs.py`, which:
- Recomputes key numbers (TIF count, plate count, wells, empty/non-empty, columns, rows, timeseries points) from `output/cleaned_raw_data/`, `output/image_processing/*.parquet`, and `output/database_creation/database.csv`.
- Recomputes Fv/Fm, Y(II), Y(NPQ) summary stats over non-empty wells.
- Recomputes the experiments-per-light-regime table.
- Parses the latest dated `comparison_*_to_*.md` and rewrites the "Latest comparison" paragraph.
- Computes identity coverage by joining `wells.parquet` with the constructed identity dataframe (excluded plates, no-data plates, dropped wells per plate).
- Checks every `assets/images/<file>` reference in `docs/index.md` exists on disk (`docs/assets/images/*.png` is whitelisted in `.gitignore`).
- Updates `docs/_config.yml` (version + date) and `docs/index.md` in-place.

Steps:

1. **Dry run** to inspect computed values and any warnings:
   ```
   /Users/murraycutforth/miniconda3/envs/chlamy/bin/python scripts/update_report_docs.py --dry-run --bump none
   ```
   Address any `WARN:` lines (missing image, missing pattern, missing comparison report) before continuing.

2. **Apply the update** with an appropriate version bump (`patch` for new data only, `minor` for algorithm/structure changes, `none` to leave the version unchanged):
   ```
   /Users/murraycutforth/miniconda3/envs/chlamy/bin/python scripts/update_report_docs.py --bump patch
   ```

3. **Review the diff** with `git diff docs/_config.yml docs/index.md`. The script only updates regex-matched sections; anything that lives in narrative prose (e.g. the identity-coverage section listing dropped-wells per plate, the "Recommendations" list, the executive-summary "stable and reproducible" paragraph) must be edited by hand to reflect the printed identity-coverage stats.

4. **Generate a changelog** of every correction made (location, old value, correct value, root cause).

5. **Commit and push** with the message `docs: sync report statistics with latest pipeline outputs`.
