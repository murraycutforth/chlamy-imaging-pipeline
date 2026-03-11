Audit the GitHub Pages report at `docs/index.md` against the actual pipeline outputs. For every statistic, figure reference, and plate count mentioned in the report:

1. Trace it back to the source data or pipeline output file (`output/image_processing/plates.parquet`, `output/image_processing/wells.parquet`, `output/image_processing/timeseries.parquet`, `output/database_creation/database.csv`, `reports/mask_comparison/global_stats.csv`, `reports/mask_comparison/per_plate.csv`)
2. Recompute the correct value using `/Users/murraycutforth/miniconda3/envs/chlamy/bin/python`
3. If it differs from what's in the report, fix it in-place

Also check that all images referenced in the report exist in `docs/assets/images/` and are not blocked by `.gitignore`.

Generate a changelog of every correction made (as a table: location, old value, correct value, root cause).

Finally, commit all changes with the message `docs: sync report statistics with latest pipeline outputs` and push.
