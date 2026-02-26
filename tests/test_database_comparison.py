"""Unit tests for database_creation/database_comparison.py

Tests cover:
- compare_databases: identical, added rows, removed rows, schema change,
  empty-well changes, well-level parameter diffs
- find_previous_database / find_previous_database_excluding_today: path discovery
- generate_comparison_report: markdown section headings
"""

import datetime
import unittest
from pathlib import Path

import pandas as pd

from chlamy_impi.database_creation.database_comparison import (
    compare_databases,
    generate_comparison_report,
    write_comparison_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(rows, tmp_path: Path, name: str = "database.csv") -> Path:
    """Write a list of dicts as a CSV and return the path."""
    df = pd.DataFrame(rows)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _base_rows():
    return [
        {"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.8,  "y2_1": 0.5, "ynpq_1": 0.1},
        {"plate": "P1", "well_id": "B02", "mask_area": 20, "fv_fm": 0.75, "y2_1": 0.4, "ynpq_1": 0.2},
        {"plate": "P2", "well_id": "A01", "mask_area": 15, "fv_fm": 0.7,  "y2_1": 0.3, "ynpq_1": 0.05},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIdenticalDatabases(unittest.TestCase):
    def test_no_diffs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old = _make_db(_base_rows(), tmp, "old.csv")
            new = _make_db(_base_rows(), tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertEqual(result["old_rows"], result["new_rows"])
        self.assertEqual(result["added_plates"], set())
        self.assertEqual(result["removed_plates"], set())
        self.assertEqual(result["schema_added"], set())
        self.assertEqual(result["schema_removed"], set())
        self.assertEqual(result["newly_empty"], [])
        self.assertEqual(result["newly_populated"], [])
        self.assertTrue(result["param_diffs"].empty)


class TestIdenticalDatabasesMultipleMeasurements(unittest.TestCase):
    """Regression test: identical databases with >1 measurement per (plate, well_id)
    must produce zero param_diffs (old code produced a Cartesian-product explosion)."""

    def test_no_diffs_with_multiple_measurements(self):
        import tempfile
        rows = [
            {"plate": "P1", "measurement": 1, "well_id": "A01", "mask_area": 10, "fv_fm": 0.8, "y2_1": 0.5, "ynpq_1": 0.1},
            {"plate": "P1", "measurement": 2, "well_id": "A01", "mask_area": 10, "fv_fm": 0.7, "y2_1": 0.4, "ynpq_1": 0.2},
            {"plate": "P1", "measurement": 1, "well_id": "B02", "mask_area": 20, "fv_fm": 0.6, "y2_1": 0.3, "ynpq_1": 0.0},
            {"plate": "P1", "measurement": 2, "well_id": "B02", "mask_area": 20, "fv_fm": 0.5, "y2_1": 0.2, "ynpq_1": 0.05},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old = _make_db(rows, tmp, "old.csv")
            new = _make_db(rows, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertTrue(result["param_diffs"].empty,
                        f"Expected no diffs for identical DBs, got {len(result['param_diffs'])} rows")


class TestAddedRows(unittest.TestCase):
    def test_new_plate_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            extra = _base_rows() + [
                {"plate": "P3", "well_id": "A01", "mask_area": 5, "fv_fm": 0.6, "y2_1": 0.2, "ynpq_1": 0.0},
            ]
            old = _make_db(_base_rows(), tmp, "old.csv")
            new = _make_db(extra, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertEqual(result["new_rows"], 4)
        self.assertIn("P3", result["added_plates"])
        self.assertEqual(result["removed_plates"], set())


class TestRemovedRows(unittest.TestCase):
    def test_removed_plate_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            reduced = [r for r in _base_rows() if r["plate"] != "P2"]
            old = _make_db(_base_rows(), tmp, "old.csv")
            new = _make_db(reduced, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertIn("P2", result["removed_plates"])
        self.assertEqual(result["added_plates"], set())


class TestSchemaChange(unittest.TestCase):
    def test_missing_column_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows_without_col = [
                {k: v for k, v in r.items() if k != "fv_fm"} for r in _base_rows()
            ]
            old = _make_db(_base_rows(), tmp, "old.csv")
            new = _make_db(rows_without_col, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertIn("fv_fm", result["schema_removed"])
        self.assertEqual(result["schema_added"], set())

    def test_added_column_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows_with_extra = [dict(r, extra_col=99) for r in _base_rows()]
            old = _make_db(_base_rows(), tmp, "old.csv")
            new = _make_db(rows_with_extra, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertIn("extra_col", result["schema_added"])


class TestEmptyWellChanges(unittest.TestCase):
    def test_newly_empty_and_populated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old_rows = [
                {"plate": "P1", "well_id": "A01", "mask_area": 0,  "fv_fm": 0.0, "y2_1": 0.0, "ynpq_1": 0.0},
                {"plate": "P1", "well_id": "B02", "mask_area": 10, "fv_fm": 0.7, "y2_1": 0.4, "ynpq_1": 0.1},
            ]
            new_rows = [
                {"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.8, "y2_1": 0.5, "ynpq_1": 0.2},
                {"plate": "P1", "well_id": "B02", "mask_area": 0,  "fv_fm": 0.0, "y2_1": 0.0, "ynpq_1": 0.0},
            ]
            old = _make_db(old_rows, tmp, "old.csv")
            new = _make_db(new_rows, tmp, "new.csv")
            result = compare_databases(old, new)

        self.assertIn(("P1", "B02"), result["newly_empty"])
        self.assertIn(("P1", "A01"), result["newly_populated"])


class TestWellLevelDiffs(unittest.TestCase):
    def test_large_diff_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old_rows = [
                {"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.8,  "y2_1": 0.5, "ynpq_1": 0.1},
                {"plate": "P1", "well_id": "B02", "mask_area": 20, "fv_fm": 0.75, "y2_1": 0.4, "ynpq_1": 0.2},
            ]
            # A01: fv_fm changes by 0.15 (above default threshold 0.05)
            # B02: fv_fm changes by 0.01 (below threshold)
            new_rows = [
                {"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.95, "y2_1": 0.5, "ynpq_1": 0.1},
                {"plate": "P1", "well_id": "B02", "mask_area": 20, "fv_fm": 0.76, "y2_1": 0.4, "ynpq_1": 0.2},
            ]
            old = _make_db(old_rows, tmp, "old.csv")
            new = _make_db(new_rows, tmp, "new.csv")
            result = compare_databases(old, new, fv_fm_threshold=0.05)

        diffs = result["param_diffs"]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs.iloc[0]["well_id"], "A01")

    def test_small_diff_not_reported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old_rows = [{"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.8, "y2_1": 0.5, "ynpq_1": 0.1}]
            new_rows = [{"plate": "P1", "well_id": "A01", "mask_area": 10, "fv_fm": 0.81, "y2_1": 0.5, "ynpq_1": 0.1}]
            old = _make_db(old_rows, tmp, "old.csv")
            new = _make_db(new_rows, tmp, "new.csv")
            result = compare_databases(old, new, fv_fm_threshold=0.05)

        self.assertTrue(result["param_diffs"].empty)


class TestFindPreviousDatabase(unittest.TestCase):
    def _make_dated_csv(self, base: Path, date_str: str) -> None:
        """Create base/YYYY-MM-DD/database_YYYY-MM-DD.csv."""
        subdir = base / date_str
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / f"database_{date_str}.csv").write_text("a")

    def test_returns_most_recent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            import chlamy_impi.paths as paths_module
            original_dir = paths_module.DATABASE_DIR
            paths_module.DATABASE_DIR = tmp

            try:
                for date_str in ["2025-01-01", "2025-06-15", "2026-01-10"]:
                    self._make_dated_csv(tmp, date_str)
                (tmp / "database.csv").write_text("canonical")  # should be excluded

                from chlamy_impi.paths import find_previous_database
                result = find_previous_database()
                self.assertEqual(result.name, "database_2026-01-10.csv")
            finally:
                paths_module.DATABASE_DIR = original_dir

    def test_returns_none_when_no_dated_files(self):
        import tempfile
        import chlamy_impi.paths as paths_module
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            original_dir = paths_module.DATABASE_DIR
            paths_module.DATABASE_DIR = tmp
            try:
                (tmp / "database.csv").write_text("canonical only")
                from chlamy_impi.paths import find_previous_database
                result = find_previous_database()
                self.assertIsNone(result)
            finally:
                paths_module.DATABASE_DIR = original_dir

    def test_excluding_today(self):
        import tempfile
        import chlamy_impi.paths as paths_module
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            original_dir = paths_module.DATABASE_DIR
            paths_module.DATABASE_DIR = tmp
            try:
                today = str(datetime.date.today())
                self._make_dated_csv(tmp, today)
                self._make_dated_csv(tmp, "2025-01-01")

                from chlamy_impi.paths import find_previous_database_excluding_today
                result = find_previous_database_excluding_today()
                self.assertEqual(result.name, "database_2025-01-01.csv")
            finally:
                paths_module.DATABASE_DIR = original_dir


class TestGenerateReportMarkdown(unittest.TestCase):
    def test_section_headings_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old = _make_db(_base_rows(), tmp, "database_2025-01-01.csv")
            new = _make_db(_base_rows(), tmp, "database_2026-02-26.csv")
            result = compare_databases(old, new)
            report = generate_comparison_report(result, old, new)

        for heading in ["# Database Comparison Report", "## Row Counts", "## Schema Changes",
                        "## Plate Changes", "## Empty-Well Changes", "## Well-Level Parameter Diffs"]:
            self.assertIn(heading, report, f"Missing section: {heading}")

    def test_report_written_to_disk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old = _make_db(_base_rows(), tmp, "database_2025-01-01.csv")
            new = _make_db(_base_rows(), tmp, "database_2026-02-26.csv")
            result = compare_databases(old, new)
            out = tmp / "comparison.md"
            write_comparison_report(result, old, new, out)
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("# Database Comparison Report", content)


if __name__ == "__main__":
    unittest.main()
