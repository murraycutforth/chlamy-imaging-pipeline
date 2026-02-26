"""Integration tests for the content of the final database.csv.

Tests are skipped if the real database file is absent (i.e. the pipeline has
not been run in this checkout).
"""

import unittest
from pathlib import Path

import pandas as pd

from chlamy_impi.paths import get_csv_filename

_DB_PATH = get_csv_filename()

# Plates documented in the Stage 2b report section as present in the
# experimental data but absent from the identity spreadsheet, and therefore
# excluded from the final database.
_PLATES_EXCLUDED_NO_IDENTITY = {"34v1", "34v2", "34v3", "35", "35v1", "35v2", "35v3"}


@unittest.skipUnless(_DB_PATH.exists(), "Real database file not found — run the pipeline first")
class TestExcludedPlatesAbsentFromDatabase(unittest.TestCase):
    """Verify that plates without an identity spreadsheet entry are absent from the database."""

    @classmethod
    def setUpClass(cls):
        cls.plates_in_db = set(pd.read_csv(_DB_PATH, usecols=["plate"], dtype={"plate": str})["plate"].unique())

    def test_excluded_plates_not_in_database(self):
        present = _PLATES_EXCLUDED_NO_IDENTITY & self.plates_in_db
        self.assertEqual(
            present,
            set(),
            f"Plates that should be excluded are present in the database: {present}",
        )

    def test_each_excluded_plate_individually(self):
        for plate in sorted(_PLATES_EXCLUDED_NO_IDENTITY):
            with self.subTest(plate=plate):
                self.assertNotIn(
                    plate,
                    self.plates_in_db,
                    f"Plate '{plate}' should be excluded (no identity entry) but is in the database",
                )
