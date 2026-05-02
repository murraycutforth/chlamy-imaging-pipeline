import pandas as pd
import pytest

from chlamy_impi.database_creation.construct_contamination_df import (
    parse_colony_codes,
)
from chlamy_impi.database_creation.main_v2 import add_contamination_column


class TestParseColonyCodes:
    def test_empty(self):
        assert parse_colony_codes("") == []
        assert parse_colony_codes("   ") == []
        assert parse_colony_codes(None) == []
        assert parse_colony_codes(float("nan")) == []

    def test_single_well(self):
        assert parse_colony_codes("L1") == ["L01"]
        assert parse_colony_codes("A01") == ["A01"]
        assert parse_colony_codes("p24") == ["P24"]

    def test_comma_list(self):
        assert parse_colony_codes("M8,M9,L9") == ["L09", "M08", "M09"]

    def test_with_hedge_in_parens(self):
        # "(almost)" suffix and "potentilly" prose are tolerated
        assert parse_colony_codes("L1, and potentilly M3, N3, N2") == ["L01", "M03", "N02", "N03"]
        assert parse_colony_codes("M8,M9,L9(almost),O9,P9(almost)") == [
            "L09", "M08", "M09", "O09", "P09",
        ]

    def test_range_same_row(self):
        assert parse_colony_codes("N12-N16") == ["N12", "N13", "N14", "N15", "N16"]

    def test_multiple_ranges(self):
        result = parse_colony_codes("N12-N16, O12-O16, P11-P17")
        expected = (
            [f"N{c:02d}" for c in range(12, 17)]
            + [f"O{c:02d}" for c in range(12, 17)]
            + [f"P{c:02d}" for c in range(11, 18)]
        )
        assert result == sorted(expected)

    def test_range_short_form(self):
        # "N12-16" — bare second column with no row letter, same row implied
        assert parse_colony_codes("N12-16") == ["N12", "N13", "N14", "N15", "N16"]

    def test_invalid_column_skipped(self):
        # column 25 is out of range (1..24), should be silently skipped
        assert parse_colony_codes("A25") == []
        # Q1 — Q is out of A..P range
        assert parse_colony_codes("Q1") == []

    def test_uppercase_normalisation(self):
        assert parse_colony_codes("a1, b2") == ["A01", "B02"]


class TestAddContaminationColumn:
    def _make_total_df(self):
        return pd.DataFrame(
            [
                {"plate": "32v1", "measurement": "M3", "start_date": "2024-08-06", "well_id": "L01"},
                {"plate": "32v1", "measurement": "M3", "start_date": "2024-08-06", "well_id": "A01"},
                {"plate": "32v1", "measurement": "M5", "start_date": "2024-08-08", "well_id": "L01"},
                {"plate": "99",   "measurement": "M1", "start_date": "2023-10-12", "well_id": "B02"},
            ]
        )

    def test_flags_matching_wells(self):
        total = self._make_total_df()
        contam = pd.DataFrame(
            [
                {"plate": "32v1", "measurement": "M3", "start_date": "2024-08-06", "well_id": "L01"},
            ]
        )
        out = add_contamination_column(total, contam)
        assert "contamination" in out.columns
        assert out["contamination"].tolist() == [1, 0, 0, 0]

    def test_no_flag_when_measurement_differs(self):
        total = self._make_total_df()
        contam = pd.DataFrame(
            [{"plate": "32v1", "measurement": "M3", "start_date": "2024-08-06", "well_id": "L01"}]
        )
        out = add_contamination_column(total, contam)
        m5_row = out[(out["measurement"] == "M5") & (out["plate"] == "32v1")]
        assert m5_row["contamination"].iloc[0] == 0

    def test_no_flag_when_start_date_differs(self):
        # Two physical runs of 20-M3 on different dates: only the matching date's wells flag.
        total = pd.DataFrame(
            [
                {"plate": "20", "measurement": "M3", "start_date": "2024-04-08", "well_id": "M14"},
                {"plate": "20", "measurement": "M3", "start_date": "2024-05-29", "well_id": "M14"},
            ]
        )
        contam = pd.DataFrame(
            [{"plate": "20", "measurement": "M3", "start_date": "2024-04-08", "well_id": "M14"}]
        )
        out = add_contamination_column(total, contam)
        assert out.set_index("start_date")["contamination"].to_dict() == {
            "2024-04-08": 1,
            "2024-05-29": 0,
        }

    def test_empty_contamination_df(self):
        total = self._make_total_df()
        contam = pd.DataFrame(columns=["plate", "measurement", "start_date", "well_id"])
        out = add_contamination_column(total, contam)
        assert (out["contamination"] == 0).all()

    def test_dtype_is_int(self):
        total = self._make_total_df()
        contam = pd.DataFrame(
            [{"plate": "32v1", "measurement": "M3", "start_date": "2024-08-06", "well_id": "L01"}]
        )
        out = add_contamination_column(total, contam)
        assert out["contamination"].dtype.kind == "i"
