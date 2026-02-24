import unittest
from datetime import datetime

from chlamy_impi.database_creation.utils import parse_name, spreadsheet_plate_name_formatting


class TestSpreadsheetPlateNameFormatting(unittest.TestCase):

        def test_spreadsheet_plate_name_formatting_valid_input_1(self):
            result = spreadsheet_plate_name_formatting("Plate 01")
            self.assertEqual(result, "1")

        def test_spreadsheet_plate_name_formatting_valid_input_2(self):
            result = spreadsheet_plate_name_formatting("Plate 10")
            self.assertEqual(result, "10")

        def test_spreadsheet_plate_name_formatting_valid_input_3(self):
            result = spreadsheet_plate_name_formatting("Plate 100")
            self.assertEqual(result, "100")

        def test_spreadsheet_plate_name_formatting_invalid_input(self):
            with self.assertRaises(AssertionError):
                spreadsheet_plate_name_formatting("Invalid plate name")


class TestParseName(unittest.TestCase):

    def test_parse_name_valid_input_1(self):
        result = parse_name("20200303_7-M4_2h-2h.npy")
        self.assertEqual(result, (7, 'M4', '2h-2h'))

    def test_parse_name_valid_input_2(self):
        result = parse_name("20231119_07-M3_20h_ML.npy")
        self.assertEqual(result, (7, 'M3', '20h_ML'))

    def test_parse_name_valid_input_3(self):
        result = parse_name("20231213_9-M5_2h-2h.npy")
        self.assertEqual(result, (9, 'M5', '2h-2h'))

    def test_parse_name_valid_input_4(self):
        result = parse_name("20231213_99-M5_2h-2h.npy")
        self.assertEqual(result, (99, 'M5', '2h-2h'))

    def test_parse_name_valid_input_5(self):
        result = parse_name("20231213_99-M5_2h-2h.npy", return_date=True)
        self.assertEqual(result, (99, 'M5', '2h-2h', datetime(year=2023, month=12, day=13)))

    def test_parse_name_invalid_input(self):
        with self.assertRaises(AssertionError):
            parse_name("invalid_filename.npy")

if __name__ == '__main__':
    unittest.main()
