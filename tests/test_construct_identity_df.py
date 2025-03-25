import unittest
import pandas as pd
from chlamy_impi.database_creation.construct_identity_df import construct_identity_dataframe

class TestConstructIdentityDataFrame(unittest.TestCase):

    def setUp(self):
        # Create a mock mutation dataframe
        self.mutation_df = pd.DataFrame({
            'mutant_ID': ['LMJ.RY0401.001', 'LMJ.RY0401.002', 'WT'],
            'confidence_level': [3, 4, 6],
            'gene': ['gene1', 'gene2', '']
        })

    def test_construct_identity_dataframe(self):
        # Call the function with the mock data
        result_df = construct_identity_dataframe(self.mutation_df, conf_threshold=5)

        # Check the shape of the resulting dataframe
        self.assertEqual(result_df.shape[1], 6)  # Expecting 6 columns

        # Check that the dataframe contains the expected columns
        expected_columns = ['mutant_ID', 'plate', 'well_id', 'feature', 'mutated_genes', 'num_mutations']
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

        # Check that the dataframe contains the expected number of rows
        self.assertGreaterEqual(len(result_df), 1)


if __name__ == '__main__':
    unittest.main()