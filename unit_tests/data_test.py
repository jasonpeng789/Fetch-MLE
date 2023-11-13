import unittest
import pandas as pd
import numpy as np
from io import StringIO
import os
import sys
sys.path.append('../core') 
from data import *

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Create a simple CSV for testing purposes
        self.csv_path = "test_data.csv"
        data = {
            "# Date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-02-01", "2022-02-02"],
            "Receipt_Count": [5, 6, 7, 8, 9]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)

    def test_load_csv_file(self):
        # Test load_csv_file function
        df = load_csv_file(self.csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertTrue("# Date" in df.columns)
        self.assertTrue("Receipt_Count" in df.columns)

    def test_data_process(self):
        # Test data_process function
        df_processed = data_process(self.csv_path)
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertTrue(df_processed.index.is_monotonic_increasing)
        self.assertFalse(df_processed.isnull().values.any())

    def test_create_sequences(self):
        # Test create_sequences function
        data = np.arange(10)
        sequences, targets = create_sequences(data, 3)
        self.assertEqual(len(sequences), len(targets))
        self.assertTrue(np.array_equal(sequences[0], np.array([0, 1, 2])))

    def test_normalize_denormalize(self):
        # Test normalize and denormalize functions
        data = np.array([1, 2, 3, 4, 5])
        mean, std = data.mean(), data.std()
        normalized_data = normalize(data, mean, std)
        denormalized_data = denormalize(normalized_data, mean, std)
        np.testing.assert_array_almost_equal(data, denormalized_data)

    def test_data_split(self):
        # Test data_split function
        X, y = np.arange(10).reshape((5, 2)), np.arange(5)
        X_train, y_train, X_val, y_val = data_split(X, y, dl=True)
        self.assertEqual(X_train.shape[0], 4)  # 80% of 5 is 4
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)

    def tearDown(self):
        # Clean up the CSV file after tests
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

if __name__ == '__main__':
    unittest.main()
