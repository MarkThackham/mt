import unittest
import sys

import pandas as pd
import numpy as np

sys.path.append('../src')  # Add the path to the src directory
from mt.univariate import gini
from mt.univariate import woe_iv_calc

class Testgini(unittest.TestCase):
    def test_gini1(self):
        # Test case 1: Perfect prediction
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1]
        expected_gini = 1.0
        self.assertAlmostEqual(gini(y_true, y_pred), expected_gini, delta=0.00001)

    def test_gini2(self):
        # Test case 2: Random prediction
        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]
        expected_gini = 0.75
        self.assertAlmostEqual(gini(y_true, y_pred), expected_gini, delta=0.00001)


class Test_woe_iv(unittest.TestCase):

    # Pima Indians Diabetes Data (first 50 rows)
    pima_preg=[
        6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 4, 10, 10, 1, 5, 7, 0, 7, 1, 1, 3, 8, 7, 9, 11, 
        10, 7, 1, 13, 5, 5, 3, 3, 6, 10, 4, 11, 9, 2, 4, 3, 7, 7, 9, 7, 0, 1, 2, 7, 7, 1]
    pima_outcome=[
        1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    df_pid=pd.DataFrame({'Preg': pima_preg, 'Outcome': pima_outcome})

    woe_table, iv_value=woe_iv_calc(
        y_true=df_pid['Outcome'], 
        feature=df_pid['Preg'], 
        num_bins=3)

    woe_values=woe_table['woe'].values.round(4)
    iv_value.round(4)

    def test_woe_iv_calc1(self):
        expected_woe_values = np.array([-0.0661, -0.1839,  0.3269])
        np.testing.assert_almost_equal(self.woe_values, expected_woe_values, decimal=4)

    def test_woe_iv_calc2(self):
        expected_iv_value = np.float64(0.0427)
        self.assertAlmostEqual(self.iv_value, expected_iv_value, delta=0.0001)

if __name__ == '__main__':
    unittest.main()
