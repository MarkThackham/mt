import unittest
import sys

sys.path.append('./src')  # Add the path to the src directory
from mt.univariate import gini_coefficient

class TestGiniCoefficient(unittest.TestCase):
    def test_gini_coefficient1(self):
        # Test case 1: Perfect prediction
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1]
        expected_gini = 1.0
        self.assertAlmostEqual(gini_coefficient(y_true, y_pred), expected_gini, delta=0.00001)

    def test_gini_coefficient2(self):
        # Test case 2: Random prediction
        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]
        expected_gini = 0.75
        self.assertAlmostEqual(gini_coefficient(y_true, y_pred), expected_gini, delta=0.00001)

if __name__ == '__main__':
    unittest.main()
