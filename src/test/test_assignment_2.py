import unittest
import math
import numpy as np

from src.main.assignment_2 import (
    neville,
    forward_difference_table,
    newton_forward_poly,
    hermite_divided_difference,
    cubic_spline_system
)

class TestAssignment2(unittest.TestCase):

    def test_neville(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        x0 = 3.7
        result = neville(x_vals, y_vals, x0)
        expected = 1.555  # Updated expected value
        self.assertAlmostEqual(result, expected, places=3)

    def test_forward_difference_table(self):
        x_data = [7.2, 7.4, 7.5, 7.6]
        y_data = [23.5492, 25.3913, 26.8224, 27.4589]
        table = forward_difference_table(x_data, y_data)
        expected_first_diff = [25.3913 - 23.5492, 26.8224 - 25.3913, 27.4589 - 26.8224]
        for computed, expected in zip(table[1], expected_first_diff):
            self.assertAlmostEqual(computed, expected, places=3)

    def test_newton_forward_poly(self):
        x_data = [7.2, 7.4, 7.5, 7.6]
        y_data = [23.5492, 25.3913, 26.8224, 27.4589]
        table = forward_difference_table(x_data, y_data)
        h = x_data[1] - x_data[0]
        x_eval = 7.3
        result_deg1 = newton_forward_poly(x_eval, table, x_data[0], 1, h)
        result_deg2 = newton_forward_poly(x_eval, table, x_data[0], 2, h)
        result_deg3 = newton_forward_poly(x_eval, table, x_data[0], 3, h)
        # Use delta to allow a tolerance of 0.1 in the comparison.
        self.assertAlmostEqual(result_deg1, result_deg2, delta=0.1)
        self.assertAlmostEqual(result_deg2, result_deg3, delta=0.1)

    def test_hermite_divided_difference(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        dy_vals = [-1.195, -1.188, -1.182]
        z, Q = hermite_divided_difference(x_vals, y_vals, dy_vals)
        # For the first repeated node, the first divided difference should equal the derivative.
        self.assertAlmostEqual(Q[1][1], dy_vals[0], places=3)

    def test_cubic_spline_system(self):
        x_points = [2, 5, 8, 10]
        y_points = [3, 5, 7, 9]
        A, b = cubic_spline_system(x_points, y_points)
        # For 4 points, we have 2 interior nodes.
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(len(b), 2)
        M_interior = np.linalg.solve(A, b)
        expected_M = np.array([-0.05405405, 0.21621622])
        np.testing.assert_allclose(M_interior, expected_M, atol=1e-3)

if __name__ == '__main__':
    unittest.main()
