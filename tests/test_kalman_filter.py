import numpy as np
import os
import sys
import unittest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# print(parent_dir)
sys.path.insert(0, parent_dir)

from deep_sort.sort import kalman_filter

class TestDistanceFunction(unittest.TestCase):
    def setUp(self):
        self.fn = kalman_filter.KalmanFilter()

    def test_kf_basic(self):
        mean = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        covariance = np.eye(8)
        mean, covariance = self.fn.predict(mean, covariance)
        expected_mean = np.array([3, 3, 3, 3, 2, 2, 2, 2])
        expected_covariance = np.array([
            [2., 0., 0., 0., 1., 0., 0., 0.],
            [0., 2., 0., 0., 0., 1., 0., 0.],
            [0., 0., 2., 0., 0., 0., 1., 0.],
            [0., 0., 0., 2., 0., 0., 0., 1.],
            [1., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0., 0., 0., 1.]
        ])
        np.testing.assert_array_almost_equal(covariance, expected_covariance, decimal=2)
        self.assertEqual(covariance.shape, expected_covariance.shape)
        np.testing.assert_almost_equal(expected_mean, mean)

if __name__ == "__main__":
    unittest.main()