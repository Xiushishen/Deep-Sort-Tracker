import numpy as np
import os
import sys
import unittest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(parent_dir)
sys.path.insert(0, parent_dir)

from deep_sort.sort import nn_matching

class TestDistanceFunction(unittest.TestCase):
    def setUp(self):
        # features = np.array([
        #     [2, 3],
        #     [2, 3],
        #     [2, 3]
        # ])
        # initial_features = np.array([
        #     [1, 1],
        #     [1, 2],
        #     [1, 3]
        # ])
        # targets = [1, 2, 3]
        # initial_targets = [1, 2, 3]
        # active_targets = [1, 2, 3]

        self.fn = nn_matching.NearestNeighborDistanceMetric(
                               metric='euclidean', 
                               matching_threshold=1.5)
    
    def test_distance_basic_2d(self):
        features = np.array([
            [1, 1],
            [2, 2],
        ])
        initial_features = np.array([
            [2, 2],
            [1, 1],
        ])
        targets = [1, 2]
        initial_targets = [1, 2]
        active_targets = [1, 2]
        expected_cost_matrix = np.array([
            [2, 0],
            [0, 2],
        ])
        self.fn.partial_fit(initial_features, initial_targets, active_targets)
        cost_matrix = self.fn.distance(features, targets)
        np.testing.assert_array_almost_equal(cost_matrix, expected_cost_matrix)
    
    def test_distance_basic_3d(self):
        features = np.array([
            [2, 3],
            [2, 3],
            [2, 3]
        ])
        initial_features = np.array([
            [1, 1],
            [1, 2],
            [1, 3]
        ])
        targets = [1, 2, 3]
        initial_targets = [1, 2, 3]
        active_targets = [1, 2, 3]
        expected_cost_matrix = np.array([
            [5, 5, 5],
            [2, 2, 2],
            [1, 1, 1]
        ])
        self.fn.partial_fit(initial_features, initial_targets, active_targets)
        cost_matrix = self.fn.distance(features, targets)
        np.testing.assert_array_almost_equal(cost_matrix, expected_cost_matrix)

    def test_distance_basic_2d_1(self):
        features = np.array([
            [0, 0],
            [0, 0],
        ])
        initial_features = np.array([
            [1, 1],
            [1, 1],
        ])
        targets = [1, 2]
        initial_targets = [1, 2]
        active_targets = [1, 2]
        expected_cost_matrix = np.array([
            [2, 2],
            [2, 2],
        ])
        self.fn.partial_fit(initial_features, initial_targets, active_targets)
        cost_matrix = self.fn.distance(features, targets)
        np.testing.assert_array_almost_equal(cost_matrix, expected_cost_matrix)

if __name__ == "__main__":
    unittest.main()