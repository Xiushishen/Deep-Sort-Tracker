import numpy as np
import os
import sys
import unittest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(parent_dir)
sys.path.insert(0, parent_dir)

from deep_sort.sort import iou_matching

class TestIOUFunction(unittest.TestCase):
    
    def test_iou(self):
        # Define the bounding box
        bbox = np.array([50, 50, 100, 100])
        
        # Define candidate bounding boxes
        candidates = np.array([
            [50, 50, 100, 100],  # Exact match
            [60, 60, 80, 80],    # Fully inside
            [0, 0, 50, 50],      # No overlap
            [50, 50, 50, 50],    # Partial overlap
            [100, 100, 100, 100] # Partial overlap with top left corner
        ])
        
        # Expected IOU results
        expected_iou = np.array([1.0, 0.64, 0.0, 0.25, 0.14285714])
        
        # Calculate IOU using the provided function
        calculated_iou = iou_matching.iou(bbox, candidates)
        
        # Assert the expected and calculated IOU values are close
        np.testing.assert_almost_equal(calculated_iou, expected_iou, decimal=6)
        
if __name__ == '__main__':
    unittest.main()
