import numpy as np
import os
import sys
import unittest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from deep_sort.sort import preprocessing

class TestNonMaxSuppression(unittest.TestCase):
    def test_nms_basic(self):
        boxes = np.array([
            [100, 100, 50, 50],
            [105, 105, 50, 50],
            [200, 200, 50, 50]
        ])
        scores = np.array([0.9, 0.85, 0.8])
        max_bbox_overlap = 0.5
        expected_pick = [0, 2]
        pick = preprocessing.non_max_suppression(boxes, max_bbox_overlap, scores)
        self.assertEqual(pick, expected_pick)

    def test_nms_no_boxes(self):
        boxes = np.array([])
        max_bbox_overlap = 0.5
        expected_pick = []
        pick = preprocessing.non_max_suppression(boxes, max_bbox_overlap)
        self.assertEqual(pick, expected_pick)

    def test_nms_single_box(self):
        boxes = np.array([
            [100, 100, 50, 50]
        ])
        max_bbox_overlap = 0.5
        expected_pick = [0]
        pick = preprocessing.non_max_suppression(boxes, max_bbox_overlap)
        self.assertEqual(pick, expected_pick)

    def test_nms_all_overlapping(self):
        boxes = np.array([
            [100, 100, 50, 50],
            [102, 102, 50, 50],
            [104, 104, 50, 50]
        ])
        scores = np.array([0.9, 0.85, 0.8])
        max_bbox_overlap = 0.5
        expected_pick = [0]
        pick = preprocessing.non_max_suppression(boxes, max_bbox_overlap, scores)
        self.assertEqual(pick, expected_pick)

if __name__ == "__main__":
    unittest.main()