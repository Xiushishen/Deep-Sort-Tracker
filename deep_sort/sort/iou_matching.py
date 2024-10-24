import numpy as np
from . import linear_assignment

def iou(bbox, candidates):
    """ Intersection over union
    Parameters
    ----------
    bbox : A bounding box in format (x_tl, y_tl, width, height)
    candidates : A matrix of candidates bounding boxes in the same format as 'bbox'.
    
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the 'bbox' and each candidate. A
        higher score means a large fraction of the 'bbox' is occluded by the candidate.
    """

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    # print(tl)
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric"""
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if (tracks[track_idx].time_since_update > 1):
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix

if __name__ == "__main__":
    bbox = np.array([50, 50, 100, 100])
        
    # Define candidate bounding boxes
    candidates = np.array([
        [50, 50, 100, 100],  # Exact match
        [60, 60, 80, 80],    # Fully inside
        [0, 0, 50, 50],      # No overlap
        [50, 50, 50, 50],    # Partial overlap
        [100, 100, 100, 100] # Partial overlap with top left corner
    ])
    res = iou(bbox, candidates)

