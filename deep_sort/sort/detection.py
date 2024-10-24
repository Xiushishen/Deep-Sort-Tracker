import numpy as np

class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    
    Parameters
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=float) # x1, y1, w, h
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
    
    def to_tlbr(self):
        """
        (x1, y1, w, h) -> (x1, y1, x2, y2)
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    def to_xyah(self):
        """
        (x1, y1, w, h) -> (x_c, y_c, a, height) # a : width / height
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

if __name__ == '__main__':
    tlwh = [50, 60, 100, 200]
    confidence = 0.9
    feature = np.random.rand(128)

    det = Detection(tlwh, confidence, feature)
    
    tlbr = det.to_tlbr()
    print(f"to_tlbr: {tlbr}")
    
    xyah = det.to_xyah()
    print(f"to_xyah: {xyah}")