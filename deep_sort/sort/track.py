class TrackState:
    """
    Tentative : newly created tracks are classified as `tentative` until enough
    evidence has been collected.
    Confirmed : track state is changed to `confirmed` with enough evidence collection.
    Deleted : tracks that are no longer alive and mark them for removal from the set
    of active tracks.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated velocities.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution. (8x1)
    covariance : ndarray
        Covariance matrix of the initial state distribution. (8 x 8)
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Number of frames since last update. (Number of fail matches)
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    _n_init : int 
        Number of times needed from tentative to confirmed state.
    _max_age : int
        Number of times needed from tentative to deleted state.
    """
    def __init__(self, mean, covariace, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariace
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        
        self._n_init = n_init
        self._max_age = max_age
    
    def to_tlwh(self):
        """
        (xc, yc, a, h) -> (x1, y1, w, h)
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_tlbr(self):
        """
        (xc, yc, a, h) -> (x1, y1, x2, y2)
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection):
        """
        Perform kalman filter measurement update step and update the feature cache.

        Parameters
        ----------
        kf : The Kalman Filter.
        detection : The associated detection.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """
        Mark this track as missed.
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        return self.state == TrackState.Deleted