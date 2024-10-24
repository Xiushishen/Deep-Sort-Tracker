import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees
of freedom, which is used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

class KalmanFilter(object):
    """
    The 8-dimensional state space
    x, y, a, h, vx, vy, va, vh

    Parameter
    ---------
    _motion_mat : (8 x 8)
        F matrix
    _update_mat : (4 x 8)
        H matrix
    _std_weight_position : double
        weight of standard deviation of position
    _std_weight_velocity : double
        weight of standard deviation of velocity
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement):
        mean_pos = measurement # position state vector (4, )
        mean_vel = np.zeros_like(mean_pos) # (4, )
        mean = np.r_[mean_pos, mean_vel] # (8 x 1)
        
        # pedestrian
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ] # position and velocity standard deviation (8, )
        covariance = np.diag(np.square(std)) # position and velocity coviarance matrix (8 x 8)
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        mean : (8 x 1) [position, velocity]
        covariance : (8 x 8) [position, velocity]
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        Project the predicted state to measurement distribution

        Parameter
        ---------
        mean : (8 x 1)
        covariance : (8 x 8)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        Estimate the best state based on prediction and measurement
        
        Parameter
        ---------
        mean : (8 x 1)
        covariance : (8 x 8)
        measurement : (xc, yc, a, h)
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        cho_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (cho_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance between state distribution and measurements.

        Parameter
        ---------
        mean : ndarray
            Mean vector over the state distribution (8 x 1).
        covariance : ndarray
            Covariance of the state distribution (8 x 8).
        measurements : ndarray
            (xc, yc, a, h)
        only_position : Optional[bool]
            if we only care about position difference.

        Returns
        -------
        squared_maha : ndarray
            Returns an array of length N, where the i-th element contains the 
            squared Mahalanobis distance between (mean, covariance) and 
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

if __name__ == '__main__':
    kf = KalmanFilter()
    mean = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    covariance = np.eye(8)
    mean1, covariance1 = kf.predict(mean, covariance)
    print("Mean: ", mean1)
    print("Covariance: ", covariance1)
    





