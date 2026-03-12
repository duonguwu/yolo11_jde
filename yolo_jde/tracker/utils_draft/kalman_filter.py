"""Kalman filter implementations for object tracking."""

import numpy as np
import scipy.linalg


# Chi-square distribution quantiles for Mahalanobis gating
chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}


class KalmanFilterXYAH:
    """
    Kalman filter for tracking bounding boxes in image space.
    
    State vector: [x, y, a, h, vx, vy, va, vh]
    - (x, y): center position
    - a: aspect ratio (w/h)
    - h: height
    - (vx, vy, va, vh): velocities
    """

    def __init__(self):
        """Initialize Kalman filter model matrices."""
        ndim, dt = 4, 1.0

        # Create motion matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """Create track from measurement [x, y, a, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, overwrite_a=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, overwrite_b=True, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, 
                       only_position: bool = False, metric: str = 'maha') -> np.ndarray:
        """Compute gating distance between state distribution and measurements."""
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements.shape[-1]
        mean, covariance = self.project(mean, covariance)
        if metric == 'gaussian':
            return np.sum((measurements - mean) ** 2, axis=1, keepdims=True)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0, keepdims=True)  # square maha
        else:
            raise ValueError('Invalid distance metric')
    
    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> tuple:
        """Run Kalman filter prediction step for multiple tracks."""
        if len(means) == 0:
            return means, covariances
            
        for i in range(len(means)):
            means[i], covariances[i] = self.predict(means[i], covariances[i])
        
        return means, covariances


class KalmanFilterXYWH:
    """
    Kalman filter for tracking bounding boxes using [x, y, w, h] representation.
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): center position
    - (w, h): width and height
    - (vx, vy, vw, vh): velocities
    """

    def __init__(self):
        """Initialize Kalman filter model matrices."""
        ndim, dt = 4, 1.0

        # Create motion matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """Create track from measurement [x, y, w, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],  # x std based on width
            2 * self._std_weight_position * measurement[3],  # y std based on height
            2 * self._std_weight_position * measurement[2],  # w std
            2 * self._std_weight_position * measurement[3],  # h std
            10 * self._std_weight_velocity * measurement[2], # vx std
            10 * self._std_weight_velocity * measurement[3], # vy std
            10 * self._std_weight_velocity * measurement[2], # vw std
            10 * self._std_weight_velocity * measurement[3], # vh std
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],  # x std based on width
            self._std_weight_position * mean[3],  # y std based on height
            self._std_weight_position * mean[2],  # w std
            self._std_weight_position * mean[3],  # h std
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],  # vx std
            self._std_weight_velocity * mean[3],  # vy std
            self._std_weight_velocity * mean[2],  # vw std
            self._std_weight_velocity * mean[3],  # vh std
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[2],  # x std
            self._std_weight_position * mean[3],  # y std
            self._std_weight_position * mean[2],  # w std
            self._std_weight_position * mean[3],  # h std
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, overwrite_a=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, overwrite_b=True, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, 
                       only_position: bool = False, metric: str = 'maha') -> np.ndarray:
        """Compute gating distance between state distribution and measurements."""
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        mean, covariance = self.project(mean, covariance)
        if metric == 'gaussian':
            return np.sum((measurements - mean) ** 2, axis=1, keepdims=True)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0, keepdims=True)  # square maha
        else:
            raise ValueError('Invalid distance metric')
