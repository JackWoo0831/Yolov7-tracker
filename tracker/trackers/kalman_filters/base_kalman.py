from filterpy.kalman import KalmanFilter
import numpy as np 
import scipy

class BaseKalman:

    def __init__(self,
                 state_dim: int = 8, 
                 observation_dim: int = 4, 
                 F: np.ndarray = np.zeros((0, )), 
                 P: np.ndarray = np.zeros((0, )),  
                 Q: np.ndarray = np.zeros((0, )),  
                 H: np.ndarray = np.zeros((0, )), 
                 R: np.ndarray = np.zeros((0, )), 
                 ) -> None:
        
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=observation_dim, dim_u=0)
        if F.shape[0] > 0: self.kf.F = F  # if valid 
        if P.shape[0] > 0: self.kf.P = P 
        if Q.shape[0] > 0: self.kf.Q = Q 
        if H.shape[0] > 0: self.kf.H = H 
        if R.shape[0] > 0: self.kf.R = R 

    def initialize(self, observation):
        return NotImplementedError

    def predict(self, ):
        self.kf.predict()

    def update(self, observation, **kwargs):
        self.kf.update(observation, self.R, self.H)

    def get_state(self, ):
        return self.kf.x
    
    def gating_distance(self, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, note the format (whether xywh or xyah or others)
            should be identical to state definition
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        
        # map state space to measurement space
        mean = self.kf.x.copy()
        mean = np.dot(self.kf.H, mean)
        covariance = np.linalg.multi_dot((self.kf.H, self.kf.P, self.kf.H.T))

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

    