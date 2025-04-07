from numpy.core.multiarray import zeros as zeros
from .base_kalman import BaseKalman
import numpy as np 
from copy import deepcopy

class SORTKalman(BaseKalman):

    def __init__(self, ):
        
        state_dim = 7  # [x, y, s, a, vx, vy, vs]  s: area
        observation_dim = 4 

        F = np.array([[1, 0, 0, 0, 1, 0, 0], 
                      [0, 1, 0, 0, 0, 1, 0], 
                      [0, 0, 1, 0, 0, 0, 1], 
                      [0, 0, 0, 1, 0, 0, 0],  
                      [0, 0, 0, 0, 1, 0, 0], 
                      [0, 0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 0, 1]])

        H = np.eye(state_dim // 2 + 1, state_dim)
    
        super().__init__(state_dim=state_dim, 
                       observation_dim=observation_dim, 
                       F=F, 
                       H=H)
        
        # TODO check
        # give high uncertainty to the unobservable initial velocities
        self.kf.R[2:, 2:] *= 10  # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]
        self.kf.P[4:, 4:] *= 1000
        self.kf.P *= 10 
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q[4:, 4:] *= 0.01 

        # keep all observations 
        self.history_obs = []
        self.attr_saved = None
        self.observed = False 
    
    def initialize(self, observation):
        """
        Args:
            observation: x-y-s-a
        """
        self.kf.x = self.kf.x.flatten()
        self.kf.x[:4] = observation


    def predict(self, is_activated=True):
        """ predict step
        
        """
        
        # s + vs
        if (self.kf.x[6] + self.kf.x[2] <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

    def update(self, z):
        """ update step

        For simplicity, directly change the self.kf as OCSORT modify the intrinsic Kalman
        
        Args:
            z: observation x-y-s-a format
        """

        self.kf.update(z)



