from .base_kalman import BaseKalman
import numpy as np 
from copy import deepcopy

class HybridSORTKalman(BaseKalman):

    def __init__(self, ):
        
        state_dim = 9  # [x, y, s, c, a, vx, vy, vs, vc]  s: area  c: confidence score
        observation_dim = 5  # confidence score is additional 

        F = np.eye(state_dim)
        for i in range(4):
            F[i, (state_dim + 1) // 2 + i] = 1  # x = x + vx, y = y + vy, s = s + vs, c = c + vc in predict step

        H = np.eye(state_dim // 2 + 1, state_dim)
    
        super().__init__(state_dim=state_dim, 
                       observation_dim=observation_dim, 
                       F=F, 
                       H=H)
        
        # TODO check
        # give high uncertainty to the unobservable initial velocities
        self.kf.R[2:, 2:] *= 10  # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]
        self.kf.P[5:, 5:] *= 1000
        self.kf.P *= 10 
        self.kf.Q[-1, -1] *= 0.01 # score
        self.kf.Q[-2, -2] *= 0.01 
        self.kf.Q[5:, 5:] *= 0.01 
    
    def initialize(self, observation):
        """
        Args:
            observation: x-y-s-c-a
        """
        self.kf.x = self.kf.x.flatten()
        self.kf.x[:5] = observation


    def predict(self, is_activated=True):
        """ predict step
        
        """
        
        # s + vs
        if (self.kf.x[7] + self.kf.x[2] <= 0):
            self.kf.x[7] *= 0.0

        self.kf.predict()


    def update(self, z):
        """ update step
        
        Args:
            z: observation x-y-s-a format
        """
        
        self.kf.update(z)


