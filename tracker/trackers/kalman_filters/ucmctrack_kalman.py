from .base_kalman import BaseKalman
import numpy as np 

class UCMCKalman(BaseKalman):
    def __init__(self, ):

        state_dim = 8
        observation_dim = 4 

        F = np.eye(state_dim, state_dim)
        '''
        [1, 0, 0, 0, 1, 0, 0]
        [0, 1, 0, 0, 0, 1, 0]
        ...
        '''
        for i in range(state_dim // 2):
            F[i, i + state_dim // 2] = 1

        H = np.eye(state_dim // 2, state_dim)
    
        super().__init__(state_dim=state_dim, 
                       observation_dim=observation_dim, 
                       F=F, 
                       H=H)
        
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160