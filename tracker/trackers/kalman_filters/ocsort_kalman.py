from .base_kalman import BaseKalman
import numpy as np 
from copy import deepcopy

class OCSORTKalman(BaseKalman):

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

    def _freeze(self, ):
        """ freeze all the param of Kalman
        
        """
        self.attr_saved = deepcopy(self.kf.__dict__)

    def _unfreeze(self, ):
        """ when observe an lost object again, use the virtual trajectory
        
        """
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)
            self.kf.__dict__ = self.attr_saved 

            self.history_obs = self.history_obs[:-1]

            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur)==0)[0]
            index1 = indices[-2]
            index2 = indices[-1]
            box1 = new_history[index1]
            x1, y1, s1, r1 = box1 
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)
            box2 = new_history[index2]
            x2, y2, s2, r2 = box2 
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx = (x2-x1)/time_gap
            dy = (y2-y1)/time_gap 
            dw = (w2-w1)/time_gap 
            dh = (h2-h1)/time_gap
            for i in range(index2 - index1):
                """
                    The default virtual trajectory generation is by linear
                    motion (constant speed hypothesis), you could modify this 
                    part to implement your own. 
                """
                x = x1 + (i+1) * dx 
                y = y1 + (i+1) * dy 
                w = w1 + (i+1) * dw 
                h = h1 + (i+1) * dh
                s = w * h 
                r = w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                """
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for 
                    easy read and understanding
                """
                self.kf.update(new_box)
                if not i == (index2-index1-1):
                    self.kf.predict()


    def update(self, z):
        """ update step

        For simplicity, directly change the self.kf as OCSORT modify the intrinsic Kalman
        
        Args:
            z: observation x-y-s-a format
        """

        self.history_obs.append(z)

        if z is None:
            if self.observed:
                self._freeze()
                self.observed = False
            
            self.kf.update(z)

            return 

        else:
            if not self.observed:  # Get observation, use online smoothing to re-update parameters
                self._unfreeze()
            
            self.kf.update(z)

        self.observed = True 


