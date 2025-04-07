from .base_kalman import BaseKalman
import numpy as np 

class UCMCKalman(BaseKalman):
    def __init__(self, 
                 sigma_x=1.0,  # noise factor in x axis (Eq. 11)
                 sigma_y=1.0,  # noise factor in y axis (Eq. 11)
                 vmax=1.0,  # TODO
                 dt=1/30,  # interval between frames
                 **kwargs
                 ):

        state_dim = 4  # [x, dx, y, dy]  where x, y is the corr on ground
        observation_dim = 2   # [x, y]

        F = np.array([[1, dt, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 1, dt], 
                      [0, 0, 0, 1]])

        H = np.array([[1, 0, 0, 0], 
                      [0, 0, 1, 0]])
        
        P = np.array([[1, 0, 0, 0], 
                      [0, vmax**2 / 3.0, 0, 0], 
                      [0, 0, 1, 0], 
                      [0, 0, 0, vmax**2 / 3.0]])
        
        # noise compensation to initialize Q by Eq. 10 and 11
        G = np.array([[0.5*dt*dt, 0], 
                      [dt, 0], 
                      [0, 0.5*dt*dt], 
                      [0, dt]])
        Q0 = np.array([[sigma_x, 0], [0, sigma_y]])
        Q = np.dot(np.dot(G, Q0), G.T)
    
        super().__init__(state_dim=state_dim, 
                       observation_dim=observation_dim, 
                       F=F, 
                       H=H, 
                       P=P,
                       Q=Q)
        
    def initialize(self, observation, R):
        """
        observation: [x, y]  where x y is the grounding coordinate
        R: the cov matrix of observation (2, 2)
        """
        
        self.kf.x[0] = observation[0]
        self.kf.x[1] = 0
        self.kf.x[2] = observation[1]
        self.kf.x[3] = 0

        self.kf.R = R

    def predict(self, is_activated=True):
        self.kf.predict()

    def update(self, z, R):
        self.kf.update(z=z, R=R)