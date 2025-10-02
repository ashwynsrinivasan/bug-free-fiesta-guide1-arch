import numpy as np
from LaserTherm.euler import euler
class thermistor:
    T_therm: float = None
    A = None
    def __init__(self, theta_jsm, tau):
        self.theta_jsm = theta_jsm
        self.tau = tau
        self.C = self.tau/self.theta_jsm
    def calc_state_space(self):
        self.A =  np.array([-1/self.tau]).reshape((1, 1))
        self.B = np.array([1/self.tau]).reshape((1, 1))
        self.C = np.array([1]).reshape((1, 1))
        self.D = np.array([0]).reshape((1, 1))

    def update(self, Ta, dt, init=False):
        if self.A is None:
            self.calc_state_space()

        if init:
            self.T_therm = Ta
        else:
            self.T_therm = euler(self.A, self.B, self.C, self.D, dt, self.T_therm, Ta)
            self.T_therm = self.T_therm[0][0]