import numpy as np
from .euler import euler

class THERMISTOR:
    _temperature_thermistor: float = None
    A = None
    def __init__(self, theta_jsm, tau, beta=3950, temperature_ambient=298.15, resistance=10000.0):
        self._theta_jsm = theta_jsm
        self._tau = tau
        self._c_thermal = self._tau / self._theta_jsm
        self._beta = beta
        self._temperature_amb = temperature_ambient
        self._resistance_at_ambient = resistance
        
    def calc_state_space(self):
        self.A =  np.array([-1 / self._tau]).reshape((1, 1))
        self.B = np.array([1 / self._tau]).reshape((1, 1))
        self.C = np.array([1]).reshape((1, 1))
        self.D = np.array([0]).reshape((1, 1))
    
    def calc_resistance(self, temperature=None):
        if temperature is not None:
            self._temperature_thermistor = temperature
        self._resistance = self._resistance_at_ambient * np.exp(self._beta * (1 / (self._temperature_thermistor) - 1 / (self._temperature_amb)))
        return self._resistance

    def update(self, Ta, dt, init=False):
        if self.A is None:
            self.calc_state_space()

        if init:
            self._temperature_thermistor = Ta
        else:
            self._temperature_thermistor = euler(self.A, self.B, self.C, self.D, dt, self._temperature_thermistor, Ta)
            self._temperature_thermistor = self._temperature_thermistor[0][0]
        self.calc_resistance()