import control
import control.matlab
import numpy as np
import matplotlib.pyplot as plt
from .euler import euler
from .thermistor import THERMISTOR

#Models the MAX1968 TEC controller
class TEC_DRIVER:

    _ss = None
    _vctli = None
    _itec = None
    vref = 1.5
    rsense = 0.05
    imax = 3.0
    xkm1 = None

    def __init__(self, thermistor, tec_set_point_temperature):
        self.s = control.matlab.tf('s')
        self.calc_transfer_function()
        self._vctli = 0
        self._itec = 0
        self._thermistor: THERMISTOR = thermistor
        self._thermistor_resistance_at_set_point = thermistor.calc_resistance(tec_set_point_temperature)


    # MAX 1968 controller
    def calc_transfer_function(self, R1=510e3, R2=10e3, C2=1e-6, C1=0.022e-6, R3=240e3, C3=10e-6, gain=-10):
        s = self.s
        Y1 = 1/R1 + 1/(R2 + 1/(s*C2))
        Z1 = 1/Y1
        Y2 = 1/(R3 + 1/(s*C3)) + s*C1
        Z2 = 1/Y2
        H = -Z2/Z1*gain
        self._ss = control.tf2ss(H.num, H.den)

    def plot_bode(self):
        print('Bode plot')
        control.bode(self._ss)
        plt.show()
    
    def update(self, r_thermistor, dt, init=False):
        error = -(self.vref*self._thermistor_resistance_at_set_point/(self._thermistor_resistance_at_set_point + 10e3) - self.vref*r_thermistor/(r_thermistor + 10e3))
        if init:
            self._xkm1 = np.zeros((np.shape(self._ss.A)[0], 1))
        self._vctli, self._xkm1 = euler(self._ss.A, self._ss.B, self._ss.C, self._ss.D, dt, self._xkm1, error)
        self._vctli = self._vctli[0,0]
        if self._vctli < 0:
            self._vctli = 0
        elif self._vctli > 5:
            self._vctli = 5
        self._itec =  (self._vctli - self.vref)/(10*self.rsense)
        if self._itec > self.imax:
            self._itec = self.imax
        elif self._itec < -self.imax:
            self._itec = -self.imax
        return self._itec
