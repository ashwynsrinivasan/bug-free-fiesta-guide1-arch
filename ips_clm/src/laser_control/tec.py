import numpy as np
from .euler import euler


class TEC:
    Th = 273 + 50  # Hot side temperature, K
    dTmax = 70  # Maximum temperature difference, K
    Qmax = 34.6  # Maximum heat pumping capacity, W
    Imax = 7  # Maximum current, A
    Umax = 8.8  # Maximum voltage, V
    ACR = 1.06  # Resistance, Ohms
    alpha_m = 0  # Seebeck coefficient, V/K
    Rm = 0  # Electrical resistance, Ohms
    C_therm = 0  # Thermal capacitance, J/K
    qa = 0  # Heat flow, W
    I = 0  # Current, A
    V = 0  # Voltage, V
    Ta = 0  # Absorption surface (cold-side) temperature, K

    A = None  # State Space A matrix
    B = None  # State Space B matrix
    C = None  # State Space C matrix
    D = None  # State Space D matrix
    xk = None  # State Space x vector

    def __init__(self, Th, dTmax, Qmax, Imax, Umax, ACR, time_constant=1):
        self.Th = Th
        self.dTmax = dTmax
        self.Qmax = Qmax
        self.Imax = Imax
        self.Umax = Umax
        self.ACR = ACR

        self.alpha_m = Umax / Th
        self.Rm = Umax / Imax * (Th - dTmax) / Th

        self.theta_m = 2 * Th * dTmax / (Imax * Umax) / (Th - dTmax)  # Thermal resistance
        self.C_therm = time_constant/self.theta_m  # Thermal capacitance

    def calc_ss(self, i, Te, init=False, qa=0):
        # A = 1 / C_therm * [-1 / theta_m - alpha_m * I(ctr)];
        # B = 1 / C_therm * [1, Th / theta_m + I(ctr) ^ 2 * Rm / 2];
        # C = 1;
        # D = [0, 0];
        # if ctr == 1
        #     xkm1 = -A\B * u(1,:)';
        # end
        self.I = i
        self.qa = qa
        self.Th = Te

        self.A = np.array([1 / self.C_therm * (-1 / self.theta_m - self.alpha_m * i)]).reshape((1, 1))
        self.B = 1 / self.C_therm * np.array([1, self.Th / self.theta_m + i**2 * self.Rm / 2]).reshape((1, 2))
        self.C = np.array([1]).reshape((1,1))
        self.D = np.array([0, 0]).reshape((1,2))

        if init:
            u = np.zeros((2, 1))
            u[0] = self.qa
            u[1] = 1
            self.xk = -np.linalg.solve(self.A, np.dot(self.B, u))

    def update(self, qa, i, dt, init=False):
        if i > self.Imax:
            i = self.Imax
        if i < -self.Imax:
            i = -self.Imax
        self.calc_ss(i=i, Te=self.Th, init=init, qa=qa)
        yk, xk = euler(self.A, self.B, self.C, self.D, dt, self.xk, np.array([qa, 1]))
        self.Ta = yk[0][0]
        self.xk = xk
        self.V = self.alpha_m * (self.Th-self.Ta) + i*self.Rm
