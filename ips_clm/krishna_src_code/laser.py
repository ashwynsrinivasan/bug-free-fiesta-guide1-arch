import numpy as np
class laser:
    # Inputs - junction temperature and drive current
    # Outputs - optical power, wavelength, and electrical power
    tj = 273 + 25 # Junction temperature, K
    tj0 = 273 + 25 # Junction temperature, K
    i = 0.2 # Drive current, A
    lambda0 = 1310e-9 # Center wavelength, m
    freq_spacing = 400e9 # Frequency spacing, Hz
    dldt = 0.1e-9 # Wavelength change per degree C, m/K
    theta_jc = 5 # Junction to case thermal resistance, K/W
    eff = 0.15 # Electrical to optical efficiency
    v = 2.0 # Voltage drop across laser, V
    po = 0 # Output optical power
    pe = 0 # Electrical power
    ploss = 0 # Power loss
    N_array = 8 # Number of lasers in array
    delta_T = 5 # Temperature difference between lasers in array, K
    r_heater = None

    def __init__(self, tj=25, i=200e-3, N_array=8, delta_T=5, r_heater=100, freq_spacing=400e9):
        self.tj0 = tj
        self.i = i
        self.N_array = N_array
        self.delta_T = delta_T
        self.freq_spacing = freq_spacing
        self.lambda_sp = self.lambda0*3e8/(3e8-self.freq_spacing*self.lambda0)-self.lambda0
        self.lambda_array0 = np.zeros((self.N_array, 1))
        for i in range(self.N_array):
            self.lambda_array0[i, 0] = self.lambda0 + (i-self.N_array/2)*self.lambda_sp

        theta_array = 4*delta_T/(self.i*self.N_array*(self.N_array-1))
        y_array = np.zeros((self.N_array, self.N_array))
        i_array = np.ones((self.N_array, 1))*self.i
        for i in range(self.N_array-1):
            y_array[i, i] = y_array[i, i]+1/theta_array + 1/(self.N_array*self.theta_jc)
            y_array[i, i+1] = y_array[i, i+1]-1/theta_array
            y_array[i+1, i] = y_array[i+1, i]-1/theta_array
            y_array[i+1, i+1] = y_array[i+1, i+1]+1/theta_array
        y_array[0, 0] += 1 / (self.N_array * self.theta_jc * 4)
        y_array[self.N_array-1, self.N_array-1] += 1/(self.N_array*self.theta_jc*0.8)
        self.y_array = y_array
        self.i_array = i_array
        self.r_heater = r_heater

    def update(self, tc, i_array, i_heater_array):
        self.i_array = i_array
        self.pe =self.i_array * self.v
        self.po = self.pe * self.eff
        self.ploss = self.pe - self.po + i_heater_array**2*self.r_heater

        self.tj = tc + np.linalg.solve(self.y_array, self.ploss)
        self.lambda_array = self.lambda_array0[:,0] + self.dldt*(self.tj-self.tj0)

        return self.po, self.lambda_array, self.pe, self.ploss