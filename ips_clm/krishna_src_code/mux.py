import pandas as pd
import numpy as np

class mux:

    # lambda array, insertion loss, BW, SB attenuation
    lambda_array = None
    il_db: float = 3
    bw_lambda = 0.1e-9
    sb_atten_db = 50
    tf_array = None

    def __init__(self, lambda_array, il_db, bw_lambda, sb_atten):
        self.lambda_array = lambda_array
        self.il_db = il_db
        self.il_mag = 10**(-self.il_db/10)
        self.bw_lambda = bw_lambda
        self.sb_atten = sb_atten
        self.sb_atten_mag = 10**(-self.sb_atten/10)

        self.lambda_sweep = np.linspace(np.min(lambda_array)-5e-9, np.max(lambda_array)+5e-9, 1000)
        self.tf_array = np.zeros((len(lambda_array), len(self.lambda_sweep)))
        for i in range(len(lambda_array)):
            # gaussian filter
            self.tf_array[i] = self.il_mag*np.exp(-((self.lambda_sweep-lambda_array[i])/bw_lambda)**2/2)
            # sideband rejection
            idx = self.tf_array[i] < self.sb_atten_mag
            self.tf_array[i][idx] = self.sb_atten_mag

    def update(self, lambda_array, pin_array):
        # lambda_array: wavelength array
        # pin_array: input power array
        # returns: output power array
        pout_array = np.zeros((len(lambda_array), 1))
        for i in range(len(lambda_array)):
            idx = np.argmin(np.abs(lambda_array[i]-self.lambda_sweep))
            pout_array[i] = pin_array[i]*self.tf_array[i][idx]
        return pout_array[:,0]





