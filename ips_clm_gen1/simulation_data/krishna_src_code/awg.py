import pandas as pd
import numpy as np


class awg:
    lambda_tuning_array = None
    csv_file = None
    lane_list = None
    il_dict = None
    wavelength_array = None
    lambda_sweep = None
    channel_array = None
    power_array = None

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        lane_list = df['Lane'].unique()
        self.lane_list = lane_list
        self.il_dict = {}
        self.wavelength_array = df['Wavelength [nm]'][df['Lane'] == 0].to_numpy()
        self.lambda_sweep = self.wavelength_array
        for lane in lane_list:
            self.il_dict[lane] = df['Insertion Loss [dB]'][df['Lane'] == lane].to_numpy()

    def tune(self, lambda_tuning_array):
        self.lambda_tuning_array = lambda_tuning_array
        # Find the channel to use for each wavelength
        channel_array = np.zeros((len(lambda_tuning_array), 1))
        power_array = np.zeros((len(lambda_tuning_array), 1))
        idx_array = np.zeros((len(lambda_tuning_array), 1))
        for i in range(len(lambda_tuning_array)):
            idx_array[i] = np.argmin(np.abs(lambda_tuning_array[i] - self.wavelength_array))
            # Sweep through all lanes and find the one with the minimum insertion loss at the wavelength
            il_array = np.zeros((len(self.lane_list), 1))
            for j in range(len(self.lane_list)):
                il_array[j] = self.il_dict[self.lane_list[j]][int(idx_array[i])]
            channel_array[i] = self.lane_list[np.argmax(il_array)]
            power_array[i] = 10 ** (np.max(il_array) / 10)
        self.channel_array = [channel_array[i][0] for i in range(len(channel_array))]
        self.power_array = np.array([power_array[i][0] for i in range(len(power_array))])

    def update(self, lambda_array, pin_array):
        # lambda_array: wavelength array
        # pin_array: input power array
        # returns: output power array
        pout_array = np.zeros((len(self.lambda_tuning_array), 1))
        # find the insertion loss for lambda on the channel to use for each wavelength
        for i in range(len(self.lambda_tuning_array)):
            idx = np.argmin(np.abs(lambda_array[i] - self.wavelength_array))
            pout_array[i] = pin_array[i] * 10 ** (self.il_dict[self.channel_array[i]][int(idx)] / 10)
        return pout_array[:, 0]