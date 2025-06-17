import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from scipy.stats import norm

class Laser_power_at_Txbank():
    def __init__(self):
        pass

    def muxloss(self):
        a = -2
        mean = -1.5
        scale = 0.15
        return skewnorm.rvs(a, scale = scale, loc = mean)
    
    def laserpower(self):
        mean = 32.4; #mW
        scale = 2; #mW - set such that 4-sigma is the ±8mW min or max value from SPPD
        return 10*np.log10(norm.rvs(loc=mean, scale=scale))

    def TX_in(self):
        p_in = self.laserpower(); #dBm
        laser_to_mux = -1.6; #dB
        mux = self.muxloss(); #dB
        splices = -0.12
        mux_to_fiber = -0.5; #dB
        fiber_to_jag = -2.05; #dB
        split = -3; #dB, split 2x on Jag
        routing = -0.17; #dB

        return p_in + laser_to_mux + mux + splices + mux_to_fiber + fiber_to_jag + split + routing
    
    def sample_laser_power(self,num_samples):
        #Outputs CLM output power into FAU in dBm
        p_in = self.laserpower(); #dBm
        laser_to_mux = -1.6; #dB
        mux = self.muxloss(); #dB
        splices = -0.12
        mux_to_fiber = -0.5; #dB

        return p_in + laser_to_mux + mux + splices + mux_to_fiber


    def sample_from_distribution(self,num_samples):
        return np.array([self.TX_in() for i in range(num_samples)])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        plt.style.use("asludds")
    except:
        pass
    num_samples_per_dist = int(1e6)
    num_bins = 100

    laser = Laser_power_at_Txbank()
    samples = laser.sample_from_distribution(num_samples_per_dist)

    plt.figure()
    plt.hist(samples,bins=num_bins)
    plt.title("Laser output power at Txbank + \n" + r"$\mu = $" + f"{round(np.mean(samples),3)} dBm, " + r"$\sigma = $" + f"{round(np.std(samples),3)} dB")
    plt.xlabel("Total power Per wavelength at TxBank (dBm)")
    plt.ylabel("Count")
    plt.show()

    sigma_4_percentile = 99.995
    print(np.percentile(samples,sigma_4_percentile))
    print(np.percentile(samples,1 - sigma_4_percentile))