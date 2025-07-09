''' Enablence MUX model'''

# pylint: disable=trailing-whitespace, invalid-name, too-many-locals, locally-disabled, dangerous-default-value, multiple-statements, fixme, line-too-long, attribute-defined-outside-init, protected-access, too-many-arguments, too-many-instance-attributes, too-few-public-methods

import warnings
import pathlib
from copy import deepcopy
import numpy as np
import pandas as pd

PATH = pathlib.Path(__file__).parent.absolute()

warnings.filterwarnings("ignore")

um = 1e-6
class EBMUX:
    ''' EB MUX model'''
    def __init__(
          self, 
          target_grid_array, 
          bw_lambda=0.3e-9, 
          il_dB = 3, 
          sb_atten_dB = 50, 
          mpd_dark_current = 1e-8, 
          mpd_responsivity=0.9, 
          mpd_tap = 5e-2, 
          ebmux_backside_temperature = 45.5 + 273.15,
          name = 'EBMUX'):
        
        self.bw_lambda = bw_lambda
        self.sb_atten_dB = sb_atten_dB
        self.sb_atten_mag = 10**(-self.sb_atten_dB/10)
        self.lambda_sweep = np.linspace(np.min(target_grid_array)-5e-9, np.max(target_grid_array)+5e-9, 10000)
        self.mpd_dark_current = mpd_dark_current
        self.mpd_tap = mpd_tap
        self.mpd_tap_excel = 5e-2
        self.mpd_responsivity = mpd_responsivity
        self.target_grid_array = target_grid_array
        self.ebmux_backside_temperature = ebmux_backside_temperature
        self.temperature = deepcopy(self.ebmux_backside_temperature)
        self.delta_temperature = self.temperature - self.ebmux_backside_temperature
        self.wavelength_temperature_sensitivity = 0.01e-9
        
        ## TODO: Delete the old lines
        self.il_mag_dB = il_dB
        self.il_mag = 10**(-self.il_mag_dB/10)
        self.name = name
        tf_array_list = np.zeros((len(target_grid_array), len(self.lambda_sweep)))
        for i, target_grid_array_value in enumerate(target_grid_array):
            tf_array_list[i,:] = self.il_mag*np.exp(-((self.lambda_sweep-target_grid_array_value)/bw_lambda)**2/4.343)
            idx = tf_array_list[i,:] < self.sb_atten_mag
            tf_array_list[i][idx] = self.sb_atten_mag
        self.enablence_mux_tf_array_list = tf_array_list**0.5

        self.ebmux_data_extract()
        
    def temperature_update(self):
        '''
        Update the temperature of the Enablence Mux.

        Args:
            temperature (float): Temperature value.

        Returns:
            None
        '''
        
        self.delta_temperature = self.temperature - self.ebmux_backside_temperature
    
    def wavelength_update(self):
        '''
        Update the wavelength of the Enablence Mux.

        Returns:
            None
        '''
        
        self.center_grid = self.target_grid_array + self.delta_temperature * self.wavelength_temperature_sensitivity

    def ebmux_data_extract(self):
        '''
        Extracts data from the EBMUX_LD_and_mPD.xlsx file and 
        calculates the transfer function of the EBMUX.

        Returns:
            None
        '''
        self.temperature_update()
        self.wavelength_update()
        ebmux_data = pd.read_excel(r"../src/EBMUX_LD_and_mPD.xlsx")
        max_lc = np.max(ebmux_data['LD ch 5'])
        max_mpd = np.max(ebmux_data['mPDch 5'])
        arg_max_lc = np.argmax(ebmux_data['LD ch 5'])
        # arg_max_mpd = np.argmax(ebmux_data['mPDch 5'])
        
        arg_min_1_lc = np.argmin(np.abs(ebmux_data['LD ch 5'].values[0:arg_max_lc]+1.0-max_lc))
        arg_min_1_mpd = np.argmin(np.abs(ebmux_data['mPDch 5'].values[0:arg_max_lc]+1.0-max_mpd))
        arg_min_2_lc = np.argmin(np.abs(ebmux_data['LD ch 5'].values[arg_max_lc::]+1.0-max_lc))+arg_max_lc
        arg_min_2_mpd = np.argmin(np.abs(ebmux_data['mPDch 5'].values[arg_max_lc::]+1.0-max_mpd))+arg_max_lc
        
        self.bw_ebmux = (ebmux_data['wavelength'].values[arg_min_2_lc] - ebmux_data['wavelength'].values[arg_min_1_lc])*0.5e-9
        self.bw_ebmux_mpd = (ebmux_data['wavelength'].values[arg_min_2_mpd] - ebmux_data['wavelength'].values[arg_min_1_mpd])*0.5e-9
        
        self.il_ebmux = 10**(max_lc/10)
        self.il_ebmux_mpd = 10**((max_mpd)/10)
        tf_array_list = np.zeros((len(self.center_grid), len(self.lambda_sweep)))
        eb_mpd_tf_array_list = np.zeros((len(self.center_grid), len(self.lambda_sweep)))
        # eb_mpd_array_list = np.zeros((len(self.target_grid_array), len(self.lambda_sweep)))
        for i, center_grid_array_value in enumerate(self.center_grid):
            tf_array_list[i,:] = self.il_ebmux*np.exp(-((self.lambda_sweep-center_grid_array_value)/self.bw_ebmux)**2/4.343)
            idx = tf_array_list[i,:] < self.sb_atten_mag
            tf_array_list[i][idx] = self.sb_atten_mag

            eb_mpd_tf_array_list[i,:] = self.il_ebmux_mpd*np.exp(-((self.lambda_sweep-center_grid_array_value)/self.bw_ebmux_mpd)**2/4.343) * self.mpd_tap/self.mpd_tap_excel
            idx = eb_mpd_tf_array_list[i,:] < self.sb_atten_mag
            eb_mpd_tf_array_list[i][idx] = self.sb_atten_mag
        
        self.ebmux_tf_array_list = deepcopy(tf_array_list)
        self.ebmux_mpd_tf_array_list = deepcopy(eb_mpd_tf_array_list)
    
    def update(self, channel_array, pin_array):
            '''
            Update the output power and current arrays of the Enablence Mux.

            Args:
                channel_array (numpy.ndarray): Array of channel values.
                pin_array (numpy.ndarray): Array of input power values.

            Returns:
                None
            '''
            
            self.ebmux_data_extract()
            self.enablence_mux_pout_array = np.zeros(len(self.ebmux_tf_array_list))
            self.enablence_mux_mpd_pout_array = np.zeros(len(self.ebmux_mpd_tf_array_list))
            
            for iddx_pout, __ in enumerate(self.enablence_mux_pout_array):
                for idd_ch, channel_array_value in enumerate(channel_array):
                    iddx = np.argmin(np.abs(channel_array_value-self.lambda_sweep))
                    self.enablence_mux_pout_array[idd_ch] += pin_array[idd_ch]*self.ebmux_tf_array_list[iddx_pout, iddx]
                    self.enablence_mux_mpd_pout_array[iddx_pout] += pin_array[idd_ch]*self.ebmux_mpd_tf_array_list[iddx_pout, iddx] 

            self.pout_array = deepcopy(self.enablence_mux_pout_array)
            self.pout_mpd_array = deepcopy(self.enablence_mux_mpd_pout_array)
            self.mpd_current_array = self.pout_mpd_array*self.mpd_responsivity + self.mpd_dark_current
if __name__=='__main__':
    '''
    questions: 
        self.temperature - self.ebmux_backside_temperature
        
    tf in power or feild? (10 or 20log?)
        
    '''
    #%%    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_palette("bright")
    
    #%%
    wl_chCenter_m=np.array([1310e-9,1320e-9])
    emux=EBMUX(wl_chCenter_m)
    
    print('enablence mux 1dB half BW_nm:', emux.bw_ebmux*1e9)
    print('enablence mpd 1dB half BW_nm:', emux.bw_ebmux_mpd*1e9)
    #%%
    ''' plot channel passband '''
    df=pd.DataFrame(columns=['wl_nm','mpd_ch0','mpd_ch1','tx_ch0','tx_ch1'])
    df.wl_nm=emux.lambda_sweep*1e9
    df.mpd_ch0=10*np.log10(emux.ebmux_mpd_tf_array_list[0,:])
    df.mpd_ch1=10*np.log10(emux.ebmux_mpd_tf_array_list[1,:])
    df.tx_ch0=10*np.log10(emux.ebmux_tf_array_list[0,:])
    df.tx_ch1=10*np.log10(emux.ebmux_tf_array_list[1,:])
    
    fig, ax= plt.subplots()
    df.plot('wl_nm',['mpd_ch0','mpd_ch1','tx_ch0','tx_ch1'],ax=ax)
    plt.show()
    #%%
    ''' ramp temps '''
    temps_c=np.linspace(0,50,6)
    dfs=[]
    for temp in temps_c:
        emux.temperature=temp+273.15
        emux.temperature_update()
        emux.wavelength_update()
        emux.ebmux_data_extract()

        df=pd.DataFrame(columns=['wl_nm','mpd_ch0','tx_ch0'])
        df.wl_nm=emux.lambda_sweep*1e9
        df.mpd_ch0=10*np.log10(emux.ebmux_mpd_tf_array_list[0,:])
        df.tx_ch0=10*np.log10(emux.ebmux_tf_array_list[0,:])
        df['Temp_c']=temp
        dfs.append(df)
        
    dfT=pd.concat(dfs)
    
    fig, ax= plt.subplots()
    sns.lineplot(data=dfT, x='wl_nm',y='mpd_ch0',hue='Temp_c', palette="bright",ax=ax)
    ax.set_ylim((-25,-15))
    ax.set_xlim((wl_chCenter_m[0]*1e9-1,wl_chCenter_m[0]*1e9+1))
    plt.show()

    fig, ax= plt.subplots()
    sns.lineplot(data=dfT, x='wl_nm',y='tx_ch0',hue='Temp_c', palette="bright",ax=ax)
    ax.set_ylim((-10,0))
    ax.set_xlim((wl_chCenter_m[0]*1e9-2,wl_chCenter_m[0]*1e9+1))
    plt.show()
    #%%
    ''' check update function '''
    
    emux.temperature=40+273.15
    emux.temperature_update()
    emux.wavelength_update()
    emux.ebmux_data_extract()
    
    pin_mw=np.array([1,0.5])
    
    dfs=[]
    wl_errors_nm=np.linspace(-1,1,101)
    cols=['mpdCur_ch0','mpdCur_ch1','Pout_ch0','Pout_ch1','wlOffset_nm']
    rcd=np.zeros((len(wl_errors_nm),len(cols)))
    
    for i, wlerr in enumerate(wl_errors_nm):
        emux.update(wl_chCenter_m+wlerr*1e-9,pin_mw) 
        
        rcd[i,0]=emux.mpd_current_array[0]
        rcd[i,1]=emux.mpd_current_array[1]
        rcd[i,2]=emux.pout_array[0]
        rcd[i,3]=emux.pout_array[1]
        rcd[i,4]=wlerr
    df=pd.DataFrame(columns=cols)
    for j, col in enumerate(cols):
        df[col]=rcd[:,j]
    
    fig,ax=plt.subplots()
    axt=ax.twinx()
    
    df.plot('wlOffset_nm',['Pout_ch0','Pout_ch1'], ax=ax) 
    df.plot('wlOffset_nm',['mpdCur_ch0', 'mpdCur_ch1'], linestyle=':',ax=axt) 
    ax.set_ylabel('Pout')
    ax.legend(loc='upper left')
    axt.set_ylabel('MPD Current')
    axt.legend(loc='upper right')
    ax.set_title('Pin: [%.2f, %.2f]'%(pin_mw[0],pin_mw[1]))
    plt.show()