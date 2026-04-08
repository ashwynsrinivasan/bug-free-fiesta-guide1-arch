''' SPPD DFB Laser Model'''

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, attribute-defined-outside-init, protected-access, too-many-arguments, too-many-instance-attributes, too-few-public-methods
from typing import Union, Callable, Any
import pathlib
import warnings
import yaml
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

PATH = pathlib.Path(__file__).parent.absolute()

class SPPDDFB:
    """ SPPDDFB class"""
    def __init__(
        self,
        current: Union[float, Callable] = 0.125,
        dfb_backside_temperature: Union[float, Callable] = 273+25,
        wavelength_fabrication: Union[float, Callable] = 1.31e-6,
        li_slope_factor: Union[float, Callable] = 1,
        name: str = 'SPPDDFB'
    ):
        self.name = name
        self._current = self._genvar(current)
        self._nominal_current = 0.125
        self._delta_current = self._current - self._nominal_current
        self._dfb_backside_temperature = self._genvar(dfb_backside_temperature)
        self._psi_matrix = pd.read_csv(PATH.__str__()+'/sppd_pic_psi_matrix.csv')
        self._psi_array = self._psi_matrix.to_numpy()
        
        self._wavelength_fabrication = self._genvar(wavelength_fabrication)
        self._li_slope_factor = self._genvar(li_slope_factor)
        with open(PATH.__str__()+"/sppd_data.yaml", "r", encoding="utf-8") as fh:
            self._sppd_data = yaml.load(fh, Loader=yaml.SafeLoader)
        self._sppd_dfb_data = self._sppd_data['DFB']
                
        self._temperature_initialize()
        
    @staticmethod
    def _genvar(gen: Any) -> Any:
        return (gen() if callable(gen) else gen)
    
    def _vdiode(self):
        """Calculate the diode voltage."""
        self._voltage = np.polyval(np.array(self._sppd_dfb_data['IV']), self._current)
        return self._voltage
    
    def _output_power_update(self):
        """Calculate the output power."""
        li_fit = np.array(self._sppd_dfb_data['LI'])
        li_fit[0] = li_fit[0] / self._li_slope_factor
        self._output_power = np.polyval(li_fit, self._current)
        return self._output_power
    
    def _wpe_func(self):
        self._dfb_elec_power = self._vdiode() * self._current
        self._output_power_update()
        self._wpe = self._output_power/self._dfb_elec_power
        return self._wpe
    
    def _temperature_initialize(self):
        _ = self._wpe_func()
        self._total_power_loss = self._dfb_elec_power - self._output_power
        self._delta_temperature = 0.0
        self._nominal_junction_temperature = self._dfb_backside_temperature + self._psi_array[0,0] * self._total_power_loss
        self._temperature_update()
    
    def _temperature_update(self):
        self._temperature = self._nominal_junction_temperature + self._delta_temperature
    
    def _current_update(self):
        self._delta_current = self._current - self._nominal_current
    
    def _wavelength_update(self):
        self._temperature_update()
        self._current_update()
        self._wavelength = self._wavelength_fabrication + 0.1e-9 * self._delta_temperature #+ 0.1e-9/17 * self._delta_current*1e-3 

class SPPDMPD:
    """ SPPDMPD class"""
    def __init__(
        self,
        dark_current: Union[float, Callable] = 1e-9,
        responsivity: Union[float, Callable] = 0.9,
        name: str = 'SPPDCLMPD'
    ):
        self.name = name
        self._dark_current = self._genvar(dark_current)
        self._responsivity = self._genvar(responsivity)

    @staticmethod
    def _genvar(gen: Any) -> Any:
        return (gen() if callable(gen) else gen)

    def _output_current_update(self, input_power):
        self._output_current = self._responsivity * input_power + self._dark_current
        return self._output_current

class SPPDCLMPIC:
    """ SPPDCLMPIC class"""
    def __init__(
        self,
        current = 0.144,
        dfb_backside_temperature = 273 + 25,
        wavelength_fabrication = np.array([1301.47,1302.60,1303.73,1304.87,1306.01,1307.14,1308.28,1309.43,1310.57,1311.72,1312.87,1314.02,1315.17,1316.33,1317.48,1318.64])*1e-9,
        wpe_slope_factor = np.ones(16),
        mpd_tap = 1e-2,
        name = 'SPPDCLMPIC'
        ):
        
        self.name = name
        self._current = current
        self._wavelength_fabrication = wavelength_fabrication
        self._dfb_backside_temperature = dfb_backside_temperature
        self._mpd_tap = mpd_tap
        self._wpe_slope_factor = wpe_slope_factor
        self._laser_array = {}
        self._lambda_array = np.zeros(len(self._wavelength_fabrication))
        self._mpd_array = {}
        self._mpd_currrent_array = np.zeros(len(self._wavelength_fabrication))
        self._psi_matrix = pd.read_csv(PATH.__str__()+'/sppd_pic_psi_matrix.csv')
        self._psi_array = self._psi_matrix.to_numpy()
        self.laser_initialize()
        
    
    def laser_initialize(self):
        ''' 
        Initialize the laser array
        '''
        for idd_laser, wavelength_value in enumerate(self._wavelength_fabrication):
            self._laser_array[idd_laser] = SPPDDFB(
              current = self._current,
              wavelength_fabrication = wavelength_value,
              li_slope_factor = self._wpe_slope_factor[idd_laser],
              dfb_backside_temperature= self._dfb_backside_temperature,
              name = 'SPPDDFB_'+str(idd_laser)
            )
            
            self._laser_array[idd_laser]._output_power_update()
            self._laser_array[idd_laser]._wavelength_update()
            self._lambda_array[idd_laser] = self._laser_array[idd_laser]._wavelength
            self._mpd_array[idd_laser] = SPPDMPD(
              name = 'SPPDMPD_'+str(idd_laser)
            )
            self._mpd_array[idd_laser]._output_current_update(self._laser_array[idd_laser]._output_power * self._mpd_tap)
            self._mpd_currrent_array[idd_laser] = self._mpd_array[idd_laser]._output_current
        
        self._total_dfb_elec_power, self._total_output_power = np.zeros(len(self._laser_array)), np.zeros(len(self._laser_array))
        for idd_laser, __ in enumerate(self._laser_array):
            self._laser_array[idd_laser]._current = self._current
            self._laser_array[idd_laser]._wpe_func()
            self._total_dfb_elec_power[idd_laser] = self._laser_array[idd_laser]._dfb_elec_power
            self._total_output_power[idd_laser] = self._laser_array[idd_laser]._output_power
            
        self._total_power_loss = self._total_dfb_elec_power - self._total_output_power
        self._delta_junction_temperature = [0.0] * len(self._laser_array)
        
        self._nominal_junction_temperature = self._dfb_backside_temperature + self._psi_array @ self._total_power_loss
        for idd_laser, wavelength_value in enumerate(self._wavelength_fabrication):
            self._laser_array[idd_laser]._delta_temperature = self._delta_junction_temperature[idd_laser]
            self._laser_array[idd_laser]._nominal_junction_temperature = self._nominal_junction_temperature[idd_laser]
        

    def update(self, tc, i_array):
        """ Updated the laser array state"""
        self._case_temperature = tc
        self._i_array = i_array
        self._total_dfb_elec_power, self._total_output_power = np.zeros(len(self._laser_array)), np.zeros(len(self._laser_array))       
        for idd_laser, __ in enumerate(self._laser_array):
            self._laser_array[idd_laser]._current = self._i_array[idd_laser]
            self._laser_array[idd_laser]._wpe_func()
            self._total_dfb_elec_power[idd_laser] = self._laser_array[idd_laser]._dfb_elec_power
            self._total_output_power[idd_laser] = self._laser_array[idd_laser]._output_power  
        self._total_power_loss = self._total_dfb_elec_power - self._total_output_power
        self._delta_junction_temperature = tc * np.ones(len(self._laser_array)) + self._psi_array @ self._total_power_loss - self._nominal_junction_temperature
        self._pout_array = np.zeros(len(self._laser_array))
        for idd_laser, __ in enumerate(self._laser_array):
            self._laser_array[idd_laser]._delta_temperature = self._delta_junction_temperature[idd_laser]
            self._laser_array[idd_laser]._temperature_update()
            self._laser_array[idd_laser]._wavelength_update()
            self._laser_array[idd_laser]._output_power_update()

            self._lambda_array[idd_laser] = self._laser_array[idd_laser]._wavelength
            self._pout_array[idd_laser] = self._laser_array[idd_laser]._output_power
            self._mpd_array[idd_laser]._output_current_update(self._laser_array[idd_laser]._output_power * self._mpd_tap)
            self._mpd_currrent_array[idd_laser] = self._mpd_array[idd_laser]._output_current
