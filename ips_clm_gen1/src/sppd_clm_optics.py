''' SPPDCLMOptics class'''

# pylint: disable=trailing-whitespace, invalid-name, too-many-locals, locally-disabled, multiple-statements, fixme, line-too-long, attribute-defined-outside-init, protected-access, too-many-arguments, too-many-instance-attributes, too-few-public-methods
# flake8: noqa
import pathlib
import warnings
from copy import deepcopy
import numpy as np
from .sppd_dfb_laser import SPPDCLMPIC
from .enablence_mux import EBMUX

warnings.filterwarnings("ignore")

PATH = pathlib.Path(__file__).parent.absolute()

class SPPDCLMOPTICS:
    ''' SPPDCLMOptics class'''
    def __init__(
        self,
        target_grid_array = np.array([1301.47,1302.60,1303.73, 1304.87,1306.01,1307.14,1308.28, 1309.43,1310.57,1311.72,1312.87, 1314.02,1315.17,1316.33,1317.48, 1318.64])*1e-9, 
        bw_lambda=0.1e-9, 
        il_dB = 3, 
        sb_atten_dB = 50, 
        mpd_dark_current = 1e-8, 
        mpd_responsivity=0.75,
        current = 0.125,
        dfb_backside_temperature = 273+45, 
        wavelength_fabrication = np.array([1301.47,1302.60,1303.73,1304.87,1306.01,1307.14,1308.28,1309.43,1310.57,1311.72,1312.87,1314.02,1315.17,1316.33,1317.48,1318.64])*1e-9,
        wpe_slope_factor = np.ones(16),
        r_heater = 100,
        sppd_eb_atten_dB = 2.3,
        ebmux_tec_bool = True,
        name: str = 'SPPDCLMOPTICS_LAMBDA_A'  
    ):
        self.name = name
        self._target_grid_array = target_grid_array
        self._bw_lambda = bw_lambda
        self._il_dB = il_dB
        self._sb_atten_dB = sb_atten_dB
        self._mpd_dark_current = mpd_dark_current
        self._mpd_responsivity = mpd_responsivity
        self._current = current
        self._sppd_eb_atten_dB = sppd_eb_atten_dB
        self._dfb_backside_temperature = dfb_backside_temperature
        self._wavelength_fabrication = wavelength_fabrication 
        self._r_heater = r_heater
        self._wpe_slope_factor = wpe_slope_factor
        self._ebmux_tec_bool = ebmux_tec_bool
        
        self._sppd_clm_pic = SPPDCLMPIC(
          current = self._current,
          wavelength_fabrication = self._wavelength_fabrication,
          wpe_slope_factor = self._wpe_slope_factor,
          dfb_backside_temperature=self._dfb_backside_temperature,
          name = 'SPPDCLMPIC_'+self.name
        )

        self._ebmux = EBMUX(
          target_grid_array=self._target_grid_array, 
          bw_lambda = self._bw_lambda, 
          il_dB = self._il_dB, 
          sb_atten_dB = self._sb_atten_dB,
          mpd_dark_current = self._mpd_dark_current, 
          mpd_responsivity = self._mpd_responsivity,
          name = 'EBMUX_'+self.name
          )
      
    def update(self, tc, i_array):
        ''' Update the SPPDCLMOPTICS class with the new temperature and current values'''
        if self._ebmux_tec_bool:
            self._ebmux.temperature = tc
            
        self._sppd_clm_pic.update(tc, i_array)
        self._ebmux.update(self._sppd_clm_pic._lambda_array, self._sppd_clm_pic._pout_array* 10**(-self._sppd_eb_atten_dB/10))

        ## TODO: Delete the old lines
        self._lambda_array = self._sppd_clm_pic._lambda_array
        self._pout_array = self._ebmux.pout_array
        self._pout_mpd_array = self._ebmux.pout_mpd_array
        self._mpd_current_array = self._ebmux.mpd_current_array
        self._pout_array = self._ebmux.pout_array
        
        ## Refactored code
        self._lambda_array = self._sppd_clm_pic._lambda_array
        self._sppd_clm_pic_pout_array  = self._sppd_clm_pic._pout_array
        self._sppd_clm_pic_mpd_current_array = self._sppd_clm_pic._mpd_currrent_array
        self._enablence_mux_mpd_current_array = self._ebmux.mpd_current_array
        self._enablence_mux_pout_array = self._ebmux.pout_array


class SPPDCLMOPTICS_LAMBDA_AB:
    ''' SPPDCLMOptics class for A and B channels'''
    def __init__(
        self,
        target_grid_array = np.array([1301.47,1302.60,1303.73, 1304.87,1306.01,1307.14,1308.28, 1309.43,1310.57,1311.72,1312.87, 1314.02,1315.17,1316.33,1317.48, 1318.64])*1e-9, 
        bw_lambda=0.1e-9, 
        il_dB = 3, 
        sb_atten_dB = 50, 
        mpd_dark_current = 1e-8, 
        ebmux_mpd_tap = 5e-2,
        mpd_responsivity=0.75,
        current = 0.125,
        dfb_backside_temperature = 273+45, 
        ebmux_backside_temperature = 273+45,
        wavelength_fabrication = np.array([1301.47,1302.60,1303.73,1304.87,1306.01,1307.14,1308.28,1309.43,1310.57,1311.72,1312.87,1314.02,1315.17,1316.33,1317.48,1318.64])*1e-9,
        wpe_slope_factor = np.ones(16),
        sppd_eb_atten_dB = 2.3,
        no_mux = 2,
        ebmux_tec_bool = True,
        name: str = 'SPPDCLMOPTICS_LAMBDA_AB'  
    ):
        self.name = name
        _target_grid_array_A, _target_grid_array_B = self.split_AB(target_grid_array)
        self._target_grid_array_A = _target_grid_array_A
        self._target_grid_array_B = _target_grid_array_B
        self._target_grid_array = target_grid_array
        self._wavelength_fabrication = wavelength_fabrication
        self._wpe_slope_factor = wpe_slope_factor
        self._bw_lambda = bw_lambda
        self._il_dB = il_dB
        self._sb_atten_dB = sb_atten_dB
        self._sppd_eb_atten_dB = sppd_eb_atten_dB
        self._mpd_dark_current = mpd_dark_current
        self._mpd_responsivity = mpd_responsivity
        self._ebmux_mpd_tap = ebmux_mpd_tap
        self._current = current
        self._dfb_backside_temperature = dfb_backside_temperature
        self._temperature = deepcopy(self._dfb_backside_temperature)
        self._ebmux_backside_temperature = ebmux_backside_temperature
        self._no_mux = no_mux
        self._ebmux_tec_bool = ebmux_tec_bool

        self._sppd_clm_pic = SPPDCLMPIC(
          current = self._current,
          dfb_backside_temperature=self._dfb_backside_temperature,
          wavelength_fabrication = self._wavelength_fabrication,
          wpe_slope_factor = self._wpe_slope_factor,
          name = 'SPPDCLMPIC_'+self.name
        )

        self._ebmux_1 = EBMUX(
          target_grid_array=self._target_grid_array_A, 
          bw_lambda = self._bw_lambda, 
          il_dB = self._il_dB, 
          sb_atten_dB = self._sb_atten_dB,
          mpd_dark_current = self._mpd_dark_current, 
          mpd_responsivity = self._mpd_responsivity,
          mpd_tap = self._ebmux_mpd_tap,
          ebmux_backside_temperature=self._ebmux_backside_temperature,
          name = 'EBMUX_1_'+self.name
          )

        self._ebmux_2 = EBMUX(
          target_grid_array=self._target_grid_array_B, 
          bw_lambda = self._bw_lambda, 
          il_dB = self._il_dB, 
          sb_atten_dB = self._sb_atten_dB,
          mpd_dark_current = self._mpd_dark_current, 
          mpd_responsivity = self._mpd_responsivity,
          mpd_tap = self._ebmux_mpd_tap,
          ebmux_backside_temperature=self._ebmux_backside_temperature,
          name = 'EBMUX_2_'+self.name
        )

        self._ebmux = EBMUX(
          target_grid_array=self._target_grid_array, 
          bw_lambda = self._bw_lambda, 
          il_dB = self._il_dB, 
          sb_atten_dB = self._sb_atten_dB,
          mpd_dark_current = self._mpd_dark_current, 
          mpd_responsivity = self._mpd_responsivity,
          mpd_tap = self._ebmux_mpd_tap,
          ebmux_backside_temperature=self._ebmux_backside_temperature,
          name = 'EBMUX_'+self.name
        )
    
    def split_AB(self, array):
        ''' Split the array into A and B channels'''
        _array_A = []
        _array_B = []
        for idx_array, array_val in enumerate(array):
            if idx_array % 2 == 0:
                _array_A += [array_val]
            else:
                _array_B += [array_val]
        return np.array(_array_A), np.array(_array_B)
    
    def combine_AB(self, array_A, array_B):
        ''' Combine the A and B channels into a single array'''
        _array = []
        for idx_array, array_a_val in enumerate(array_A):
            _array += [array_a_val]
            _array += [array_B[idx_array]]
        return np.array(_array)
      
    def update(self, tc, i_array):
        ''' Update the SPPDCLMOPTICS class with the new temperature and current values'''
        if self._ebmux_tec_bool:
            self._ebmux.temperature = tc
        
        self._sppd_clm_pic.update(tc, i_array)
        if self._no_mux == 2:
            _wavelength_fabrication_A, _wavelength_fabrication_B = self.split_AB(self._sppd_clm_pic._lambda_array)
            _pout_array_A, _pout_array_B = self.split_AB(self._sppd_clm_pic._pout_array)
            self._ebmux_1.update(_wavelength_fabrication_A, _pout_array_A * 10**(-self._sppd_eb_atten_dB/10))
            self._ebmux_2.update(_wavelength_fabrication_B, _pout_array_B * 10**(-self._sppd_eb_atten_dB/10))
        elif self._no_mux == 1:
            self._ebmux.update(self._sppd_clm_pic._lambda_array, self._sppd_clm_pic._pout_array * 10**(-self._sppd_eb_atten_dB/10))

        ## TODO: Delete the old lines
        self._lambda_array = self._sppd_clm_pic._lambda_array
        if self._no_mux == 2:
            self._pout_array = self.combine_AB(self._ebmux_1.pout_array, self._ebmux_2.pout_array)
            self._pout_mpd_array = self.combine_AB(self._ebmux_1.pout_mpd_array, self._ebmux_2.pout_mpd_array)
            self._mpd_current_array = self.combine_AB(self._ebmux_1.mpd_current_array, self._ebmux_2.mpd_current_array)
        elif self._no_mux == 1:
            self._pout_array = self._ebmux.pout_array
            self._pout_mpd_array = self._ebmux.pout_mpd_array
            self._mpd_current_array = self._ebmux.mpd_current_array
        
        ## Refactored code
        self._lambda_array = self._sppd_clm_pic._lambda_array
        self._sppd_clm_pic_pout_array  = self._sppd_clm_pic._pout_array
        self._sppd_clm_pic_mpd_current_array = self._sppd_clm_pic._mpd_currrent_array
        if self._no_mux == 2:
            self._enablence_mux_pout_array = self.combine_AB(self._ebmux_1.pout_array, self._ebmux_2.pout_array)
            self._enablence_mux_mpd_current_array = self.combine_AB(self._ebmux_1.mpd_current_array, self._ebmux_2.mpd_current_array)
        elif self._no_mux == 1:
            self._enablence_mux_pout_array = self._ebmux.pout_array
            self._enablence_mux_mpd_current_array = self._ebmux.mpd_current_array