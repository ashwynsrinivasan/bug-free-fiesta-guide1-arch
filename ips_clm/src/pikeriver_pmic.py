''' Pikeriver PMIC'''

# pylint: disable=trailing-whitespace, invalid-name, too-many-locals, locally-disabled, dangerous-default-value, multiple-statements, fixme, line-too-long, attribute-defined-outside-init, protected-access, too-many-arguments, too-many-instance-attributes, too-few-public-methods
from typing import Union, Callable, Any
import numpy as np

class TIA:
    """ TIA"""
    def __init__(
        self,
        input_current: Union[float, Callable] = 10e-6,
        gain_setting: Union[int, Callable] = 3,
        non_inverting_gain: Union[float, Callable] = 4,
        voltage_noise: Union[float, Callable] = 155e-6,
        bandwidth: Union[float, Callable] = 2e6,
        vin_offset: Union[float, Callable] = 0,
        rterm_range: Union[dict, Callable] = {0: 3670, 1: 1223, 2: 611.6, 3: 367},
        name: str = 'TIA'
    ):
        self.name = name
        self._input_current = self._genvar(input_current)
        self._gain_setting = self._genvar(gain_setting)
        self._non_inverting_gain = self._genvar(non_inverting_gain)
        self._voltage_noise = self._genvar(voltage_noise)
        self._bandwidth = self._genvar(bandwidth)
        self._vin_offset = self._genvar(vin_offset)
        self._rterm_range = self._genvar(rterm_range)
        self.update()

    @staticmethod
    def _genvar(gen: Any) -> Any:
        return (gen() if callable(gen) else gen)

    def _gain_setting_check(self):
        if (self._gain_setting == 0) & ((self._input_current > 80e-6) or (self._input_current < 10e-6)):
            print('The input current is out of range for the gain setting')
        elif (self._gain_setting == 1) & ((self._input_current > 240e-6) or (self._input_current < 53e-6)):
            print('The input current is out of range for the gain setting')
        elif (self._gain_setting == 2) & ((self._input_current > 480e-6) or (self._input_current < 106e-6)):
            print('The input current is out of range for the gain setting')
        elif (self._gain_setting == 3) & ((self._input_current > 1000e-6) or (self._input_current < 177e-6)):
            print('The input current is out of range for the gain setting')
      
    def _rterm_update(self):
        self._rterm = self._rterm_range[self._gain_setting]

    def update(self, gain_settings_check = 0):
        ''' Update the TIA'''
        self._rterm_update()
        if gain_settings_check:
            self._gain_setting_check()
        self._output_voltage = (self._rterm * self._input_current + self._vin_offset) * self._non_inverting_gain + np.random.normal(0, self._voltage_noise, 1)[0]

    @property
    def gain_setting(self):
        ''' Gain setting'''
        return self._gain_setting
    
    @gain_setting.setter
    def gain_setting(self, value):
        self._gain_setting = value
        self.update()

    @property
    def input_current(self):
        ''' Input current'''
        return self._input_current

    @input_current.setter
    def input_current(self, value):
        self._input_current = value
        self.update()

    @property
    def vin_offset(self):
        ''' Vin offset '''
        return self._vin_offset

    @vin_offset.setter
    def vin_offset(self, value):
        self._vin_offset = value
        self.update()
    
    @property
    def voltage_noise(self):
        ''' Voltage noise '''
        return self._voltage_noise
    
    @voltage_noise.setter
    def voltage_noise(self, value):
        self._voltage_noise = value
        self.update()


class ADC:
    ''' ADC'''
    def __init__(
        self,
        bit_resolution: Union[int, Callable] = 12,
        input_voltage: Union[float, Callable] = 0.5,
        peak_input_voltage: Union[float, Callable] = 1.8,
        adc_clock_frequency: Union[float, Callable] = 50e6,
        adc_conversion_rate: Union[float, Callable] = 3.125e6,
        different_non_linearity: Union[int, Callable] = 6,
        integral_non_linearity: Union[int, Callable] = 12,
        offset_error: Union[int, Callable] = 6,
        gain_error: Union[int, Callable] = 6,
        total_error: Union[int, Callable] = 14,
        name: str = 'ADC'
    ):
        self.name = name
        self._bit_resolution = self._genvar(bit_resolution)
        self._input_voltage = self._genvar(input_voltage)
        self._peak_input_voltage = self._genvar(peak_input_voltage)
        self._adc_clock_frequency = self._genvar(adc_clock_frequency)
        self._adc_conversion_rate = self._genvar(adc_conversion_rate)
        self._different_non_linearity = self._genvar(different_non_linearity)
        self._integral_non_linearity = self._genvar(integral_non_linearity)
        self._offset_error = self._genvar(offset_error)
        self._gain_error = self._genvar(gain_error)
        self._total_error = self._genvar(total_error)

    @staticmethod
    def _genvar(gen: Any) -> Any:
        return (gen() if callable(gen) else gen)

    @property
    def input_voltage(self):
        ''' Input voltage'''
        return self._input_voltage
    
    @input_voltage.setter
    def input_voltage(self, value):
        self._input_voltage = value
        self.update()

    def update(self):
        ''' Update the ADC'''
        self._output_adc_code = int(self._input_voltage / self._peak_input_voltage * 2**self._bit_resolution + (np.random.uniform(low=-1.0, high=1.0)  * self._total_error))
        if self._output_adc_code > 2**self._bit_resolution-1:
            self._output_adc_code = 2**self._bit_resolution-1
        if self._output_adc_code < 0:
            self._output_adc_code = 0
        self._output_adc_code = hex(self._output_adc_code)

class DAC: 
    ''' DAC'''
    def __init__(
        self,
        bit_resolution: Union[int, Callable] = 10,
        different_non_linearity: Union[float, Callable] = 0.5,
        integral_non_linearity: Union[float, Callable] = 5,
        votlage_dac: Union[float, Callable] = 1.8,
        current_noise: Union[float, Callable] = 200e-6,
        settling_time: Union[float, Callable] = 25e-6,
        dac_code_hex: Union[str, Callable] = hex(2**10-1),
        fs_trim_hex: Union[str, Callable] = '0x0',
        fs_trim_dynamic_range: Union[dict, Callable] = {'0x0': 0.2, '0x1': 0.166, '0x2': 0.126, '0x3': 0.1},
        name: str = 'DAC'
    ):
        self.name = name
        self._bit_resolution = self._genvar(bit_resolution)
        self._differential_non_linearity = self._genvar(different_non_linearity)
        self._integral_non_linearity = self._genvar(integral_non_linearity)
        self._voltage_dac = self._genvar(votlage_dac)
        self._current_noise = self._genvar(current_noise)
        self._output_setting_time = self._genvar(settling_time)
        self._dac_code_hex = self._genvar(dac_code_hex)
        self._fs_trim_hex = self._genvar(fs_trim_hex)
        self._fs_trim_dynamic_range = self._genvar(fs_trim_dynamic_range)
        self.update()

    @staticmethod
    def _genvar(gen: Any) -> Any:
        return (gen() if callable(gen) else gen)
    
    @property
    def dac_code_hex(self):
        ''' DAC code hex'''
        return self._dac_code_hex
    
    @dac_code_hex.setter
    def dac_code_hex(self, value):
        self._dac_code_hex = value
        self.update()

    @property
    def fs_trim_hex(self):
        ''' FS trim hex'''
        return self._fs_trim_hex
    
    @fs_trim_hex.setter
    def fs_trim_hex(self, value):
        self._fs_trim_hex = value
        self.update()
    
    def update(self):
        ''' Update the DAC'''
        self._dynamic_range = self._fs_trim_dynamic_range[self._fs_trim_hex]
        self._output_current = self._dynamic_range*int(self._dac_code_hex, 16)/2**self._bit_resolution + np.random.normal(0, self._current_noise, 1)[0]

class PMIC:
    ''' Pikeriver PMIC'''
    def __init__(
        self,
        no_laser_mpd: Union[int, Callable] = 16,
        no_mux_mpd: Union[int, Callable] = 16,
        pic_mpd_tia_gain_setting: Union[np.array, Callable] = np.ones(16)*3,
        mux_mpd_tia_gain_setting: Union[np.array, Callable] = np.ones(16)*3,
        name: str = 'PMIC'
    ):
        self.name = name
        self._adc_sppd_clm_pic_mpd = [ADC(name=f"PKPMIC_SPPD_CLM_PIC_MPD_ADC_{i}") for i in range(no_laser_mpd)]
        self._adc_enablence_mux_mpd = [ADC(name=f"PKPMIC_ENABLENCE_MUX_MPD_ADC_{i}") for i in range(no_mux_mpd)]
        self._tia_sppd_clm_pic_mpd = [TIA(name=f"PKPMIC_SPPD_CLM_PIC_MPD_TIA_{i}", gain_setting=pic_mpd_tia_gain_setting[i]) for i in range(no_laser_mpd)]
        self._tia_enablence_mux_mpd = [TIA(name=f"PKPMIC_ENABLENCE_MUX_MPD_TIA_{i}", gain_setting=mux_mpd_tia_gain_setting[i]) for i in range(no_mux_mpd)]
        self._dac_drive_current_laser = [DAC(name=f"PKPMIC_LASER_DAC_{i}") for i in range(no_laser_mpd)]

        self._sppd_clm_pic_mpd_current_array = np.ones(no_laser_mpd)*35e-3*1e-2*0.9
        self._enablence_mux_mpd_current_array = np.zeros(no_mux_mpd)*35e-3*5e-2*0.9*0.7
        self._drive_current_array_dac_code_hex = [hex(int(0.125/0.2*2**10)) for i in range(no_laser_mpd)]

    @property
    def sppd_clm_pic_mpd_current_array(self):
        """SPPD CLM PIC MPD"""
        return self._sppd_clm_pic_mpd_current_array
    
    @sppd_clm_pic_mpd_current_array.setter
    def sppd_clm_pic_mpd_current_array(self, value):
        self._sppd_clm_pic_mpd_current_array = value
        self.update()
    
    @property
    def enablence_mux_mpd_current_array(self):
        ''' Enablence mux mpd current array'''
        return self._enablence_mux_mpd_current_array
    
    @enablence_mux_mpd_current_array.setter
    def enablence_mux_mpd_current_array(self, value):
        self._enablence_mux_mpd_current_array = value
        self.update()

    @property
    def drive_current_array_dac_code_hex(self):
        ''' Drive current array dac code hex'''
        return self._drive_current_array_dac_code_hex
    
    @drive_current_array_dac_code_hex.setter
    def drive_current_array_dac_code_hex(self, value):
        self._drive_current_array_dac_code_hex = value
        self.update()

    def tia_sppd_clm_pic_mpd_update(self, tune_gain_setting = 0):
        ''' Update TIA SPPD CLM PIC MPD'''
        for  idd_laser, __ in enumerate(self._tia_sppd_clm_pic_mpd):
            _input_current = self._sppd_clm_pic_mpd_current_array[idd_laser]
            self._tia_sppd_clm_pic_mpd[idd_laser].gain_setting = 3
            if tune_gain_setting:
                if (_input_current >= 106e-6) & (_input_current <= 480e-6):
                    self._tia_sppd_clm_pic_mpd[idd_laser].gain_setting = 2
                elif (_input_current >= 53e-6) & (_input_current <= 240e-6):
                    self._tia_sppd_clm_pic_mpd[idd_laser].gain_setting = 1
                elif (_input_current >= 10e-6) & (_input_current <= 80e-6):
                    self._tia_sppd_clm_pic_mpd[idd_laser].gain_setting = 0
            self._tia_sppd_clm_pic_mpd[idd_laser].input_current = _input_current

    def tia_enablence_mux_mpd_update(self, tune_gain_setting = 0):
        ''' Update TIA enablence mux mpd'''
        for idd_mux, __ in enumerate(self._tia_enablence_mux_mpd):
            _input_current = self._enablence_mux_mpd_current_array[idd_mux]
            self._tia_enablence_mux_mpd[idd_mux].gain_setting = 3
            if tune_gain_setting:
                if (_input_current >= 106e-6) & (_input_current <= 480e-6):
                    self._tia_enablence_mux_mpd[idd_mux].gain_setting = 2
                elif (_input_current >= 53e-6) & (_input_current <= 240e-6):
                    self._tia_enablence_mux_mpd[idd_mux].gain_setting = 1
                elif (_input_current >= 10e-6) & (_input_current <= 80e-6):
                    self._tia_enablence_mux_mpd[idd_mux].gain_setting = 0
                
            self._tia_enablence_mux_mpd[idd_mux].input_current = _input_current

    def adc_sppd_clm_pic_mpd_update(self):
        ''' Update the ADC sppd clm pic mpd'''
        self.tia_sppd_clm_pic_mpd_update()
        for idd_laser, __ in enumerate(self._adc_sppd_clm_pic_mpd):
            self._adc_sppd_clm_pic_mpd[idd_laser].input_voltage = self._tia_sppd_clm_pic_mpd[idd_laser]._output_voltage
        

    def adc_enablence_mux_mpd_update(self):
        ''' Update the ADC enablence mux mpd'''
        self.tia_enablence_mux_mpd_update()
        for idd_mux, __ in enumerate(self._adc_enablence_mux_mpd):
            self._adc_enablence_mux_mpd[idd_mux].input_voltage = self._tia_enablence_mux_mpd[idd_mux]._output_voltage

    def dac_laser_drive_current_update(self):
        ''' Update the DAC laser drive current'''
        for idd_laser, __ in enumerate(self._dac_drive_current_laser):
            _dac_code_hex = self._drive_current_array_dac_code_hex[idd_laser]
            self._dac_drive_current_laser[idd_laser].fs_trim_hex = '0x0'
            if int(_dac_code_hex, 16)*2**10-1 < 0.166/0.2:
                self._dac_drive_current_laser[idd_laser].fs_trim_hex = '0x1'
            elif int(_dac_code_hex, 16)*2**10-1 < 0.126/0.2:
                self._dac_drive_current_laser[idd_laser].fs_trim_hex = '0x2'
            elif int(_dac_code_hex, 16)*2**10-1 < 0.1/0.2:
                self._dac_drive_current_laser[idd_laser].fs_trim_hex = '0x3'
            self._dac_drive_current_laser[idd_laser].dac_code_hex = _dac_code_hex
            self._dac_drive_current_laser[idd_laser].update()

    def adc_sppd_clm_pic_mpd_code_update(self):
        ''' Update the ADC sppd clm pic mpd code'''
        self._adc_sppd_clm_pic_mpd_code = []
        for idd_laser, adc_sppd_clm_pic_mpd_value in enumerate(self._adc_sppd_clm_pic_mpd):
            self._adc_sppd_clm_pic_mpd[idd_laser].update()
            self._adc_sppd_clm_pic_mpd_code.append(int(adc_sppd_clm_pic_mpd_value._output_adc_code, 16))
        self._adc_sppd_clm_pic_mpd_code = np.array(self._adc_sppd_clm_pic_mpd_code)
    
    def adc_enablence_mux_mpd_code_update(self):
        ''' Update the ADC enablence mux mpd code'''
        self._adc_enablence_mux_mpd_code =[]
        for idd_mux, adc_enablence_mux_mpd_value in enumerate(self._adc_enablence_mux_mpd):
            self._adc_enablence_mux_mpd[idd_mux].update()
            self._adc_enablence_mux_mpd_code.append(int(adc_enablence_mux_mpd_value._output_adc_code, 16))
        self._adc_enablence_mux_mpd_code = np.array(self._adc_enablence_mux_mpd_code)
    
    def dac_laser_drive_current_code_update(self):
        ''' Update the DAC laser drive current code'''
        self._dac_laser_drive_current_array = []
        for idd_laser, dac_drive_current_laser_value in enumerate(self._dac_drive_current_laser):
            self._dac_drive_current_laser[idd_laser].update()
            self._dac_laser_drive_current_array.append(dac_drive_current_laser_value._output_current)
        self._dac_laser_drive_current_array = np.array(self._dac_laser_drive_current_array)

    def update(self):
        ''' Update the PMIC'''
        self.adc_sppd_clm_pic_mpd_update()
        self.adc_enablence_mux_mpd_update()
        self.dac_laser_drive_current_update()
        self.adc_sppd_clm_pic_mpd_code_update()
        self.adc_enablence_mux_mpd_code_update()
        self.dac_laser_drive_current_code_update()