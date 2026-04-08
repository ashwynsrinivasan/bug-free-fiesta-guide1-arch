''' SPPD CLM Controller'''

# pylint: too-many-statements, disable=trailing-whitespace, invalid-name, too-many-locals, locally-disabled, dangerous-default-value, multiple-statements, fixme, line-too-long, attribute-defined-outside-init, protected-access, too-many-arguments, too-many-instance-attributes, too-few-public-methods
import numpy as np
from .sppd_clm_optics import SPPDCLMOPTICS_LAMBDA_AB as SPPDCLMOPTICS
from .pikeriver_pmic import PMIC
from .laser_control.tec import TEC 
from .laser_control.pid import pid as PID
from .laser_control.thermistor import THERMISTOR
from tqdm import tqdm
import matplotlib.pyplot as plt

class CONTROLLER:
    ''' Controller class for SPPD CLM'''
    _sppd_clm_optics: SPPDCLMOPTICS = None
    _pmic: PMIC = None
    _tec: TEC = None
    _pid_tec: PID = None
    _pid_idrive: PID = None
    _thermistor: THERMISTOR = None
    _drive_current_array: np.ndarray
    _tstart = 0
    _tend = 32.5
    _tdither = 10
    _tagg_1 = 17.5
    _tagg_2 = 25
    
    _agg_1_amp = 5
    _agg_2_amp = 5
    
    _dt = 1e-2
    
    def __init__(self, 
                 sppd_clm_optics = SPPDCLMOPTICS(), 
                 pmic = PMIC(), 
                 drive_current_array = np.ndarray,
                 tec = None, 
                 pid_tec = None, 
                 pid_idrive = None, 
                 thermistor = None,
                 dt = 1e-2):
        
        self._sppd_clm_optics = sppd_clm_optics
        self._pmic = pmic
        
        self._drive_current_array = drive_current_array
        
        self._dt = dt
        if not tec is None:
            self._tec = tec
        else:
            self._tec = TEC(Th = 274 + 65, dTmax = 70, Qmax = 34.6, Imax = 7, Umax = 8.8, ACR = 1.06, time_constant=1)
        if not pid_tec is None:
            self._pid_tec = pid_tec
        else:
            self._pid_tec = PID(kp=0.583, ki=0.0928, kd=0.01, dt=self._dt, umax=3, umin=-3)
        if not pid_idrive is None:
            self._pid_idrive = pid_idrive
        else:
            self._pid_idrive = [PID(kp=10, ki=1000/(10*self._dt), kd=0, dt=self._dt) for i in range(len(self._drive_current_array))]
        if not thermistor is None:
            self._thermistor = thermistor
        else:
            self._thermistor = THERMISTOR(theta_jsm=1e3, tau=2.5)                                                                                             

    
    def drive_current_tuning(self,
                            tc,
                            drive_current_range,
                            target_adc_sppd_clm_mpd_adc_code,
                            routine = {
                                "power_wavelength_calibration": True,
                                "drive_current_tuning": False,
                            }):
        '''
        Drive current tuning
        '''

        no_lasers = len(self._sppd_clm_optics._wavelength_fabrication)
        drive_current_array_dac_code_hex_matched = {}
        _pmic_adc_clm_pic_mpd_code_temp = []
        _pmic_adc_enablence_mux_code_temp = []
        drive_current_array_dac_code_hex_original = self._pmic.drive_current_array_dac_code_hex
        
        for idd_laser in range(int(no_lasers/2)):
            _pmic_adc_clm_pic_mpd_code_temp = []
            _pmic_adc_enablence_mux_code_temp = []
            for __, drive_current_value in enumerate(drive_current_range):
                ## Sweeping the drive current
                drive_current_array_dac_code_hex_original[8+idd_laser] = hex(int(drive_current_value/0.2*2**10))
                drive_current_array_dac_code_hex_original[7-idd_laser] = hex(int(drive_current_value/0.2*2**10))
                self._pmic.drive_current_array_dac_code_hex = drive_current_array_dac_code_hex_original
                _dac_drive_current = self._pmic._dac_laser_drive_current_array
                self._sppd_clm_optics.update(i_array = _dac_drive_current, tc =tc)
                
                ## Updating the PMIC with MPD current array updates
                self._pmic.sppd_clm_pic_mpd_current_array = self._sppd_clm_optics._sppd_clm_pic_mpd_current_array
                self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array
                ## Storing the sweep updates into temp list
                _pmic_adc_clm_pic_mpd_code_temp.append(self._pmic._adc_sppd_clm_pic_mpd_code)
                _pmic_adc_enablence_mux_code_temp.append(self._pmic._adc_enablence_mux_mpd_code)

            _pmic_adc_enablence_mux_code_array = np.array(_pmic_adc_enablence_mux_code_temp).reshape((len(drive_current_range), no_lasers))
            _pmic_adc_clm_pic_mpd_code_array = np.array(_pmic_adc_clm_pic_mpd_code_temp).reshape((len(drive_current_range), no_lasers))

            if not routine["drive_current_tuning"]:
                if routine["power_wavelength_calibration"]:
                    idd_match = np.argmin(np.abs(_pmic_adc_clm_pic_mpd_code_array[:, 8+idd_laser] - target_adc_sppd_clm_mpd_adc_code[8+idd_laser]))
                    drive_current_array_dac_code_hex_matched[8+idd_laser] = hex(int( drive_current_range[idd_match]/0.2*2**10))
                    idd_match = np.argmin(np.abs(_pmic_adc_clm_pic_mpd_code_array[:, 7-idd_laser] - target_adc_sppd_clm_mpd_adc_code[7-idd_laser]))
                    drive_current_array_dac_code_hex_matched[7-idd_laser] = hex(int( drive_current_range[idd_match]/0.2*2**10))
                else:
                    idd_match = np.argmax(np.abs(_pmic_adc_enablence_mux_code_array[:, 8+idd_laser]))
                    drive_current_array_dac_code_hex_matched[8+idd_laser] = hex(int(drive_current_range[idd_match]/0.2*2**10))
                    idd_match = np.argmax(np.abs(_pmic_adc_enablence_mux_code_array[:, 7-idd_laser]))
                    drive_current_array_dac_code_hex_matched[7-idd_laser] = hex(int(drive_current_range[idd_match]/0.2*2**10))
            else:
                ## Final drive current tuning
                idd_match = np.argmax(np.abs(_pmic_adc_enablence_mux_code_array[:, 8+idd_laser]))
                drive_current_array_dac_code_hex_matched[8+idd_laser] = hex(int(drive_current_range[idd_match]/0.2*2**10))
                # if idd_match > 0.75*len(drive_current_range) or idd_match < 0.25*len(drive_current_range):
                #     drive_current_array_dac_code_hex_matched[8+idd_laser] = hex(int(drive_current_range[int(len(drive_current_range)/2)]/0.2*2**10))
                
                idd_match_2 = np.argmax(np.abs(_pmic_adc_enablence_mux_code_array[:, 7-idd_laser]))
                drive_current_array_dac_code_hex_matched[7-idd_laser] = hex(int(drive_current_range[idd_match_2]/0.2*2**10))
                # if idd_match_2 > 0.75*len(drive_current_range) or idd_match_2 < 0.25*len(drive_current_range):
                #     drive_current_array_dac_code_hex_matched[7-idd_laser] = hex(int(drive_current_range[int(len(drive_current_range)/2)]/0.2*2**10))
                # plt.figure(figsize=(10,3))
                # plt.subplot(121)
                # plt.plot(drive_current_range, _pmic_adc_enablence_mux_code_array[:, 8+idd_laser]*3, label = f"EBMUX MPD {8+idd_laser}")
                # plt.plot(drive_current_range, _pmic_adc_enablence_mux_code_array[:, 7-idd_laser]*3, label = f"EBMUX MPD {7-idd_laser}")
                # plt.plot(drive_current_range, _pmic_adc_clm_pic_mpd_code_array[:, 8+idd_laser], label = f"DFB PIC MPD {8+idd_laser}")
                # plt.plot(drive_current_range, _pmic_adc_clm_pic_mpd_code_array[:, 7-idd_laser], label = f"DFB PIC MPD {7-idd_laser}")
                # plt.title(f"{8+idd_laser}, {drive_current_range[idd_match]:0.3f} \n {7-idd_laser}, {drive_current_range[idd_match_2]:0.3f}")
                # plt.subplot(122)
                # plt.plot(drive_current_range, _pmic_adc_enablence_mux_code_array[:, 8+idd_laser]/_pmic_adc_clm_pic_mpd_code_array[:, 8+idd_laser])
                # plt.plot(drive_current_range, _pmic_adc_enablence_mux_code_array[:, 7-idd_laser]/_pmic_adc_clm_pic_mpd_code_array[:, 7-idd_laser])
                # plt.title(f"{8+idd_laser}, {drive_current_range[idd_match_3]:0.3f} \n {7-idd_laser}, {drive_current_range[idd_match_4]:0.3f}")
        self._pmic.drive_current_array_dac_code_hex = drive_current_array_dac_code_hex_matched

 
    def tec_tuning(self,
                    temperature_array
                    ):
        '''
        TEC Tuning
        ''' 
    
        enablence_mux_output_power_array = np.zeros((len(temperature_array), len(self._sppd_clm_optics._enablence_mux_mpd_current_array)))
        enablence_mux_output_lambda_array = np.zeros((len(temperature_array), len(self._sppd_clm_optics._enablence_mux_mpd_current_array)))
        _pmic_adc_enablence_mux_mpd_current_array = np.zeros((len(temperature_array), len(self._sppd_clm_optics._enablence_mux_mpd_current_array)))

        # Sweeping the TEC temperature and registering the MPD current from SPPD PIC and EBMUX
        for idd_temperature, temperature_value in enumerate(temperature_array):
            self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = temperature_value)
            self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array
            
            _pmic_adc_enablence_mux_mpd_current_array[idd_temperature, :] = self._pmic._adc_enablence_mux_mpd_code
            enablence_mux_output_power_array[idd_temperature, :] = self._sppd_clm_optics._enablence_mux_pout_array
            enablence_mux_output_lambda_array[idd_temperature, :] = self._sppd_clm_optics._lambda_array

        return enablence_mux_output_power_array, enablence_mux_output_lambda_array, _pmic_adc_enablence_mux_mpd_current_array
    

    def tec_temperature_optimum(self,
                                _pmic_adc_enablence_mux_mpd_current_array,
                                temperature_array
                                ):
        '''
        Searching the optimal temperature for TEC
        '''
        nominal_temperature = np.mean(temperature_array)
        idx_temp = np.argmin(np.abs(temperature_array - nominal_temperature))
        nominal_temperature_array = temperature_array[idx_temp] * np.ones(len(self._sppd_clm_optics._wavelength_fabrication))
        
        _pmic_adc_enablence_mux_mpd_peak_current_array = np.zeros(len(self._sppd_clm_optics._wavelength_fabrication))
        output_power_array = np.zeros(len(self._sppd_clm_optics._wavelength_fabrication))
        output_lambda_array = np.zeros(len(self._sppd_clm_optics._wavelength_fabrication))

        tec_set_point_temperature = temperature_array[np.argmax(np.sum(_pmic_adc_enablence_mux_mpd_current_array, axis = 1))]
        
        return tec_set_point_temperature, output_lambda_array, output_power_array, nominal_temperature_array, _pmic_adc_enablence_mux_mpd_peak_current_array

    def calibrate(
                self, 
                tc = 273 + 44.90,
                temperature_array = np.linspace(36, 55, 51)+273, 
                drive_current_array= np.linspace(0.1, 0.2, 101), 
                sppd_clm_mpd_current_array_target =  36e-3*1e-2*0.9,
                power_wavelength_calibration = True,
                drive_current_tuning = False,
                tec_temperature_tuning = True,
                percent_range = np.linspace(0.8, 1.2, 41)
                ):
        '''
        Initialize SPPD CLM
        '''
        
        no_lasers = len(self._sppd_clm_optics._wavelength_fabrication)

        self._pmic.sppd_clm_pic_mpd_current_array = sppd_clm_mpd_current_array_target * np.ones(no_lasers)
        target_adc_sppd_clm_mpd_adc_code = self._pmic._adc_sppd_clm_pic_mpd_code
        self._pmic.sppd_clm_mpd_current_array = self._sppd_clm_optics._sppd_clm_pic_mpd_current_array
        self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array

        # Find the closest drive current for each laser to match the target laser output power
        self.drive_current_tuning(
            tc, 
            drive_current_array, 
            target_adc_sppd_clm_mpd_adc_code, 
            routine = {
                "power_wavelength_calibration": power_wavelength_calibration,
                "drive_current_tuning": False,
            })
        self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = np.mean(temperature_array))

        # Find the closest temperature to the nominal temperature for each laser
        enablence_mux_output_power_array, enablence_mux_output_lambda_array, _pmic_adc_enablence_mux_mpd_current_array = self.tec_tuning(temperature_array)
        
        if tec_temperature_tuning:
            self._warning_flag = 0
            tec_set_point_temperature, output_lambda_array, output_power_array, nominal_temperature_array, _pmic_adc_enablence_mux_mpd_peak_current_array = self.tec_temperature_optimum(_pmic_adc_enablence_mux_mpd_current_array, temperature_array)
        else:
            self._warning_flag = 0
            tec_set_point_temperature = tc
            tc_arg = np.argmin(np.abs(temperature_array - tc))
            nominal_temperature_array = temperature_array[tc_arg] * np.ones(len(self._sppd_clm_optics._wavelength_fabrication))
            _pmic_adc_enablence_mux_mpd_peak_current_array = _pmic_adc_enablence_mux_mpd_current_array[tc_arg, :]
            output_power_array = enablence_mux_output_power_array[tc_arg, :]
            output_lambda_array = enablence_mux_output_lambda_array[tc_arg, :]

        if drive_current_tuning:
            drive_current_range = np.mean(self._pmic._dac_laser_drive_current_array) * percent_range 
            self.drive_current_tuning(
                tec_set_point_temperature, 
                drive_current_range, 
                target_adc_sppd_clm_mpd_adc_code, routine = {
                    "power_wavelength_calibration": False,
                    "drive_current_tuning": drive_current_tuning,
                    })
            self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = tec_set_point_temperature)
            output_lambda_array = self._sppd_clm_optics._lambda_array
            output_power_array = self._sppd_clm_optics._enablence_mux_pout_array

        self._tec_set_point_temperature = tec_set_point_temperature
        self._nominal_temperature_array = nominal_temperature_array
        self._pmic_adc_enablence_mux_mpd_current_array = _pmic_adc_enablence_mux_mpd_current_array
        self._pmic_adc_enablence_mux_mpd_peak_current_array = _pmic_adc_enablence_mux_mpd_peak_current_array
        self._enablence_output_power_array = output_power_array
        self._lambda_array = output_lambda_array
        _laser_junction_temperature_list = [self._sppd_clm_optics._sppd_clm_pic._laser_array[idd_laser]._temperature for idd_laser in range(no_lasers)]
        self._laser_junction_temperature_array = np.array(_laser_junction_temperature_list)
        self._laser_pout_array = self._sppd_clm_optics._sppd_clm_pic._pout_array
        self._drive_current_array = drive_current_array
        self._temperature_array = temperature_array
        
        self._data_export_dict = {       
            "nom_temp_array" : self._nominal_temperature_array,
            "tec_set_point_temperature" : self._tec_set_point_temperature,
            "mux_current_array" : self._pmic_adc_enablence_mux_mpd_current_array,
            "peak_current_array" : self._pmic_adc_enablence_mux_mpd_peak_current_array,
            "output_power_array" : self._enablence_output_power_array,
            "output_lambda_array" : self._lambda_array,
            "current_array" : self._drive_current_array,
            "cal_temperature_array" : self._temperature_array,
            "drive_current_array" : self._pmic._dac_laser_drive_current_array,
            "laser_junction_temperature_array" : self._laser_junction_temperature_array,
            "laser_pout_array" : self._laser_pout_array,
            "warning_flag" : self._warning_flag
        }
    
    def time_step_statistics(self,
                             time_offset = 2
                             ):
        _time_arg = np.argmin(np.abs(self._t - self._tdither - time_offset))
        _time_arg_agg_1 = np.argmin(np.abs(self._t - self._tagg_1))
        _time_arg_agg_2 = np.argmin(np.abs(self._t - self._tagg_2))
        
        self._time_step_statistics = {
            "drive_current": {
                "std": np.std(self._drive_current_array_time_step[:, _time_arg::], axis = 1),
                "mean": np.mean(self._drive_current_array_time_step[:, _time_arg::], axis = 1),
                "max": np.max(self._drive_current_array_time_step[:, _time_arg::], axis = 1),
                "min": np.min(self._drive_current_array_time_step[:, _time_arg::], axis = 1),
                "std_no_agg": np.std(self._drive_current_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "mean_no_agg": np.mean(self._drive_current_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "max_no_agg": np.max(self._drive_current_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "min_no_agg": np.min(self._drive_current_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "std_agg_1": np.std(self._drive_current_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "mean_agg_1": np.mean(self._drive_current_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "max_agg_1": np.max(self._drive_current_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "min_agg_1": np.min(self._drive_current_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "std_agg_2": np.std(self._drive_current_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "mean_agg_2": np.mean(self._drive_current_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "max_agg_2": np.max(self._drive_current_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "min_agg_2": np.min(self._drive_current_array_time_step[:, _time_arg_agg_2::], axis = 1),
                },
            "enablence_output_power": {
                "std": np.std(self._enablence_output_power_array_time_step[:, _time_arg::], axis = 1),
                "mean": np.mean(self._enablence_output_power_array_time_step[:, _time_arg::], axis = 1),
                "max": np.max(self._enablence_output_power_array_time_step[:, _time_arg::], axis = 1),
                "min": np.min(self._enablence_output_power_array_time_step[:, _time_arg::], axis = 1),
                "std_no_agg": np.std(self._enablence_output_power_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "mean_no_agg": np.mean(self._enablence_output_power_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "max_no_agg": np.max(self._enablence_output_power_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "min_no_agg": np.min(self._enablence_output_power_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "std_agg_1": np.std(self._enablence_output_power_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "mean_agg_1": np.mean(self._enablence_output_power_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "max_agg_1": np.max(self._enablence_output_power_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "min_agg_1": np.min(self._enablence_output_power_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "std_agg_2": np.std(self._enablence_output_power_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "mean_agg_2": np.mean(self._enablence_output_power_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "max_agg_2": np.max(self._enablence_output_power_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "min_agg_2": np.min(self._enablence_output_power_array_time_step[:, _time_arg_agg_2::], axis = 1),
                },
            "enablence_output_lambda": {
                "std": np.std(self._enablence_output_lambda_array_time_step[:, _time_arg::], axis = 1),
                "mean": np.mean(self._enablence_output_lambda_array_time_step[:, _time_arg::], axis = 1),
                "max": np.max(self._enablence_output_lambda_array_time_step[:, _time_arg::], axis = 1),
                "min": np.min(self._enablence_output_lambda_array_time_step[:, _time_arg::], axis = 1),
                "std_no_agg": np.std(self._enablence_output_lambda_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "mean_no_agg": np.mean(self._enablence_output_lambda_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "max_no_agg": np.max(self._enablence_output_lambda_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "min_no_agg": np.min(self._enablence_output_lambda_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "std_agg_1": np.std(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "mean_agg_1": np.mean(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "max_agg_1": np.max(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "min_agg_1": np.min(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "std_agg_2": np.std(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "mean_agg_2": np.mean(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "max_agg_2": np.max(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "min_agg_2": np.min(self._enablence_output_lambda_array_time_step[:, _time_arg_agg_2::], axis = 1),
                },
            "sppd_pout": {
                "std": np.std(self._sppd_pout_array_time_step[:, _time_arg::], axis = 1),
                "mean": np.mean(self._sppd_pout_array_time_step[:, _time_arg::], axis = 1),
                "max": np.max(self._sppd_pout_array_time_step[:, _time_arg::], axis = 1),
                "min": np.min(self._sppd_pout_array_time_step[:, _time_arg::], axis = 1),
                "std_no_agg": np.std(self._sppd_pout_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "mean_no_agg": np.mean(self._sppd_pout_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "max_no_agg": np.max(self._sppd_pout_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "min_no_agg": np.min(self._sppd_pout_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "std_agg_1": np.std(self._sppd_pout_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "mean_agg_1": np.mean(self._sppd_pout_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "max_agg_1": np.max(self._sppd_pout_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "min_agg_1": np.min(self._sppd_pout_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "std_agg_2": np.std(self._sppd_pout_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "mean_agg_2": np.mean(self._sppd_pout_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "max_agg_2": np.max(self._sppd_pout_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "min_agg_2": np.min(self._sppd_pout_array_time_step[:, _time_arg_agg_2::], axis = 1),
                },
            "sppd_laser_junction_temperature": {
                "std": np.std(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg::], axis = 1),
                "mean": np.mean(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg::], axis = 1),
                "max": np.max(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg::], axis = 1),
                "min": np.min(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg::], axis = 1),
                "std_no_agg": np.std(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "mean_no_agg": np.mean(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "max_no_agg": np.max(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "min_no_agg": np.min(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg:_time_arg_agg_1], axis = 1),
                "std_agg_1": np.std(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "mean_agg_1": np.mean(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "max_agg_1": np.max(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "min_agg_1": np.min(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_1:_time_arg_agg_2], axis = 1),
                "std_agg_2": np.std(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "mean_agg_2": np.mean(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "max_agg_2": np.max(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_2::], axis = 1),
                "min_agg_2": np.min(self._sppd_laser_junction_temperature_array_time_step[:, _time_arg_agg_2::], axis = 1),
                },
        }
        
    
    def control(
            self,
            dither_amp = 0.002,
            update_step = 0.001,
            aggressor_duration = 1.0,
            pid_lock_bool = False,
            lin_sweep_range = 0.2,
            output_power_clamp_bool = False,
            output_power_clamp = 40e-3,
            aggressor_bool = True,
    ):
        ''' Control the SPPD CLM'''
        t = np.arange(self._tstart, self._tend + self._dt, self._dt)
        n_time_steps = len(t)
        no_lasers = len(self._sppd_clm_optics._wavelength_fabrication)
        
        # Controlling the drive current
        _drive_current_array = self._pmic._dac_laser_drive_current_array
        _drive_current_array_time_step = np.ones((no_lasers, n_time_steps))
        
        for idd_laser in range(no_lasers):
            _drive_current_array_time_step[idd_laser, :] = _drive_current_array[idd_laser] * np.ones(n_time_steps)
        
        _enablence_output_power_array_time_step = np.zeros((no_lasers, n_time_steps))
        _enablence_output_lambda_array_time_step = np.zeros((no_lasers, n_time_steps))
        
        _pmic_adc_clm_pic_mpd_code_time_step = np.zeros((no_lasers, n_time_steps))
        _pmic_adc_enablence_mux_mpd_code_time_step = np.zeros((no_lasers, n_time_steps))
        
        _sppd_pout_array_time_step = np.zeros((no_lasers, n_time_steps))
        
        _sppd_laser_junction_temperature_array_time_step = np.zeros((no_lasers, n_time_steps))
        
        # Controlling the TEC
        q = np.zeros((n_time_steps, 1))
        i_tec = np.zeros((n_time_steps, 1))
        v_tec = np.zeros((n_time_steps, 1))
        t_case = np.zeros((n_time_steps, 1))
        tec_th = np.zeros((n_time_steps, 1))
        
        tref = self._tec_set_point_temperature
        tick = 0
            
        ## Pikeriver quantization for sppd laser drive current
        _drive_current_array_dac_code_temp = [hex(int((_drive_current_array[idd_laser] + tick * dither_amp - (1-tick)*dither_amp)/0.2*(2**10))) for idd_laser in range(no_lasers)]
        self._pmic.drive_current_array_dac_code_hex = _drive_current_array_dac_code_temp
        self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = self._tec.Ta)
            
        ## Updating the PMIC with MPD current array updates
        self._pmic.sppd_clm_pic_mpd_current_array = self._sppd_clm_optics._sppd_clm_pic_mpd_current_array
        self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array
        
        ## Target MPD code for SPPD PIC
        sppd_clm_mpd_current_array_target =  output_power_clamp*1e-2*0.9
        self._pmic.sppd_clm_pic_mpd_current_array = sppd_clm_mpd_current_array_target * np.ones(no_lasers)
        target_adc_sppd_clm_mpd_adc_code = self._pmic._adc_sppd_clm_pic_mpd_code
        
        for step in range(n_time_steps):
            init = not bool(step)
            
            self._tec.update(qa = q[step,0], i = i_tec[step, 0], dt = self._dt, init = init)
            v_tec[step, 0]  = self._tec.V
            
            # Thermistor update            
            self._thermistor.update(Ta = self._tec.Ta, dt = self._dt, init = init)
            t_case[step, 0] = self._thermistor._temperature_thermistor            
            _drive_current_array_time_step[:, step] = _drive_current_array
            
            # Laser update
            if tick == 0:
                tick = 1
            else:
                tick = 0
            
            ## Pikeriver quantization for sppd laser drive current                               
            _drive_current_array_dac_code_temp = [hex(int((_drive_current_array[idd_laser] + tick * dither_amp - (1-tick)*dither_amp)/0.2*(2**10))) for idd_laser in range(no_lasers)]
            self._pmic.drive_current_array_dac_code_hex = _drive_current_array_dac_code_temp
            self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = self._tec.Ta)
            
            ## Updating the PMIC with MPD current array updates
            self._pmic.sppd_clm_pic_mpd_current_array = self._sppd_clm_optics._sppd_clm_pic_mpd_current_array
            self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array
            
            ## Storing the sweep updates into the time step arrays
            _enablence_output_power_array_time_step[:, step] = self._sppd_clm_optics._enablence_mux_pout_array
            _enablence_output_lambda_array_time_step[:, step] = self._sppd_clm_optics._lambda_array
            
            _pmic_adc_clm_pic_mpd_code_time_step[:, step] = self._pmic._adc_sppd_clm_pic_mpd_code
            _pmic_adc_enablence_mux_mpd_code_time_step[:, step] = self._pmic._adc_enablence_mux_mpd_code
            
            for idd_laser in range(no_lasers):
                _sppd_laser_junction_temperature_array_time_step[idd_laser, step] = self._sppd_clm_optics._sppd_clm_pic._laser_array[idd_laser]._temperature
            
            # TEC PID control
            error = -1 * (tref - t_case[step, 0])
            i_tec_step = self._pid_tec.update(error)
            
            if step < n_time_steps - 1:
                i_tec[step + 1, 0] = i_tec_step
                
            # Drive current control - wait 20 seconds
            if t[step] > self._tdither and np.mod(step,2) == 0:
                error_code = _pmic_adc_enablence_mux_mpd_code_time_step[:, step]/_pmic_adc_clm_pic_mpd_code_time_step[:,step] - _pmic_adc_enablence_mux_mpd_code_time_step[:, step-1]/_pmic_adc_clm_pic_mpd_code_time_step[:,step-1]                
            
                for idd_laser in range(no_lasers):
                    if pid_lock_bool:
                        _drive_current_array[idd_laser] = _drive_current_array[idd_laser] + self._pid_idrive[idd_laser].update(np.mean(error_code))
                    else: 
                        if error_code[idd_laser] > 0:
                            _drive_current_array[idd_laser] = _drive_current_array[idd_laser] + update_step
                        else:
                            _drive_current_array[idd_laser] = _drive_current_array[idd_laser] - update_step
                        
                        if _drive_current_array[idd_laser] < _drive_current_array_time_step[idd_laser,0] * (1-lin_sweep_range):
                            _drive_current_array[idd_laser] = _drive_current_array_time_step[idd_laser,0] * (1-lin_sweep_range)
                        if _drive_current_array[idd_laser] > _drive_current_array_time_step[idd_laser,0] * (1+lin_sweep_range):
                            _drive_current_array[idd_laser] = _drive_current_array_time_step[idd_laser,0] * (1+lin_sweep_range)
            
             
            ## Pikeriver quantization for sppd laser drive current                               
            _drive_current_array_dac_code_temp = [hex(int((_drive_current_array[idd_laser] + tick * dither_amp - (1-tick)*dither_amp)/0.2*(2**10))) for idd_laser in range(no_lasers)]
            self._pmic.drive_current_array_dac_code_hex = _drive_current_array_dac_code_temp
            self._sppd_clm_optics.update(i_array = self._pmic._dac_laser_drive_current_array, tc = self._tec.Ta)
            
            ## Updating the PMIC with MPD current array updates
            self._pmic.sppd_clm_pic_mpd_current_array = self._sppd_clm_optics._sppd_clm_pic_mpd_current_array
            self._pmic.enablence_mux_mpd_current_array = self._sppd_clm_optics._enablence_mux_mpd_current_array
            
            if output_power_clamp_bool:
                for idd_laser in range(no_lasers):
                    if self._pmic._adc_sppd_clm_pic_mpd_code[idd_laser] > target_adc_sppd_clm_mpd_adc_code[idd_laser]:
                        _drive_current_array[idd_laser] = _drive_current_array[idd_laser] - update_step
                
                
            if step < n_time_steps - 1:
                q[step+1, 0]  = np.sum(self._sppd_clm_optics._sppd_clm_pic._total_power_loss) * 2
            
            if aggressor_bool:
                if t[step] >= self._tagg_1 and t[step] < (self._tagg_1 + aggressor_duration):
                    self._tec.Th = 273 + 65 - (t[step] - self._tagg_1)*self._agg_1_amp
                
                if t[step] >= self._tagg_2 and t[step] < (self._tagg_2 + aggressor_duration):
                    self._tec.Th = 273 + 65 + (t[step] - self._tagg_2)*self._agg_2_amp - self._agg_1_amp * aggressor_duration
            
            _sppd_pout_array_time_step[:, step]  = self._sppd_clm_optics._sppd_clm_pic._pout_array
            
            tec_th[step] = self._tec.Th
                
        self._t = t
        self._drive_current_array_time_step = _drive_current_array_time_step
        
        self._enablence_output_power_array_time_step = _enablence_output_power_array_time_step * 0.89
        self._enablence_output_lambda_array_time_step = _enablence_output_lambda_array_time_step
        
        self._pmic_adc_clm_pic_mpd_code_time_step = _pmic_adc_clm_pic_mpd_code_time_step
        self._pmic_adc_enablence_mux_mpd_code_time_step = _pmic_adc_enablence_mux_mpd_code_time_step
        
        self._sppd_pout_array_time_step = _sppd_pout_array_time_step
        
        self._sppd_laser_junction_temperature_array_time_step = _sppd_laser_junction_temperature_array_time_step
        
        self._q = q
        self._i_tec = i_tec
        self._v_tec = v_tec
        self._t_case = t_case
        self._tec_th = tec_th
        self.time_step_statistics()
