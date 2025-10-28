#!/usr/bin/env python3
"""
Module Analysis Framework
========================

This module provides comprehensive analysis for IPS CLM EVT OFC data analysis.

Main Class:
- module_analysis: Main analysis class with methods for calibration, state validation, and mission mode analysis

Author: Analysis Framework
Date: October 2025
"""

import pandas as pd
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class module_analysis:
    """
    Main analysis class for IPS CLM EVT OFC data analysis.
    
    This class provides methods for:
    - calibration: Analysis and validation of calibration data
    - state_validation: System state validation and monitoring  
    - mission_mode: Mission mode operation analysis
    """
    
    def __init__(self, base_path):
        """Initialize module analysis with base path."""
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "analysis_results"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.state_validation_path = self.results_path / "state_validation"
        self.state_validation_path.mkdir(parents=True, exist_ok=True)
        
        self.mission_mode_path = self.results_path / "mission_mode"
        self.mission_mode_path.mkdir(parents=True, exist_ok=True)
        
        self.calibration_path = self.results_path / "calibration"
        self.calibration_path.mkdir(parents=True, exist_ok=True)
        
        # Load reference grid
        self.reference_grid = self._load_reference_grid()
        
        # Load specifications
        self.specifications = self._load_specifications()
        
        # Available modules
        self.modules = ['156', '157', '164', '165', '167', '168', '170', '171']
        self.current_module = None
    
    def _load_reference_grid(self):
        """Load reference grid from YAML file"""
        ref_grid_path = self.base_path / "analysis_src" / "reference_grid.yaml"
        try:
            with open(ref_grid_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load reference grid: {e}")
            return None
    
    def _load_specifications(self):
        """Load specifications from YAML file"""
        spec_path = self.base_path / "analysis_src" / "specification.yaml"
        try:
            with open(spec_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load specifications: {e}")
            return None
        
    def calibration(self, modules=None):
        """
        Calibration analysis and validation method.
        
        This method handles analysis of calibration data, responsivity measurements,
        and validation of calibration parameters across different modules.
        
        Args:
            modules (list): List of module serial numbers to analyze. If None, analyze all modules.
        """
        if modules is None:
            modules = self.modules
        
        print("=" * 80)
        print("Calibration - FRU Summary Analysis")
        print("=" * 80)
        
        # Collect data for all modules
        all_modules_data = {}
        
        for module in modules:
            try:
                # Create individual FRU plot
                self.create_fru_summary(module)
                
                # Collect data for summary
                module_path = self.base_path / module
                config_files = list(module_path.glob("*config*.yaml"))
                full_sn = module
                if config_files:
                    parts = config_files[0].name.split('_')
                    if len(parts) >= 3:
                        full_sn = parts[2]
                
                # Load FRU data for summary
                endeavour_fru = self._load_fru_data(module_path, 'Endeavour')
                kenya_fru = self._load_fru_data(module_path, 'Kenya')
                onet_data = self._load_onet_data(module_path)
                
                all_modules_data[full_sn] = {
                    'endeavour_fru': endeavour_fru,
                    'kenya_fru': kenya_fru,
                    'onet': onet_data
                }
                
            except Exception as e:
                print(f"\n✗ Error analyzing calibration for module {module}: {e}")
        
        # Create summary plot with all modules
        if all_modules_data:
            self.create_fru_summary_all_modules(all_modules_data)
        
        # Create calibration plots from Raw log data
        for module in modules:
            try:
                self.create_calibration_log_plot(module)
            except Exception as e:
                print(f"\n✗ Error creating calibration log plot for module {module}: {e}")
        
        # Create calibration setpoints summary
        self.create_calibration_setpoints_summary(modules)
        
        print("\n" + "=" * 80)
        print("Calibration analysis completed!")
        print(f"Results saved to: analysis_results/calibration/")
        print("=" * 80)
    
    def state_validation(self, modules=None):
        """
        System state validation and monitoring method.
        
        This method handles validation of system states, laser safety and handshake power analysis.
        
        Args:
            modules (list): List of module serial numbers to analyze. If None, analyze all modules.
        """
        if modules is None:
            modules = self.modules
        
        print("=" * 80)
        print("State Validation - Laser Safety & Handshake Power Analysis")
        print("=" * 80)
        
        # Collect data for all modules
        modules_data = {}
        
        for module in modules:
            try:
                self.analyze_module_power(module)
                
                # Collect data for summary
                module_path = self.base_path / module
                config_files = list(module_path.glob("*config*.yaml"))
                full_sn = module  # fallback
                if config_files:
                    config_name = config_files[0].name
                    parts = config_name.split('_')
                    if len(parts) >= 3:
                        full_sn = parts[2]  # Y2534000165
                
                # Load data for summary
                endeavour_data = self._load_evt_data(module_path, 'Endeavour')
                kenya_data = self._load_evt_data(module_path, 'Kenya')
                
                if endeavour_data is not None or kenya_data is not None:
                    modules_data[full_sn] = {
                        'endeavour': endeavour_data,
                        'kenya': kenya_data
                    }
                
            except Exception as e:
                print(f"\n✗ Error analyzing module {module}: {e}")
        
        # Create summary plot
        if modules_data:
            self.create_summary_plot(modules_data)
        
        print("\n" + "=" * 80)
        print("State Validation completed!")
        print(f"Individual results: analysis_results/state_validation/")
        print(f"Summary plot: analysis_results/lasersafety_handshake_summary.png")
        print("=" * 80)
    
    def mission_mode(self, modules=None):
        """
        Mission mode operation analysis method.
        
        This method handles analysis of mission mode operations from Wavemeter tab,
        including power delivery and frequency error analysis.
        
        Args:
            modules (list): List of module serial numbers to analyze. If None, analyze all modules.
        """
        if modules is None:
            modules = self.modules
        
        print("\n" + "=" * 80)
        print("Mission Mode - Power & Frequency Error Analysis")
        print("=" * 80)
        
        # Collect data for summary plot
        summary_data = {}
        
        for module in modules:
            try:
                self.analyze_mission_mode_power(module)
                self.analyze_mission_mode_frequency(module)
                
                # Collect data for summary
                module_path = self.base_path / module
                config_files = list(module_path.glob("*config*.yaml"))
                full_sn = module  # fallback
                if config_files:
                    config_name = config_files[0].name
                    parts = config_name.split('_')
                    if len(parts) >= 3:
                        full_sn = parts[2]
                
                # Load wavemeter data
                endeavour_data = self._load_wavemeter_data(module_path, 'Endeavour')
                kenya_data = self._load_wavemeter_data(module_path, 'Kenya')
                
                if endeavour_data is not None or kenya_data is not None:
                    summary_data[full_sn] = {
                        'endeavour': endeavour_data,
                        'kenya': kenya_data
                    }
                
                # Create operating points plot for this module
                if endeavour_data is not None or kenya_data is not None:
                    self.plot_operating_points(module, full_sn, endeavour_data, kenya_data)
                    
            except Exception as e:
                print(f"\n✗ Error analyzing module {module} mission mode: {e}")
        
        # Create summary plot
        if summary_data:
            self.create_mission_mode_summary(summary_data)
            self.create_mission_mode_statistical_summary(summary_data)
        
        print("\n" + "=" * 80)
        print("Mission Mode analysis completed!")
        print(f"Results saved to: analysis_results/mission_mode/")
        print("=" * 80)
    
    def mission_mode_alabama(self, modules=None):
        """
        Generate Alabama plots where Kenya uses Endeavour's frequency requirements.
        """
        if modules is None:
            modules = self.modules
        
        print("\n" + "=" * 80)
        print("Mission Mode - Alabama Analysis (Kenya with Endeavour Freq Specs)")
        print("=" * 80)
        
        # Create alabama_plots folder
        alabama_path = self.mission_mode_path / 'alabama_plots'
        alabama_path.mkdir(exist_ok=True)
        
        # Save original specs
        original_endeavour_spec = None
        original_kenya_spec = None
        
        if self.specifications:
            # Save and update Endeavour spec
            if 'endeavour' in self.specifications:
                original_endeavour_spec = self.specifications['endeavour'].copy()
                self.specifications['endeavour']['wavelength_error'] = {
                    'value': 20.0,
                    'unit': 'GHz'
                }
                print("  ✓ Applied Alabama frequency requirements to Endeavour: ±20 GHz")
            
            # Save and update Kenya spec
            if 'kenya' in self.specifications:
                original_kenya_spec = self.specifications['kenya'].copy()
                self.specifications['kenya']['wavelength_error'] = {
                    'value': 20.0,
                    'unit': 'GHz'
                }
                print("  ✓ Applied Alabama frequency requirements to Kenya: ±20 GHz")
        
        # Temporarily change mission_mode_path to alabama_plots
        original_path = self.mission_mode_path
        self.mission_mode_path = alabama_path
        
        for module in modules:
            try:
                # Get full serial number
                module_path = self.base_path / module
                config_files = list(module_path.glob("*config*.yaml"))
                full_sn = module  # fallback
                if config_files:
                    config_name = config_files[0].name
                    parts = config_name.split('_')
                    if len(parts) >= 3:
                        full_sn = parts[2]
                
                # Load wavemeter data
                endeavour_data = self._load_wavemeter_data(module_path, 'Endeavour')
                kenya_data = self._load_wavemeter_data(module_path, 'Kenya')
                
                if endeavour_data is not None or kenya_data is not None:
                    # Generate power plot
                    self._plot_mission_mode_power(endeavour_data, kenya_data, full_sn)
                    
                    # Generate frequency error plot
                    if self.reference_grid:
                        self._plot_mission_mode_frequency_error(endeavour_data, kenya_data, full_sn)
                    
                    # Generate combined Alabama plot
                    if self.reference_grid:
                        self._plot_alabama_combined(endeavour_data, kenya_data, full_sn)
                    
            except Exception as e:
                print(f"\n✗ Error analyzing module {module} Alabama mode: {e}")
        
        # Restore original paths and specs
        self.mission_mode_path = original_path
        if self.specifications:
            if original_endeavour_spec:
                self.specifications['endeavour'] = original_endeavour_spec
            if original_kenya_spec:
                self.specifications['kenya'] = original_kenya_spec
        
        print("\n" + "=" * 80)
        print("Alabama analysis completed!")
        print(f"Results saved to: analysis_results/mission_mode/alabama_plots/")
        print("=" * 80)
    
    def _plot_alabama_combined(self, endeavour_data, kenya_data, full_sn):
        """Create combined Alabama plot with power vs time and frequency error violin plots"""
        print(f"  Generating combined Alabama plot for {full_sn}...")
        
        # Create figure with custom layout
        # Power plot: 1:1.75 aspect ratio (height:width)
        # Freq plots: 1.75:1 aspect ratio (height:width)
        fig = plt.figure(figsize=(12.5, 7.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1], hspace=0.3, wspace=0.3)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        
        # Subplot 1: Power vs Time for both Endeavour and Kenya
        ax = axes[0]
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Plot Endeavour power data
        if endeavour_data is not None:
            power_col = None
            for col in endeavour_data.columns:
                if 'mpd_pic' in col.lower():
                    power_col = col
                    break
            
            if power_col:
                for idx, row in endeavour_data.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[power_col]):
                        continue
                    
                    bank = int(row['bank'])
                    channel = int(row['channel'])
                    
                    power_values_uw = row[power_col]
                    if isinstance(power_values_uw, str):
                        power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                    elif not isinstance(power_values_uw, list):
                        power_values_uw = [power_values_uw]
                    
                    if power_values_uw:
                        power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                        x_values = np.arange(len(power_dbm))
                        
                        if bank == 0:
                            ax.plot(x_values, power_dbm, '-', label=f'B0-Ch{channel}', 
                                   linewidth=1.5, color=colors_b0[channel])
                        else:
                            ax.plot(x_values, power_dbm, '-', label=f'B1-Ch{channel}', 
                                   linewidth=1.5, color=colors_b1[channel])
        
        # Plot Kenya power data (with different line style to distinguish)
        if kenya_data is not None:
            power_col = None
            for col in kenya_data.columns:
                if 'mpd_pic' in col.lower():
                    power_col = col
                    break
            
            if power_col:
                for idx, row in kenya_data.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[power_col]):
                        continue
                    
                    bank = int(row['bank'])
                    channel = int(row['channel'])
                    
                    power_values_uw = row[power_col]
                    if isinstance(power_values_uw, str):
                        power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                    elif not isinstance(power_values_uw, list):
                        power_values_uw = [power_values_uw]
                    
                    if power_values_uw:
                        power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                        x_values = np.arange(len(power_dbm))
                        
                        # Use solid line for Kenya (Power Mode-2)
                        if bank == 0:
                            ax.plot(x_values, power_dbm, '-', linewidth=1.5, color=colors_b0[channel])
                        else:
                            ax.plot(x_values, power_dbm, '-', linewidth=1.5, color=colors_b1[channel])
        
        # Add specification limits
        # Power Mode-1: 10 to 12.3 dBm
        # Power Mode-2: 6.3 to 8.6 dBm
        ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Power Mode - 1 Min Spec: 10 dBm')
        ax.axhline(y=12.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Power Mode - 1 Max Spec: 12.3 dBm')
        ax.axhspan(10, 12.3, color='green', alpha=0.1)
        
        # Add annotation for Power Mode-1 on top of max spec line
        ax.text(1500, 12.5, 'Power Mode - 1', fontsize=10, ha='center', 
               color='blue', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', edgecolor='blue', alpha=0.8))
        
        ax.axhline(y=6.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Power Mode - 2 Min Spec: 6.3 dBm')
        ax.axhline(y=8.6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Power Mode - 2 Max Spec: 8.6 dBm')
        ax.axhspan(6.3, 8.6, color='green', alpha=0.1)
        
        # Add annotation for Power Mode-2 on top of max spec line
        ax.text(1500, 8.8, 'Power Mode - 2', fontsize=10, ha='center', 
               color='orange', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', edgecolor='orange', alpha=0.8))
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Power in fiber (dBm)', fontsize=10)
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 14)
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Power Mode-1 Frequency Error Box Plot
        ax = axes[1]
        self._plot_freq_violin_alabama(ax, endeavour_data, 'Power Mode - 1')
        
        # Subplot 3: Power Mode-2 Frequency Error Box Plot
        ax = axes[2]
        self._plot_freq_violin_alabama(ax, kenya_data, 'Power Mode - 2')
        
        fig.tight_layout()
        plot_filename = f'missionmode_alabama_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Combined Alabama plot saved: {plot_filename}")
    
    def _plot_freq_violin_alabama(self, ax, data, title):
        """Plot frequency error as box plot with scatter overlay for Alabama combined view"""
        if data is None:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=10, fontweight='bold')
            return
        
        # Speed of light constant
        c_speed_light = 299792.458
        
        # Color maps
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Collect frequency error data
        freq_error_data = {}
        
        wl_col = None
        for col in data.columns:
            if col.lower() == 'wavelength':
                wl_col = col
                break
        
        if wl_col:
            for idx, row in data.iterrows():
                if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[wl_col]):
                    continue
                
                bank = int(row['bank'])
                channel = int(row['channel'])
                set_name = 'set_a' if bank == 0 else 'set_b'
                
                wavelength_values_nm = row[wl_col]
                if isinstance(wavelength_values_nm, str):
                    wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
                elif not isinstance(wavelength_values_nm, list):
                    wavelength_values_nm = [wavelength_values_nm]
                
                if wavelength_values_nm and self.reference_grid:
                    grid_num = channel + 1
                    grid_key = f'grid_{grid_num}'
                    
                    if set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                        ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                        
                        freq_errors = []
                        for wl in wavelength_values_nm:
                            measured_freq_thz = c_speed_light / wl
                            freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                            freq_errors.append(freq_error_ghz)
                        
                        # Filter outliers
                        freq_errors_array = np.array(freq_errors)
                        q1 = np.percentile(freq_errors_array, 25)
                        q3 = np.percentile(freq_errors_array, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        valid_freq_errors = freq_errors_array[(freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound)]
                        
                        if len(valid_freq_errors) > 0:
                            freq_error_data[(bank, channel)] = valid_freq_errors.tolist()
        
        if freq_error_data:
            # Prepare data for box plot
            positions = []
            data_list = []
            colors_list = []
            labels_list = []
            
            # Sort by bank and channel (reverse order for bottom-to-top display)
            sorted_keys = sorted(freq_error_data.keys(), reverse=True)
            
            for i, (bank, channel) in enumerate(sorted_keys):
                positions.append(i)
                data_list.append(freq_error_data[(bank, channel)])
                
                if bank == 0:
                    colors_list.append(colors_b0[channel])
                    labels_list.append(f'B0-Ch{channel}')
                else:
                    colors_list.append(colors_b1[channel])
                    labels_list.append(f'B1-Ch{channel}')
            
            # Create box plot (horizontal orientation)
            bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(linewidth=1.5),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
            
            # Color each box
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Overlay scatter points for all data
            for i, data in enumerate(data_list):
                # Add jitter to y-position for better visibility
                y_positions = np.random.normal(positions[i], 0.08, size=len(data))
                ax.scatter(data, y_positions, alpha=0.4, s=20, color=colors_list[i], marker='o')
            
            # Add specification limit (±20 GHz for Alabama)
            ax.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='±20 GHz')
            ax.axvline(x=-20, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvspan(-20, 20, color='green', alpha=0.1, label='Spec Range')
            
            # Add zero line
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            
            ax.set_yticks(positions)
            ax.set_yticklabels(labels_list)
            ax.set_xlabel('Frequency Error (GHz)', fontsize=10)
            ax.set_ylabel('Bank-Channel', fontsize=10)
            
            # Add dashed line between B0 and B1
            if len(positions) > 8:
                ax.axhline(y=7.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlim(-40, 40)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def analyze_module_power(self, module_sn='165'):
        """
        Analyze both Endeavour and Kenya specifications for a module.
        Creates a single plot with 2 subplots (Endeavour and Kenya), each showing Bank0 and Bank1 curves.
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        self.current_module = module_sn
        module_path = self.base_path / module_sn
        
        # Get full serial number from config file
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn  # fallback
        if config_files:
            config_name = config_files[0].name
            parts = config_name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]  # Y2534000165
        
        print(f"\n{'='*60}")
        print(f"Analyzing Module {full_sn}")
        print(f"{'='*60}")
        
        # Load Endeavour data
        endeavour_data = self._load_evt_data(module_path, 'Endeavour')
        
        # Load Kenya data
        kenya_data = self._load_evt_data(module_path, 'Kenya')
        
        # Plot combined results
        if endeavour_data is not None or kenya_data is not None:
            self._plot_combined_power_analysis(endeavour_data, kenya_data, full_sn)
        else:
            print(f"No data available for module {full_sn}")
    
    def _load_evt_data(self, module_path, spec_type):
        """Load EVT data for specified specification type (Endeavour or Kenya)"""
        # Look for EVT file
        if spec_type == 'Endeavour':
            evt_files = (list(module_path.glob("*Endeavour*EVT*.xlsx")) + 
                        list(module_path.glob("*Endevour*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Endeavour*.xlsx")) +
                        list(module_path.glob("*EVT*Endevour*.xlsx")) +
                        list(module_path.glob("*EVT*result*Endevour*.xlsx")))
        else:  # Kenya
            evt_files = (list(module_path.glob("*Kenya*EVT*.xlsx")) + 
                        list(module_path.glob("*kenya*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Kenya*.xlsx")) +
                        list(module_path.glob("*EVT*kenya*.xlsx")))
        
        if not evt_files:
            print(f"  No {spec_type} EVT file found")
            return None
        
        evt_file = evt_files[0]
        print(f"  Loading {spec_type}: {evt_file.name}")
        
        try:
            excel_file = pd.ExcelFile(evt_file)
            
            # Find validate states sheet
            validate_sheet = None
            for sheet in excel_file.sheet_names:
                sheet_lower = sheet.lower()
                if (('validate' in sheet_lower and 'state' in sheet_lower) or
                    ('vaidate' in sheet_lower and 'state' in sheet_lower) or
                    ('valdate' in sheet_lower and 'state' in sheet_lower) or
                    ('validate' in sheet_lower and 'power' in sheet_lower)):
                    validate_sheet = sheet
                    break
            
            if validate_sheet is None:
                print(f"  No validate states sheet found in {spec_type}")
                return None
            
            df = pd.read_excel(evt_file, sheet_name=validate_sheet)
            
            # Check for required columns
            if 'Total_Power_HandshakePower' not in df.columns or 'Total_Power_LaserSafetyPower' not in df.columns:
                print(f"  Required power columns not found in {spec_type}")
                return None
            
            return df
            
        except Exception as e:
            print(f"  Error loading {spec_type} data: {e}")
            return None
    
    def _plot_combined_power_analysis(self, endeavour_data, kenya_data, full_sn):
        """Plot combined Endeavour and Kenya power analysis in 2 subplots"""
        print(f"\n  Generating combined power analysis plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Endeavour
        if endeavour_data is not None:
            self._plot_spec_data(axes[0], endeavour_data, 'Endeavour', full_sn)
        else:
            axes[0].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0].set_title(f'Endeavour - Module {full_sn}')
        
        # Plot Kenya
        if kenya_data is not None:
            self._plot_spec_data(axes[1], kenya_data, 'Kenya', full_sn)
        else:
            axes[1].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1].set_title(f'Kenya - Module {full_sn}')
        
        plt.tight_layout()
        plot_filename = f'lasersafety_handshake_power_{full_sn}.png'
        plt.savefig(self.state_validation_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
    
    def _plot_spec_data(self, ax, df, spec_name, full_sn):
        """Plot power data for one specification (Bank 0 and Bank 1 curves)"""
        # Extract data for Bank 0 and Bank 1
        bank0_data = df[df['bank'] == 0].iloc[0]
        bank1_data = df[df['bank'] == 1].iloc[0]
        
        # Process Bank 0
        bank0_handshake = bank0_data['Total_Power_HandshakePower']
        if isinstance(bank0_handshake, str):
            bank0_handshake_values = [float(x.strip()) for x in bank0_handshake.strip('[]').split(',')]
        else:
            bank0_handshake_values = bank0_handshake
        
        bank0_safety = bank0_data['Total_Power_LaserSafetyPower']
        if isinstance(bank0_safety, str):
            bank0_safety = float(bank0_safety)
        
        # Process Bank 1
        bank1_handshake = bank1_data['Total_Power_HandshakePower']
        if isinstance(bank1_handshake, str):
            bank1_handshake_values = [float(x.strip()) for x in bank1_handshake.strip('[]').split(',')]
        else:
            bank1_handshake_values = bank1_handshake
        
        bank1_safety = bank1_data['Total_Power_LaserSafetyPower']
        if isinstance(bank1_safety, str):
            bank1_safety = float(bank1_safety)
        
        # Scale time to 10 seconds
        time_scale_bank0 = 10.0 / len(bank0_handshake_values)
        time_scale_bank1 = 10.0 / len(bank1_handshake_values)
        
        time_bank0 = np.arange(len(bank0_handshake_values)) * time_scale_bank0
        time_bank1 = np.arange(len(bank1_handshake_values)) * time_scale_bank1
        
        # Plot Bank 0
        ax.plot(time_bank0, bank0_handshake_values, 'o-', 
                label=f'Bank 0 Handshake', linewidth=2, markersize=3, color='blue')
        
        # Plot Bank 1
        ax.plot(time_bank1, bank1_handshake_values, 's-', 
                label=f'Bank 1 Handshake', linewidth=2, markersize=3, color='orange')
        
        # Get peak values
        bank0_peak = max(bank0_handshake_values)
        bank1_peak = max(bank1_handshake_values)
        
        # Set labels and title
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Power in fiber (dBm)')
        ax.set_title(f'{spec_name} - Tile SN: {full_sn}\nHandshake Peak: B0={bank0_peak:.1f}dBm, B1={bank1_peak:.1f}dBm | Safety: B0={bank0_safety:.1f}dBm, B1={bank1_safety:.1f}dBm')
        ax.set_ylim(-10, 20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def create_summary_plot(self, modules_data):
        """
        Create a summary plot showing handshake and safety power for all modules.
        
        Args:
            modules_data: Dictionary with module data
                {full_sn: {'endeavour': df, 'kenya': df}}
        """
        print(f"\n{'='*60}")
        print("Creating Summary Plot")
        print(f"{'='*60}")
        
        # Extract data for plotting
        endeavour_data = {'sn': [], 'bank0_handshake': [], 'bank1_handshake': [], 
                         'bank0_safety': [], 'bank1_safety': []}
        kenya_data = {'sn': [], 'bank0_handshake': [], 'bank1_handshake': [], 
                     'bank0_safety': [], 'bank1_safety': []}
        
        for full_sn, data in sorted(modules_data.items()):
            # Process Endeavour data
            if data['endeavour'] is not None:
                df = data['endeavour']
                bank0 = df[df['bank'] == 0].iloc[0]
                bank1 = df[df['bank'] == 1].iloc[0]
                
                # Process Bank 0
                bank0_handshake = bank0['Total_Power_HandshakePower']
                if isinstance(bank0_handshake, str):
                    bank0_handshake_values = [float(x.strip()) for x in bank0_handshake.strip('[]').split(',')]
                else:
                    bank0_handshake_values = bank0_handshake
                bank0_safety = float(bank0['Total_Power_LaserSafetyPower']) if not isinstance(bank0['Total_Power_LaserSafetyPower'], float) else bank0['Total_Power_LaserSafetyPower']
                
                # Process Bank 1
                bank1_handshake = bank1['Total_Power_HandshakePower']
                if isinstance(bank1_handshake, str):
                    bank1_handshake_values = [float(x.strip()) for x in bank1_handshake.strip('[]').split(',')]
                else:
                    bank1_handshake_values = bank1_handshake
                bank1_safety = float(bank1['Total_Power_LaserSafetyPower']) if not isinstance(bank1['Total_Power_LaserSafetyPower'], float) else bank1['Total_Power_LaserSafetyPower']
                
                endeavour_data['sn'].append(full_sn)
                endeavour_data['bank0_handshake'].append(max(bank0_handshake_values))
                endeavour_data['bank1_handshake'].append(max(bank1_handshake_values))
                endeavour_data['bank0_safety'].append(bank0_safety)
                endeavour_data['bank1_safety'].append(bank1_safety)
            
            # Process Kenya data
            if data['kenya'] is not None:
                df = data['kenya']
                bank0 = df[df['bank'] == 0].iloc[0]
                bank1 = df[df['bank'] == 1].iloc[0]
                
                # Process Bank 0
                bank0_handshake = bank0['Total_Power_HandshakePower']
                if isinstance(bank0_handshake, str):
                    bank0_handshake_values = [float(x.strip()) for x in bank0_handshake.strip('[]').split(',')]
                else:
                    bank0_handshake_values = bank0_handshake
                bank0_safety = float(bank0['Total_Power_LaserSafetyPower']) if not isinstance(bank0['Total_Power_LaserSafetyPower'], float) else bank0['Total_Power_LaserSafetyPower']
                
                # Process Bank 1
                bank1_handshake = bank1['Total_Power_HandshakePower']
                if isinstance(bank1_handshake, str):
                    bank1_handshake_values = [float(x.strip()) for x in bank1_handshake.strip('[]').split(',')]
                else:
                    bank1_handshake_values = bank1_handshake
                bank1_safety = float(bank1['Total_Power_LaserSafetyPower']) if not isinstance(bank1['Total_Power_LaserSafetyPower'], float) else bank1['Total_Power_LaserSafetyPower']
                
                kenya_data['sn'].append(full_sn)
                kenya_data['bank0_handshake'].append(max(bank0_handshake_values))
                kenya_data['bank1_handshake'].append(max(bank1_handshake_values))
                kenya_data['bank0_safety'].append(bank0_safety)
                kenya_data['bank1_safety'].append(bank1_safety)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Endeavour
        ax1 = axes[0]
        x = np.arange(len(endeavour_data['sn']))
        
        ax1.scatter(x, endeavour_data['bank0_handshake'], s=100, label='Bank 0 Handshake', 
                   color='blue', marker='o', alpha=0.8)
        ax1.scatter(x, endeavour_data['bank1_handshake'], s=100, label='Bank 1 Handshake', 
                   color='orange', marker='s', alpha=0.8)
        ax1.scatter(x, endeavour_data['bank0_safety'], s=100, label='Bank 0 Safety', 
                   color='lightblue', marker='^', alpha=0.8)
        ax1.scatter(x, endeavour_data['bank1_safety'], s=100, label='Bank 1 Safety', 
                   color='lightsalmon', marker='v', alpha=0.8)
        
        ax1.set_xlabel('Serial Number')
        ax1.set_ylabel('Power in fiber (dBm)')
        ax1.set_title('Endeavour - Handshake & Laser Safety Power')
        ax1.set_xticks(x)
        ax1.set_xticklabels(endeavour_data['sn'], rotation=45, ha='right')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot Kenya
        ax2 = axes[1]
        x = np.arange(len(kenya_data['sn']))
        
        ax2.scatter(x, kenya_data['bank0_handshake'], s=100, label='Bank 0 Handshake', 
                   color='blue', marker='o', alpha=0.8)
        ax2.scatter(x, kenya_data['bank1_handshake'], s=100, label='Bank 1 Handshake', 
                   color='orange', marker='s', alpha=0.8)
        ax2.scatter(x, kenya_data['bank0_safety'], s=100, label='Bank 0 Safety', 
                   color='lightblue', marker='^', alpha=0.8)
        ax2.scatter(x, kenya_data['bank1_safety'], s=100, label='Bank 1 Safety', 
                   color='lightsalmon', marker='v', alpha=0.8)
        
        ax2.set_xlabel('Serial Number')
        ax2.set_ylabel('Power in fiber (dBm)')
        ax2.set_title('Kenya - Handshake & Laser Safety Power')
        ax2.set_xticks(x)
        ax2.set_xticklabels(kenya_data['sn'], rotation=45, ha='right')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to analysis_results root
        summary_path = self.base_path / "analysis_results"
        plot_filename = 'lasersafety_handshake_summary.png'
        plt.savefig(summary_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Summary plot saved: {plot_filename}")
        print(f"  Location: analysis_results/{plot_filename}")
    
    def analyze_endeavour_evt(self, module_sn='165'):
        """
        Analyze Endeavour EVT.xlsx file for specified module.
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        self.current_module = module_sn
        module_path = self.base_path / module_sn
        
        # Get full serial number from config file
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn  # fallback
        if config_files:
            config_name = config_files[0].name
            print(f"Config file found: {config_name}")
            # Extract SN from filename like "3_0_Y2534000165_config_file_Endeavour_14mW_1d.yaml"
            parts = config_name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]  # Y2534000165
                print(f"Extracted serial number: {full_sn}")
        print(f"Using serial number: {full_sn}")
        
        # Look for Endeavour EVT.xlsx file (handle various naming patterns)
        evt_files = (list(module_path.glob("*Endeavour*EVT*.xlsx")) + 
                    list(module_path.glob("*Endevour*EVT*.xlsx")) +
                    list(module_path.glob("*EVT*Endeavour*.xlsx")) +
                    list(module_path.glob("*EVT*Endevour*.xlsx")) +
                    list(module_path.glob("*EVT*result*Endevour*.xlsx")))
        if not evt_files:
            print(f"No Endeavour/Endevour EVT.xlsx file found for module {module_sn}")
            return None
            
        evt_file = evt_files[0]
        print(f"Analyzing Endeavour EVT file: {evt_file.name}")
        
        try:
            # Read the Excel file and get sheet names
            excel_file = pd.ExcelFile(evt_file)
            print(f"Available sheets: {excel_file.sheet_names}")
            
            # Look for Validate states tab
            validate_sheet = None
            # Look for various validate sheet name patterns
            for sheet in excel_file.sheet_names:
                sheet_lower = sheet.lower()
                if (('validate' in sheet_lower and 'state' in sheet_lower) or
                    ('vaidate' in sheet_lower and 'state' in sheet_lower) or  # Handle typo
                    ('valdate' in sheet_lower and 'state' in sheet_lower) or  # Handle another typo
                    ('validate' in sheet_lower and 'power' in sheet_lower)):
                    validate_sheet = sheet
                    break
            
            if validate_sheet is None:
                print("No 'Validate states' tab found. Available sheets:")
                for sheet in excel_file.sheet_names:
                    print(f"  - {sheet}")
                return None
                
            print(f"Found Validate states sheet: {validate_sheet}")
            
            # Read the Validate states sheet
            df = pd.read_excel(evt_file, sheet_name=validate_sheet)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Look for power columns
            handshake_cols = [col for col in df.columns if 'handshake' in col.lower() and 'power' in col.lower()]
            safety_cols = [col for col in df.columns if 'safety' in col.lower() and 'power' in col.lower()]
            print(f"Handshake power columns found: {handshake_cols}")
            print(f"Laser safety power columns found: {safety_cols}")
            
            if handshake_cols and safety_cols:
                self.plot_power_analysis(df, handshake_cols, safety_cols, full_sn)
            else:
                print("Required power columns not found")
                print("Available columns:")
                for col in df.columns:
                    print(f"  - {col}")
                    
        except Exception as e:
            print(f"Error analyzing Excel file: {e}")
            return None
    
    def plot_power_analysis(self, df, handshake_cols, safety_cols, full_sn):
        """
        Plot power analysis with HandshakePower and LaserSafetyPower for each channel by bank.
        
        Args:
            df (DataFrame): Data from Validate states sheet
            handshake_cols (list): List of handshake power columns
            safety_cols (list): List of laser safety power columns
            full_sn (str): Full serial number (e.g., 'Y2534000165')
        """
        print(f"\nPlotting power analysis for module {full_sn}")
        
        # Extract handshake power data - it's stored as arrays in each cell
        handshake_data = []
        safety_data = []
        
        for idx, row in df.iterrows():
            bank = row['bank']
            channel = row['channel']
            
            # Process handshake power array
            handshake_array = row['Total_Power_HandshakePower']
            if isinstance(handshake_array, str):
                handshake_values = [float(x.strip()) for x in handshake_array.strip('[]').split(',')]
            else:
                handshake_values = handshake_array
            
            # Process laser safety power (single value, not array)
            safety_power = row['Total_Power_LaserSafetyPower']
            if isinstance(safety_power, str):
                safety_power = float(safety_power)
            
            # Create data for each measurement point
            for i, h_power in enumerate(handshake_values):
                handshake_data.append({
                    'bank': bank,
                    'channel': channel,
                    'measurement': i,
                    'power_mw': h_power
                })
                # Safety power is constant for each channel
                safety_data.append({
                    'bank': bank,
                    'channel': channel,
                    'measurement': i,
                    'power_mw': safety_power
                })
        
        # Convert to DataFrames
        handshake_df = pd.DataFrame(handshake_data)
        safety_df = pd.DataFrame(safety_data)
        print(f"Extracted {len(handshake_df)} handshake power measurements")
        print(f"Extracted {len(safety_df)} laser safety power measurements")
        
        # Data is already in dBm, no conversion needed
        handshake_df['power_dbm'] = handshake_df['power_mw']
        safety_df['power_dbm'] = safety_df['power_mw']
        
        # Scale time to 10 seconds
        max_measurements = max(len(handshake_df[handshake_df['bank'] == 0]['measurement'].unique()),
                              len(handshake_df[handshake_df['bank'] == 1]['measurement'].unique()))
        time_scale = 10.0 / max_measurements  # Scale to 10 seconds
        
        # Create 2 subplots - one for each bank
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot for Bank 0
        ax1 = axes[0]
        bank0_handshake = handshake_df[handshake_df['bank'] == 0]
        bank0_safety = safety_df[safety_df['bank'] == 0]
        
        # Get peak handshake power and single safety power for title
        bank0_handshake_peak = f"{bank0_handshake['power_dbm'].max():.1f}dBm"
        bank0_safety_value = f"{bank0_safety['power_dbm'].iloc[0]:.1f}dBm"
        
        for channel in sorted(bank0_handshake['channel'].unique()):
            # Handshake power (peak of curve)
            ch_handshake = bank0_handshake[bank0_handshake['channel'] == channel]
            time_points = ch_handshake['measurement'] * time_scale
            handshake_power = ch_handshake['power_dbm']
            
            # Laser safety power (constant value from column)
            ch_safety = bank0_safety[bank0_safety['channel'] == channel]
            safety_power = ch_safety['power_dbm'].iloc[0]  # Take first value as it's constant
            
            # Plot handshake power as line
            ax1.plot(time_points, handshake_power, 'o-', 
                    label=f'Ch{channel} Handshake', linewidth=2, markersize=3)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Power in fiber (dBm)')
        ax1.set_title(f'Bank 0 - Handshake Peak: {bank0_handshake_peak}, Safety: {bank0_safety_value}')
        ax1.set_ylim(-10, 20)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot for Bank 1
        ax2 = axes[1]
        bank1_handshake = handshake_df[handshake_df['bank'] == 1]
        bank1_safety = safety_df[safety_df['bank'] == 1]
        
        # Get peak handshake power and single safety power for title
        bank1_handshake_peak = f"{bank1_handshake['power_dbm'].max():.1f}dBm"
        bank1_safety_value = f"{bank1_safety['power_dbm'].iloc[0]:.1f}dBm"
        
        for channel in sorted(bank1_handshake['channel'].unique()):
            # Handshake power (peak of curve)
            ch_handshake = bank1_handshake[bank1_handshake['channel'] == channel]
            time_points = ch_handshake['measurement'] * time_scale
            handshake_power = ch_handshake['power_dbm']
            
            # Laser safety power (constant value from column)
            ch_safety = bank1_safety[bank1_safety['channel'] == channel]
            safety_power = ch_safety['power_dbm'].iloc[0]  # Take first value as it's constant
            
            # Plot handshake power as line
            ax2.plot(time_points, handshake_power, 'o-', 
                    label=f'Ch{channel} Handshake', linewidth=2, markersize=3)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power in fiber (dBm)')
        ax2.set_title(f'Bank 1 - Handshake Peak: {bank1_handshake_peak}, Safety: {bank1_safety_value}')
        ax2.set_ylim(-10, 20)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f'lasersafety_handshake_endeavour_power_{full_sn}.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Power analysis plot saved: {plot_filename}")
        
        # Print summary
        print(f"\nPower Analysis Summary for Module {full_sn}:")
        print(f"Handshake power range: {handshake_df['power_dbm'].min():.2f} to {handshake_df['power_dbm'].max():.2f} dBm")
        print(f"Laser safety power range: {safety_df['power_dbm'].min():.2f} to {safety_df['power_dbm'].max():.2f} dBm")
        
        # Bank comparison
        bank0_handshake_mean = bank0_handshake['power_dbm'].mean()
        bank1_handshake_mean = bank1_handshake['power_dbm'].mean()
        print(f"Bank 0 handshake mean: {bank0_handshake_mean:.2f} dBm")
        print(f"Bank 1 handshake mean: {bank1_handshake_mean:.2f} dBm")
        print(f"Bank difference: {abs(bank0_handshake_mean - bank1_handshake_mean):.2f} dBm")

    def analyze_kenya_evt(self, module_sn='165'):
        """
        Analyze Kenya EVT.xlsx file for handshake power
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        self.current_module = module_sn
        module_path = self.base_path / module_sn
        
        # Get full serial number from config file
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn  # fallback
        if config_files:
            config_name = config_files[0].name
            print(f"Config file found: {config_name}")
            # Extract SN from filename like "3_0_Y2534000165_config_file_Kenya_6p4mW_1d.yaml"
            parts = config_name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]  # Y2534000165
                print(f"Extracted serial number: {full_sn}")
        print(f"Using serial number: {full_sn}")
        
        # Look for Kenya EVT.xlsx file (handle various naming patterns)
        evt_files = (list(module_path.glob("*Kenya*EVT*.xlsx")) + 
                    list(module_path.glob("*kenya*EVT*.xlsx")) +
                    list(module_path.glob("*EVT*Kenya*.xlsx")) +
                    list(module_path.glob("*EVT*kenya*.xlsx")))
        if not evt_files:
            print(f"No Kenya EVT.xlsx file found for module {module_sn}")
            return None
            
        evt_file = evt_files[0]
        print(f"Analyzing Kenya EVT file: {evt_file.name}")
        
        try:
            # Read the Excel file and get sheet names
            excel_file = pd.ExcelFile(evt_file)
            print(f"Available sheets: {excel_file.sheet_names}")
            
            # Look for Validate states tab
            validate_sheet = None
            # Look for various validate sheet name patterns
            for sheet in excel_file.sheet_names:
                sheet_lower = sheet.lower()
                if (('validate' in sheet_lower and 'state' in sheet_lower) or
                    ('vaidate' in sheet_lower and 'state' in sheet_lower) or  # Handle typo
                    ('valdate' in sheet_lower and 'state' in sheet_lower) or  # Handle another typo
                    ('validate' in sheet_lower and 'power' in sheet_lower)):
                    validate_sheet = sheet
                    break
            
            if validate_sheet is None:
                print("No 'Validate states' tab found. Available sheets:")
                for sheet in excel_file.sheet_names:
                    print(f"  - {sheet}")
                return None
                
            print(f"Found Validate states sheet: {validate_sheet}")
            
            # Read the Validate states sheet
            df = pd.read_excel(evt_file, sheet_name=validate_sheet)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Find handshake and safety power columns
            handshake_cols = [col for col in df.columns if 'handshake' in col.lower() and 'power' in col.lower()]
            safety_cols = [col for col in df.columns if 'safety' in col.lower() and 'power' in col.lower()]
            print(f"Handshake power columns found: {handshake_cols}")
            print(f"Laser safety power columns found: {safety_cols}")
            
            if handshake_cols and safety_cols:
                self.plot_kenya_power_analysis(df, handshake_cols, safety_cols, full_sn)
            else:
                print("Required power columns not found")
                print("Available columns:")
                for col in df.columns:
                    print(f"  - {col}")
                    
        except Exception as e:
            print(f"Error analyzing Excel file: {e}")
            return None

    def plot_kenya_power_analysis(self, df, handshake_cols, safety_cols, full_sn):
        """
        Plot Kenya handshake power analysis
        
        Args:
            df: DataFrame with power data
            handshake_cols: List of handshake power column names
            safety_cols: List of safety power column names
            full_sn: Full serial number
        """
        print(f"\nPlotting Kenya power analysis for module {full_sn}")
        
        # Extract handshake power data - it's stored as arrays in each cell
        handshake_data = []
        safety_data = []
        
        for idx, row in df.iterrows():
            bank = row['bank']
            channel = row['channel']
            
            # Process handshake power array
            handshake_array = row['Total_Power_HandshakePower']
            if isinstance(handshake_array, str):
                handshake_values = [float(x.strip()) for x in handshake_array.strip('[]').split(',')]
            else:
                handshake_values = handshake_array
            
            # Process laser safety power (single value, not array)
            safety_power = row['Total_Power_LaserSafetyPower']
            if isinstance(safety_power, str):
                safety_power = float(safety_power)
            
            # Create data for each measurement point
            for i, h_power in enumerate(handshake_values):
                handshake_data.append({
                    'bank': bank,
                    'channel': channel,
                    'measurement': i,
                    'power_mw': h_power
                })
                # Safety power is constant for each channel
                safety_data.append({
                    'bank': bank,
                    'channel': channel,
                    'measurement': i,
                    'power_mw': safety_power
                })
        
        # Convert to DataFrames
        handshake_df = pd.DataFrame(handshake_data)
        safety_df = pd.DataFrame(safety_data)
        print(f"Extracted {len(handshake_df)} handshake power measurements")
        print(f"Extracted {len(safety_df)} laser safety power measurements")
        
        # Data is already in dBm, no conversion needed
        handshake_df['power_dbm'] = handshake_df['power_mw']
        safety_df['power_dbm'] = safety_df['power_mw']
        
        # Scale time to 10 seconds
        max_measurements = max(len(handshake_df[handshake_df['bank'] == 0]['measurement'].unique()),
                              len(handshake_df[handshake_df['bank'] == 1]['measurement'].unique()))
        time_scale = 10.0 / max_measurements  # Scale to 10 seconds
        
        # Create 2 subplots - one for each bank
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot for Bank 0
        ax1 = axes[0]
        bank0_handshake = handshake_df[handshake_df['bank'] == 0]
        bank0_safety = safety_df[safety_df['bank'] == 0]
        
        # Get peak handshake power and single safety power for title
        bank0_handshake_peak = f"{bank0_handshake['power_dbm'].max():.1f}dBm"
        bank0_safety_value = f"{bank0_safety['power_dbm'].iloc[0]:.1f}dBm"
        
        for channel in sorted(bank0_handshake['channel'].unique()):
            # Handshake power (peak of curve)
            ch_handshake = bank0_handshake[bank0_handshake['channel'] == channel]
            time_points = ch_handshake['measurement'] * time_scale
            handshake_power = ch_handshake['power_dbm']
            
            # Plot handshake power as line
            ax1.plot(time_points, handshake_power, 'o-', 
                    label=f'Ch{channel} Handshake', linewidth=2, markersize=3)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Power in fiber (dBm)')
        ax1.set_title(f'Bank 0 - Handshake Peak: {bank0_handshake_peak}, Safety: {bank0_safety_value}')
        ax1.set_ylim(-10, 20)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot for Bank 1
        ax2 = axes[1]
        bank1_handshake = handshake_df[handshake_df['bank'] == 1]
        bank1_safety = safety_df[safety_df['bank'] == 1]
        
        # Get peak handshake power and single safety power for title
        bank1_handshake_peak = f"{bank1_handshake['power_dbm'].max():.1f}dBm"
        bank1_safety_value = f"{bank1_safety['power_dbm'].iloc[0]:.1f}dBm"
        
        for channel in sorted(bank1_handshake['channel'].unique()):
            # Handshake power (peak of curve)
            ch_handshake = bank1_handshake[bank1_handshake['channel'] == channel]
            time_points = ch_handshake['measurement'] * time_scale
            handshake_power = ch_handshake['power_dbm']
            
            # Plot handshake power as line
            ax2.plot(time_points, handshake_power, 'o-', 
                    label=f'Ch{channel} Handshake', linewidth=2, markersize=3)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power in fiber (dBm)')
        ax2.set_title(f'Bank 1 - Handshake Peak: {bank1_handshake_peak}, Safety: {bank1_safety_value}')
        ax2.set_ylim(-10, 20)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f'lasersafety_handshake_kenya_power_{full_sn}.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Kenya power analysis plot saved: {plot_filename}")
        
        # Print summary
        print(f"\nKenya Power Analysis Summary for Module {full_sn}:")
        print(f"Handshake power range: {handshake_df['power_dbm'].min():.2f} to {handshake_df['power_dbm'].max():.2f} dBm")
        print(f"Laser safety power range: {safety_df['power_dbm'].min():.2f} to {safety_df['power_dbm'].max():.2f} dBm")
        
        bank0_handshake_mean = bank0_handshake['power_dbm'].mean()
        bank1_handshake_mean = bank1_handshake['power_dbm'].mean()
        print(f"Bank 0 handshake mean: {bank0_handshake_mean:.2f} dBm")
        print(f"Bank 1 handshake mean: {bank1_handshake_mean:.2f} dBm")
        print(f"Bank difference: {abs(bank0_handshake_mean - bank1_handshake_mean):.2f} dBm")
    
    def analyze_mission_mode_power(self, module_sn='165'):
        """
        Analyze mission mode power from Wavemeter tab (MPD_PIC columns)
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        module_path = self.base_path / module_sn
        
        # Get full serial number
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn
        if config_files:
            parts = config_files[0].name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]
        
        print(f"\n{'='*60}")
        print(f"Mission Mode Power Analysis - Module {full_sn}")
        print(f"{'='*60}")
        
        # Load Endeavour data
        endeavour_data = self._load_wavemeter_data(module_path, 'Endeavour')
        
        # Load Kenya data
        kenya_data = self._load_wavemeter_data(module_path, 'Kenya')
        
        if endeavour_data is not None or kenya_data is not None:
            self._plot_mission_mode_power(endeavour_data, kenya_data, full_sn)
        else:
            print(f"  No wavemeter data available for module {full_sn}")
    
    def analyze_mission_mode_frequency(self, module_sn='165'):
        """
        Analyze frequency error from Wavemeter tab compared to reference grid
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        module_path = self.base_path / module_sn
        
        # Get full serial number
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn
        if config_files:
            parts = config_files[0].name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]
        
        print(f"\n{'='*60}")
        print(f"Mission Mode Frequency Error Analysis - Module {full_sn}")
        print(f"{'='*60}")
        
        # Load Endeavour data
        endeavour_data = self._load_wavemeter_data(module_path, 'Endeavour')
        
        # Load Kenya data
        kenya_data = self._load_wavemeter_data(module_path, 'Kenya')
        
        if endeavour_data is not None or kenya_data is not None:
            self._plot_mission_mode_frequency_error(endeavour_data, kenya_data, full_sn)
        else:
            print(f"  No wavemeter data available for module {full_sn}")
    
    def _load_wavemeter_data(self, module_path, spec_type):
        """Load Wavemeter data for specified specification type"""
        # Look for EVT file
        if spec_type == 'Endeavour':
            evt_files = (list(module_path.glob("*Endeavour*EVT*.xlsx")) + 
                        list(module_path.glob("*Endevour*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Endeavour*.xlsx")) +
                        list(module_path.glob("*EVT*Endevour*.xlsx")) +
                        list(module_path.glob("*EVT*result*Endevour*.xlsx")))
        else:  # Kenya
            evt_files = (list(module_path.glob("*Kenya*EVT*.xlsx")) + 
                        list(module_path.glob("*kenya*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Kenya*.xlsx")) +
                        list(module_path.glob("*EVT*kenya*.xlsx")))
        
        if not evt_files:
            return None
        
        evt_file = evt_files[0]
        
        try:
            excel_file = pd.ExcelFile(evt_file)
            
            # Find Wavemeter sheet
            wavemeter_sheet = None
            for sheet in excel_file.sheet_names:
                if 'wavemeter' in sheet.lower():
                    wavemeter_sheet = sheet
                    break
            
            if wavemeter_sheet is None:
                return None
            
            df = pd.read_excel(evt_file, sheet_name=wavemeter_sheet)
            return df
            
        except Exception as e:
            print(f"  Error loading {spec_type} wavemeter data: {e}")
            return None
    
    def _plot_mission_mode_power(self, endeavour_data, kenya_data, full_sn):
        """Plot mission mode power from MPD_PIC columns"""
        print(f"  Generating mission mode power plot...")
        
        # Create figure with equal subplot widths
        # Both time series and distribution: 1.5:1 aspect ratio (height:width)
        fig = plt.figure(figsize=(16, 24))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
        
        # Plot Endeavour - time series and distribution
        if endeavour_data is not None:
            power_data = self._plot_power_spec(axes[0, 0], endeavour_data, 'Endeavour', full_sn)
            self._plot_power_distribution(axes[0, 1], power_data, 'Endeavour', full_sn)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0, 0].set_title(f'Endeavour - Module {full_sn}')
            axes[0, 1].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0, 1].set_title(f'Endeavour Distribution - Module {full_sn}')
        
        # Plot Kenya - time series and distribution
        if kenya_data is not None:
            power_data = self._plot_power_spec(axes[1, 0], kenya_data, 'Kenya', full_sn)
            self._plot_power_distribution(axes[1, 1], power_data, 'Kenya', full_sn)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title(f'Kenya - Module {full_sn}')
            axes[1, 1].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title(f'Kenya Distribution - Module {full_sn}')
        
        plt.tight_layout()
        plot_filename = f'missionmode_power_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
    
    def _plot_power_spec(self, ax, df, spec_name, full_sn):
        """Plot power for one specification from mpd_pic column
        Returns: dict with {(bank, channel): [power_dbm_values]} for distribution plotting
        """
        # Find mpd_pic column
        power_col = None
        for col in df.columns:
            if 'mpd_pic' in col.lower():
                power_col = col
                break
        
        if power_col is None:
            ax.text(0.5, 0.5, 'No mpd_pic data found', ha='center', va='center', fontsize=12)
            return {}
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Store power data for distribution plot
        power_data = {}
        
        # Plot each bank and channel combination
        for idx, row in df.iterrows():
            if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[power_col]):
                continue
                
            bank = int(row['bank'])
            channel = int(row['channel'])
            
            # Parse the array string or get list
            power_values_uw = row[power_col]
            if isinstance(power_values_uw, str):
                # Parse string array like '[13980, 13890, ...]'
                # Filter out empty strings
                power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
            elif not isinstance(power_values_uw, list):
                power_values_uw = [power_values_uw]
            
            # Skip if no valid power data
            if not power_values_uw:
                continue
            
            # Convert to dBm for each datapoint
            power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
            
            # Store for distribution
            power_data[(bank, channel)] = power_dbm
            
            # Create x-axis as datapoint index
            x_values = np.arange(len(power_dbm))
            
            # Plot based on bank
            if bank == 0:
                ax.plot(x_values, power_dbm, '-', 
                       label=f'B0-Ch{channel}', 
                       linewidth=1.5, color=colors_b0[channel])
            else:
                ax.plot(x_values, power_dbm, '--', 
                       label=f'B1-Ch{channel}', 
                       linewidth=1.5, color=colors_b1[channel])
        
        # Add specification limits if available
        if self.specifications:
            spec_key = spec_name.lower()
            if spec_key in self.specifications:
                spec = self.specifications[spec_key]
                
                # Add min/max power limit lines
                if 'min_power_per_wavelength' in spec:
                    min_power = spec['min_power_per_wavelength']['value']
                    ax.axhline(y=min_power, color='red', linestyle='--', linewidth=2, 
                              label=f'Min Spec: {min_power} dBm', alpha=0.7)
                
                if 'max_power_per_wavelength' in spec:
                    max_power = spec['max_power_per_wavelength']['value']
                    ax.axhline(y=max_power, color='red', linestyle='--', linewidth=2, 
                              label=f'Max Spec: {max_power} dBm', alpha=0.7)
                
                # Add shaded region for specification range
                if 'min_power_per_wavelength' in spec and 'max_power_per_wavelength' in spec:
                    ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 
                                   min_power, max_power, 
                                   color='green', alpha=0.1, label='Spec Range')
        
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Power in fiber (dBm)')
        ax.set_title(f'{spec_name} - Tile SN: {full_sn} - Mission Mode Power')
        
        # Set x-axis limits
        ax.set_xlim(0, 3000)
        
        # Set y-axis limits based on specification
        if spec_name == 'Endeavour':
            ax.set_ylim(9, 14)
            ax.set_yticks(np.arange(9, 14.5, 0.5))
        else:  # Kenya
            ax.set_ylim(5, 10)
            ax.set_yticks(np.arange(5, 10.5, 0.5))
        
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        return power_data
    
    def _plot_power_distribution(self, ax, power_data, spec_name, full_sn):
        """Plot statistical distribution of power for each channel"""
        if not power_data:
            ax.text(0.5, 0.5, 'No power data', ha='center', va='center', fontsize=12)
            return
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Prepare data for box plot
        positions = []
        data_list = []
        colors_list = []
        labels_list = []
        
        # Sort by bank and channel (reverse order for bottom-to-top display)
        sorted_keys = sorted(power_data.keys(), reverse=True)
        
        for i, (bank, channel) in enumerate(sorted_keys):
            positions.append(i)
            data_list.append(power_data[(bank, channel)])
            
            if bank == 0:
                colors_list.append(colors_b0[channel])
                labels_list.append(f'B0-Ch{channel}')
            else:
                colors_list.append(colors_b1[channel])
                labels_list.append(f'B1-Ch{channel}')
        
        # Create box plot (horizontal orientation)
        bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Color each box
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay scatter points for all data
        for i, data in enumerate(data_list):
            # Add jitter to y-position for better visibility
            y_positions = np.random.normal(positions[i], 0.08, size=len(data))
            ax.scatter(data, y_positions, alpha=0.4, s=20, color=colors_list[i], marker='o')
        
        # Add median and sigma annotations to the right of each box
        ax_xlim = ax.get_xlim()
        annotation_x_offset = 0.75  # Position as fraction of plot width
        
        for i, data in enumerate(data_list):
            median_val = np.median(data)
            sigma_val = np.std(data)
            
            # Position annotation to the right
            y_pos = positions[i]
            
            # Add text annotation with box
            annotation_text = f'μ̃={median_val:.2f}dBm\nσ={sigma_val:.2f}dB'
            ax.text(annotation_x_offset, y_pos, annotation_text, 
                   fontsize=8, ha='left', va='center',
                   transform=ax.get_yaxis_transform(),
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=colors_list[i], linewidth=1.5, alpha=0.9))
        
        # Add specification limits if available
        if self.specifications:
            spec_key = spec_name.lower()
            if spec_key in self.specifications:
                spec = self.specifications[spec_key]
                
                if 'min_power_per_wavelength' in spec:
                    min_power = spec['min_power_per_wavelength']['value']
                    ax.axvline(x=min_power, color='red', linestyle='--', linewidth=2, 
                              label=f'Min Spec: {min_power} dBm', alpha=0.7)
                
                if 'max_power_per_wavelength' in spec:
                    max_power = spec['max_power_per_wavelength']['value']
                    ax.axvline(x=max_power, color='red', linestyle='--', linewidth=2, 
                              label=f'Max Spec: {max_power} dBm', alpha=0.7)
                
                # Add shaded region for specification range
                if 'min_power_per_wavelength' in spec and 'max_power_per_wavelength' in spec:
                    ax.axvspan(min_power, max_power, 
                              color='green', alpha=0.1, label='Spec Range')
        
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_list)
        ax.set_xlabel('Power in fiber (dBm)')
        ax.set_ylabel('Bank-Channel')
        
        # Add dashed line between B0 and B1
        if len(positions) > 8:
            ax.axhline(y=7.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set x-axis limits based on specification
        if spec_name == 'Endeavour':
            ax.set_xlim(9, 14)
            ax.set_xticks(np.arange(9, 14.5, 0.5))
        else:  # Kenya
            ax.set_xlim(5, 10)
            ax.set_xticks(np.arange(5, 10.5, 0.5))
        
        # Calculate overall statistics
        all_data = [val for data in data_list for val in data]
        overall_median = np.median(all_data)
        overall_sigma = np.std(all_data)
        
        # Extract number of tiles from the full_sn parameter
        if 'Modules' in full_sn:
            n_tiles_text = full_sn  # Already formatted as "N Modules"
        else:
            n_tiles_text = f'Tile SN: {full_sn}'
        
        ax.set_title(f'Statistical Distribution\nμ̃={overall_median:.2f}dBm, σ={overall_sigma:.2f}dB, {n_tiles_text}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        if self.specifications:
            ax.legend(loc='best', fontsize=8)
    
    def _plot_mission_mode_frequency_error(self, endeavour_data, kenya_data, full_sn):
        """Plot frequency error compared to reference grid"""
        print(f"  Generating mission mode frequency error plot...")
        
        if self.reference_grid is None:
            print(f"  Warning: No reference grid available, skipping frequency error analysis")
            return
        
        # Create figure with equal subplot widths
        # Both time series and distribution: 1.5:1 aspect ratio (height:width)
        fig = plt.figure(figsize=(16, 24))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
        
        # Plot Endeavour - time series and distribution (Bank 0 uses set_a, Bank 1 uses set_b)
        if endeavour_data is not None:
            freq_error_data = self._plot_freq_error_spec(axes[0, 0], endeavour_data, 'Endeavour', full_sn)
            self._plot_freq_error_distribution(axes[0, 1], freq_error_data, 'Endeavour', full_sn)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0, 0].set_title(f'Endeavour - Module {full_sn}')
            axes[0, 1].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0, 1].set_title(f'Endeavour Distribution - Module {full_sn}')
        
        # Plot Kenya - time series and distribution (Bank 0 uses set_a, Bank 1 uses set_b)
        if kenya_data is not None:
            freq_error_data = self._plot_freq_error_spec(axes[1, 0], kenya_data, 'Kenya', full_sn)
            self._plot_freq_error_distribution(axes[1, 1], freq_error_data, 'Kenya', full_sn)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title(f'Kenya - Module {full_sn}')
            axes[1, 1].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title(f'Kenya Distribution - Module {full_sn}')
        
        plt.tight_layout()
        plot_filename = f'missionmode_freqerror_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
    
    def _plot_freq_error_spec(self, ax, df, spec_name, full_sn):
        """Plot frequency error for one specification using wavelength column
        Bank 0 uses set_a, Bank 1 uses set_b
        Returns: dict with {(bank, channel): [freq_error_values]} for distribution plotting
        """
        # Find wavelength column
        wl_col = None
        for col in df.columns:
            if col.lower() == 'wavelength':
                wl_col = col
                break
        
        if wl_col is None:
            ax.text(0.5, 0.5, 'No wavelength data', ha='center', va='center', fontsize=12)
            return {}
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Speed of light constant
        c_speed_light = 299792.458  # Speed of light in nm*THz
        
        # Store frequency error data for distribution plot
        freq_error_data = {}
        
        # Plot each bank and channel combination
        for idx, row in df.iterrows():
            if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[wl_col]):
                continue
                
            bank = int(row['bank'])
            channel = int(row['channel'])
            
            # Bank 0 uses set_a, Bank 1 uses set_b
            set_name = 'set_a' if bank == 0 else 'set_b'
            
            # Parse the array string or get list
            wavelength_values_nm = row[wl_col]
            if isinstance(wavelength_values_nm, str):
                # Parse string array like '[1301.36779, 1301.36780, ...]'
                # Filter out empty strings
                wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
            elif not isinstance(wavelength_values_nm, list):
                wavelength_values_nm = [wavelength_values_nm]
            
            # Skip if no valid wavelength data
            if not wavelength_values_nm:
                continue
            
            # Get reference wavelength from grid
            grid_num = channel + 1  # Grid 1-8 corresponds to channel 0-7
            grid_key = f'grid_{grid_num}'
            
            if set_name not in self.reference_grid or grid_key not in self.reference_grid[set_name]:
                continue
            
            ref_wl_nm = self.reference_grid[set_name][grid_key]['wavelength_nm']
            ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
            
            # Calculate frequency error for each wavelength measurement
            freq_errors = []
            for wl in wavelength_values_nm:
                measured_freq_thz = c_speed_light / wl
                freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000  # THz to GHz
                freq_errors.append(freq_error_ghz)
            
            # Remove outliers using median and IQR method
            freq_errors_array = np.array(freq_errors)
            median = np.median(freq_errors_array)
            q1 = np.percentile(freq_errors_array, 25)
            q3 = np.percentile(freq_errors_array, 75)
            iqr = q3 - q1
            
            # Define outlier threshold (1.5 * IQR is standard, but we can use 3 for more tolerance)
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Filter out outliers
            valid_indices = np.where((freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound))[0]
            
            if len(valid_indices) == 0:
                continue  # Skip if all points are outliers
            
            x_values = valid_indices
            freq_errors_filtered = freq_errors_array[valid_indices]
            
            # Store for distribution
            freq_error_data[(bank, channel)] = freq_errors_filtered.tolist()
            
            # Plot based on bank
            if bank == 0:
                ax.plot(x_values, freq_errors_filtered, '-', 
                       label=f'B0-Ch{channel}', 
                       linewidth=1.5, color=colors_b0[channel])
            else:
                ax.plot(x_values, freq_errors_filtered, '--', 
                       label=f'B1-Ch{channel}', 
                       linewidth=1.5, color=colors_b1[channel])
        
        # Add specification limits if available
        if self.specifications:
            spec_key = spec_name.lower()
            if spec_key in self.specifications:
                spec = self.specifications[spec_key]
                
                # Add wavelength error limits
                if 'wavelength_error' in spec:
                    error_limit = spec['wavelength_error']['value']
                    ax.axhline(y=error_limit, color='red', linestyle='--', linewidth=2, 
                              label=f'Spec Limit: ±{error_limit} GHz', alpha=0.7)
                    ax.axhline(y=-error_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    
                    # Add shaded region for specification range
                    ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 
                                   -error_limit, error_limit, 
                                   color='green', alpha=0.1, label='Spec Range')
        
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Frequency Error (GHz)')
        ax.set_title(f'{spec_name} - Tile SN: {full_sn} - Mission Mode Frequency Error')
        
        # Set x-axis limits
        ax.set_xlim(0, 3000)
        
        ax.set_ylim(-40, 40)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        return freq_error_data
    
    def _plot_freq_error_distribution(self, ax, freq_error_data, spec_name, full_sn):
        """Plot statistical distribution of frequency error for each channel"""
        if not freq_error_data:
            ax.text(0.5, 0.5, 'No frequency error data', ha='center', va='center', fontsize=12)
            return
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Prepare data for box plot
        positions = []
        data_list = []
        colors_list = []
        labels_list = []
        
        # Sort by bank and channel (reverse order for bottom-to-top display)
        sorted_keys = sorted(freq_error_data.keys(), reverse=True)
        
        for i, (bank, channel) in enumerate(sorted_keys):
            positions.append(i)
            data_list.append(freq_error_data[(bank, channel)])
            
            if bank == 0:
                colors_list.append(colors_b0[channel])
                labels_list.append(f'B0-Ch{channel}')
            else:
                colors_list.append(colors_b1[channel])
                labels_list.append(f'B1-Ch{channel}')
        
        # Create box plot (horizontal orientation)
        bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Color each box
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay scatter points for all data
        for i, data in enumerate(data_list):
            # Add jitter to y-position for better visibility
            y_positions = np.random.normal(positions[i], 0.08, size=len(data))
            ax.scatter(data, y_positions, alpha=0.4, s=20, color=colors_list[i], marker='o')
        
        # Add median and sigma annotations to the right of each box
        annotation_x_offset = 0.75  # Position as fraction of plot width
        
        for i, data in enumerate(data_list):
            median_val = np.median(data)
            sigma_val = np.std(data)
            
            # Position annotation to the right
            y_pos = positions[i]
            
            # Add text annotation with box
            annotation_text = f'μ̃={median_val:.2f}GHz\nσ={sigma_val:.2f}GHz'
            ax.text(annotation_x_offset, y_pos, annotation_text, 
                   fontsize=8, ha='left', va='center',
                   transform=ax.get_yaxis_transform(),
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=colors_list[i], linewidth=1.5, alpha=0.9))
        
        # Add specification limits if available
        if self.specifications:
            spec_key = spec_name.lower()
            if spec_key in self.specifications:
                spec = self.specifications[spec_key]
                
                # Add wavelength error limits
                if 'wavelength_error' in spec:
                    error_limit = spec['wavelength_error']['value']
                    ax.axvline(x=error_limit, color='red', linestyle='--', linewidth=2, 
                              label=f'Spec Limit: ±{error_limit} GHz', alpha=0.7)
                    ax.axvline(x=-error_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    
                    # Add shaded region for specification range
                    ax.axvspan(-error_limit, error_limit, 
                              color='green', alpha=0.1, label='Spec Range')
        
        # Add zero line
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_list)
        ax.set_xlabel('Frequency Error (GHz)')
        ax.set_ylabel('Bank-Channel')
        
        # Add dashed line between B0 and B1
        if len(positions) > 8:
            ax.axhline(y=7.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlim(-40, 40)
        
        # Calculate overall statistics
        all_data = [val for data in data_list for val in data]
        overall_median = np.median(all_data)
        overall_sigma = np.std(all_data)
        
        # Extract number of tiles from the full_sn parameter
        if 'Modules' in full_sn:
            n_tiles_text = full_sn  # Already formatted as "N Modules"
        else:
            n_tiles_text = f'Tile SN: {full_sn}'
        
        ax.set_title(f'Statistical Distribution\nμ̃={overall_median:.2f}GHz, σ={overall_sigma:.2f}GHz, {n_tiles_text}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        if self.specifications:
            ax.legend(loc='best', fontsize=8)
    
    def _load_wavemeter_data(self, module_path, spec_type):
        """Load wavemeter data for a given specification type"""
        if spec_type == 'Endeavour':
            evt_files = (list(module_path.glob("*Endeavour*EVT*.xlsx")) + 
                        list(module_path.glob("*Endevour*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Endeavour*.xlsx")) +
                        list(module_path.glob("*EVT*Endevour*.xlsx")) +
                        list(module_path.glob("*EVT*result*Endevour*.xlsx")))
        else:  # Kenya
            evt_files = (list(module_path.glob("*Kenya*EVT*.xlsx")) + 
                        list(module_path.glob("*kenya*EVT*.xlsx")) +
                        list(module_path.glob("*EVT*Kenya*.xlsx")) +
                        list(module_path.glob("*EVT*kenya*.xlsx")))
        
        if not evt_files:
            return None
        
        evt_file = evt_files[0]
        
        try:
            excel_file = pd.ExcelFile(evt_file)
            
            # Find wavemeter sheet
            wavemeter_sheet = None
            for sheet in excel_file.sheet_names:
                if 'wavemeter' in sheet.lower():
                    wavemeter_sheet = sheet
                    break
            
            if wavemeter_sheet is None:
                return None
            
            df = pd.read_excel(evt_file, sheet_name=wavemeter_sheet)
            return df
            
        except Exception as e:
            return None
    
    def _plot_boxplot_with_scatter(self, ax, data, all_sns, spec_type, data_type, title):
        """Helper to plot boxplot with scatter points for banks"""
        from matplotlib.patches import Patch
        
        bank0_color = 'blue'
        bank1_color = 'orange'
        
        bank0_data = []
        bank1_data = []
        scatter_b0 = []
        scatter_b1 = []
        
        for i, sn in enumerate(all_sns):
            pos_b0 = i * 2
            pos_b1 = i * 2 + 0.5
            
            sn_b0_values = []
            sn_b1_values = []
            
            if sn in data:
                if 0 in data[sn]:  # Bank 0
                    for ch, value in data[sn][0].items():
                        sn_b0_values.append(value)
                        scatter_b0.append((pos_b0, value))
                
                if 1 in data[sn]:  # Bank 1
                    for ch, value in data[sn][1].items():
                        sn_b1_values.append(value)
                        scatter_b1.append((pos_b1, value))
            
            bank0_data.append(sn_b0_values)
            bank1_data.append(sn_b1_values)
        
        # Create boxplots (filter empty lists)
        bp0_positions = [i * 2 for i in range(len(all_sns))]
        bp1_positions = [i * 2 + 0.5 for i in range(len(all_sns))]
        
        bp0_data = [d for d in bank0_data if len(d) > 0]
        bp1_data = [d for d in bank1_data if len(d) > 0]
        bp0_pos = [bp0_positions[i] for i, d in enumerate(bank0_data) if len(d) > 0]
        bp1_pos = [bp1_positions[i] for i, d in enumerate(bank1_data) if len(d) > 0]
        
        if bp0_data:
            ax.boxplot(bp0_data, positions=bp0_pos, widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=bank0_color, alpha=0.5),
                      medianprops=dict(color='black', linewidth=2))
        
        if bp1_data:
            ax.boxplot(bp1_data, positions=bp1_pos, widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=bank1_color, alpha=0.5),
                      medianprops=dict(color='black', linewidth=2))
        
        # Add scatter points
        for pos, value in scatter_b0:
            ax.scatter(pos, value, c=bank0_color, s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        for pos, value in scatter_b1:
            ax.scatter(pos, value, c=bank1_color, s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        # Add spec limits
        if self.specifications and spec_type in self.specifications:
            spec = self.specifications[spec_type]
            if data_type == 'power':
                if 'min_power_per_wavelength' in spec:
                    ax.axhline(y=spec['min_power_per_wavelength']['value'], color='red', linestyle='--', linewidth=2, alpha=0.7)
                if 'max_power_per_wavelength' in spec:
                    ax.axhline(y=spec['max_power_per_wavelength']['value'], color='red', linestyle='--', linewidth=2, alpha=0.7)
            elif data_type == 'freq':
                if 'wavelength_error' in spec:
                    error_limit = spec['wavelength_error']['value']
                    ax.axhline(y=error_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax.axhline(y=-error_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            elif data_type == 'smsr':
                # Add 40dB spec line for SMSR (both Endeavour and Kenya)
                ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Spec: 40dB')
        
        # Set labels
        ax.set_xlabel('Tile SN')
        if data_type == 'power':
            ylabel = 'Power in fiber (dBm)'
        elif data_type == 'freq':
            ylabel = 'Frequency Error (GHz)'
        else:  # smsr
            ylabel = 'SMSR (dB)'
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        center_positions = [i * 2 + 0.25 for i in range(len(all_sns))]
        ax.set_xticks(center_positions)
        ax.set_xticklabels(all_sns, rotation=45, ha='right')
        
        # Set y-limits for SMSR plots
        if data_type == 'smsr':
            ax.set_ylim(30, 60)
        
        # Legend
        legend_elements = [Patch(facecolor=bank0_color, alpha=0.5, label='Bank 0'),
                          Patch(facecolor=bank1_color, alpha=0.5, label='Bank 1')]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_mission_mode_summary(self, modules_data):
        """Create mission mode summary plot with 3x2 subplots (power, freq error, SMSR)"""
        print(f"\n{'='*60}")
        print("Creating Mission Mode Summary Plot")
        print(f"{'='*60}")
        
        # Create figure with 3x2 subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Process data for each module
        # Structure: {sn: {bank: {channel: value}}}
        endeavour_power_data = {}
        endeavour_freq_data = {}
        endeavour_smsr_data = {}
        kenya_power_data = {}
        kenya_freq_data = {}
        kenya_smsr_data = {}
        
        # Speed of light constant for frequency calculation
        c_speed_light = 299792.458
        
        for full_sn, data in sorted(modules_data.items()):
            # Load SMSR data from "Power and SMSR" tab
            module_sn = full_sn.replace('Y2534000', '')
            module_path = self.base_path / module_sn
            
            # Load Endeavour SMSR
            endeavour_smsr_df = self._load_smsr_data(module_path, 'endeavour')
            if endeavour_smsr_df is not None:
                for idx, row in endeavour_smsr_df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row['SMSR']):
                        continue
                    channel = int(row['channel'])
                    bank = int(row['bank'])
                    smsr_value = float(row['SMSR'])
                    
                    if full_sn not in endeavour_smsr_data:
                        endeavour_smsr_data[full_sn] = {}
                    if bank not in endeavour_smsr_data[full_sn]:
                        endeavour_smsr_data[full_sn][bank] = {}
                    endeavour_smsr_data[full_sn][bank][channel] = smsr_value
            
            # Load Kenya SMSR
            kenya_smsr_df = self._load_smsr_data(module_path, 'kenya')
            if kenya_smsr_df is not None:
                for idx, row in kenya_smsr_df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row['SMSR']):
                        continue
                    channel = int(row['channel'])
                    bank = int(row['bank'])
                    smsr_value = float(row['SMSR'])
                    
                    if full_sn not in kenya_smsr_data:
                        kenya_smsr_data[full_sn] = {}
                    if bank not in kenya_smsr_data[full_sn]:
                        kenya_smsr_data[full_sn][bank] = {}
                    kenya_smsr_data[full_sn][bank][channel] = smsr_value
            
            # Process Endeavour data
            if data['endeavour'] is not None:
                df = data['endeavour']
                for idx, row in df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']):
                        continue
                    
                    channel = int(row['channel'])
                    bank = int(row['bank'])
                    
                    # Bank 0 uses set_a, Bank 1 uses set_b
                    set_name = 'set_a' if bank == 0 else 'set_b'
                    
                    # Process power data
                    if 'mpd_pic' in df.columns and not pd.isna(row['mpd_pic']):
                        power_values_uw = row['mpd_pic']
                        if isinstance(power_values_uw, str):
                            power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                            if power_values_uw:
                                power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                                avg_power = np.mean(power_dbm)
                                
                                if full_sn not in endeavour_power_data:
                                    endeavour_power_data[full_sn] = {}
                                if bank not in endeavour_power_data[full_sn]:
                                    endeavour_power_data[full_sn][bank] = {}
                                endeavour_power_data[full_sn][bank][channel] = avg_power
                    
                    # Process frequency error data
                    if 'wavelength' in df.columns and not pd.isna(row['wavelength']):
                        wavelength_values_nm = row['wavelength']
                        if isinstance(wavelength_values_nm, str):
                            wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
                            if wavelength_values_nm and self.reference_grid:
                                grid_num = channel + 1
                                grid_key = f'grid_{grid_num}'
                                
                                if set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                                    ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                                    
                                    freq_errors = []
                                    for wl in wavelength_values_nm:
                                        measured_freq_thz = c_speed_light / wl
                                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                                        freq_errors.append(freq_error_ghz)
                                    
                                    # Filter outliers
                                    freq_errors_array = np.array(freq_errors)
                                    q1 = np.percentile(freq_errors_array, 25)
                                    q3 = np.percentile(freq_errors_array, 75)
                                    iqr = q3 - q1
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    valid_freq_errors = freq_errors_array[(freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound)]
                                    
                                    if len(valid_freq_errors) > 0:
                                        avg_freq_error = np.mean(valid_freq_errors)
                                        
                                        if full_sn not in endeavour_freq_data:
                                            endeavour_freq_data[full_sn] = {}
                                        if bank not in endeavour_freq_data[full_sn]:
                                            endeavour_freq_data[full_sn][bank] = {}
                                        endeavour_freq_data[full_sn][bank][channel] = avg_freq_error
            
            # Process Kenya data
            if data['kenya'] is not None:
                df = data['kenya']
                for idx, row in df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']):
                        continue
                    
                    channel = int(row['channel'])
                    bank = int(row['bank'])
                    
                    # Bank 0 uses set_a, Bank 1 uses set_b
                    set_name = 'set_a' if bank == 0 else 'set_b'
                    
                    # Process power data
                    if 'mpd_pic' in df.columns and not pd.isna(row['mpd_pic']):
                        power_values_uw = row['mpd_pic']
                        if isinstance(power_values_uw, str):
                            power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                            if power_values_uw:
                                power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                                avg_power = np.mean(power_dbm)
                                
                                if full_sn not in kenya_power_data:
                                    kenya_power_data[full_sn] = {}
                                if bank not in kenya_power_data[full_sn]:
                                    kenya_power_data[full_sn][bank] = {}
                                kenya_power_data[full_sn][bank][channel] = avg_power
                    
                    # Process frequency error data
                    if 'wavelength' in df.columns and not pd.isna(row['wavelength']):
                        wavelength_values_nm = row['wavelength']
                        if isinstance(wavelength_values_nm, str):
                            wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
                            if wavelength_values_nm and self.reference_grid:
                                grid_num = channel + 1
                                grid_key = f'grid_{grid_num}'
                                
                                if set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                                    ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                                    
                                    freq_errors = []
                                    for wl in wavelength_values_nm:
                                        measured_freq_thz = c_speed_light / wl
                                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                                        freq_errors.append(freq_error_ghz)
                                    
                                    # Filter outliers
                                    freq_errors_array = np.array(freq_errors)
                                    q1 = np.percentile(freq_errors_array, 25)
                                    q3 = np.percentile(freq_errors_array, 75)
                                    iqr = q3 - q1
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    valid_freq_errors = freq_errors_array[(freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound)]
                                    
                                    if len(valid_freq_errors) > 0:
                                        avg_freq_error = np.mean(valid_freq_errors)
                                        
                                        if full_sn not in kenya_freq_data:
                                            kenya_freq_data[full_sn] = {}
                                        if bank not in kenya_freq_data[full_sn]:
                                            kenya_freq_data[full_sn][bank] = {}
                                        kenya_freq_data[full_sn][bank][channel] = avg_freq_error
        
        # Get sorted list of SNs
        all_sns = sorted(modules_data.keys())
        
        # Colors for banks
        bank0_color = 'blue'
        bank1_color = 'orange'
        
        # Plot 1: Endeavour Power (row 0, left)
        ax = axes[0, 0]
        self._plot_boxplot_with_scatter(ax, endeavour_power_data, all_sns, 'endeavour', 'power', 
                                        'Endeavour - Mission Mode Power')
        
        # Plot 2: Kenya Power (row 0, right)
        ax = axes[0, 1]
        self._plot_boxplot_with_scatter(ax, kenya_power_data, all_sns, 'kenya', 'power',
                                        'Kenya - Mission Mode Power')
        
        # Plot 3: Endeavour Frequency Error (row 1, left)
        ax = axes[1, 0]
        self._plot_boxplot_with_scatter(ax, endeavour_freq_data, all_sns, 'endeavour', 'freq',
                                        'Endeavour - Mission Mode Frequency Error')
        
        # Plot 4: Kenya Frequency Error (row 1, right)
        ax = axes[1, 1]
        self._plot_boxplot_with_scatter(ax, kenya_freq_data, all_sns, 'kenya', 'freq',
                                        'Kenya - Mission Mode Frequency Error')
        
        # Plot 5: Endeavour SMSR (row 2, left)
        ax = axes[2, 0]
        self._plot_boxplot_with_scatter(ax, endeavour_smsr_data, all_sns, 'endeavour', 'smsr',
                                        'Endeavour - SMSR')
        
        # Plot 6: Kenya SMSR (row 2, right)
        ax = axes[2, 1]
        self._plot_boxplot_with_scatter(ax, kenya_smsr_data, all_sns, 'kenya', 'smsr',
                                        'Kenya - SMSR')
        
        plt.tight_layout()
        
        # Save to analysis_results root
        summary_path = self.results_path
        plot_filename = 'missionmode_summary.png'
        plt.savefig(summary_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Mission Mode summary plot saved: {plot_filename}")
        print(f"  Location: analysis_results/{plot_filename}")
    
    def create_mission_mode_statistical_summary(self, modules_data):
        """Create statistical distribution summary plot combining all modules"""
        print(f"\n{'='*60}")
        print("Creating Mission Mode Statistical Distribution Summary")
        print(f"{'='*60}")
        
        # Create figure with 2x2 layout (Endeavour/Kenya x Power/FreqError)
        # Each subplot has 1.5:1 aspect ratio (height:width)
        fig, axes = plt.subplots(2, 2, figsize=(16, 24))
        
        # Collect all channel data from all modules
        endeavour_power_all = {}
        kenya_power_all = {}
        endeavour_freq_all = {}
        kenya_freq_all = {}
        
        # Speed of light constant for frequency calculation
        c_speed_light = 299792.458
        
        for full_sn, data in sorted(modules_data.items()):
            # Process Endeavour power data
            if data['endeavour'] is not None:
                df = data['endeavour']
                for idx, row in df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']):
                        continue
                    
                    bank = int(row['bank'])
                    channel = int(row['channel'])
                    key = (bank, channel)
                    
                    # Process power data
                    if 'mpd_pic' in df.columns and not pd.isna(row['mpd_pic']):
                        power_values_uw = row['mpd_pic']
                        if isinstance(power_values_uw, str):
                            power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                            if power_values_uw:
                                power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                                if key not in endeavour_power_all:
                                    endeavour_power_all[key] = []
                                endeavour_power_all[key].extend(power_dbm)
                    
                    # Process frequency error data
                    set_name = 'set_a' if bank == 0 else 'set_b'
                    if 'wavelength' in df.columns and not pd.isna(row['wavelength']):
                        wavelength_values_nm = row['wavelength']
                        if isinstance(wavelength_values_nm, str):
                            wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
                            if wavelength_values_nm and self.reference_grid:
                                grid_num = channel + 1
                                grid_key = f'grid_{grid_num}'
                                
                                if set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                                    ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                                    
                                    freq_errors = []
                                    for wl in wavelength_values_nm:
                                        measured_freq_thz = c_speed_light / wl
                                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                                        freq_errors.append(freq_error_ghz)
                                    
                                    # Filter outliers
                                    freq_errors_array = np.array(freq_errors)
                                    q1 = np.percentile(freq_errors_array, 25)
                                    q3 = np.percentile(freq_errors_array, 75)
                                    iqr = q3 - q1
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    valid_freq_errors = freq_errors_array[(freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound)]
                                    
                                    if len(valid_freq_errors) > 0:
                                        if key not in endeavour_freq_all:
                                            endeavour_freq_all[key] = []
                                        endeavour_freq_all[key].extend(valid_freq_errors.tolist())
            
            # Process Kenya data
            if data['kenya'] is not None:
                df = data['kenya']
                for idx, row in df.iterrows():
                    if pd.isna(row['bank']) or pd.isna(row['channel']):
                        continue
                    
                    bank = int(row['bank'])
                    channel = int(row['channel'])
                    key = (bank, channel)
                    
                    # Process power data
                    if 'mpd_pic' in df.columns and not pd.isna(row['mpd_pic']):
                        power_values_uw = row['mpd_pic']
                        if isinstance(power_values_uw, str):
                            power_values_uw = [float(x.strip()) for x in power_values_uw.strip('[]').split(',') if x.strip()]
                            if power_values_uw:
                                power_dbm = [10 * np.log10(p / 1000.0) for p in power_values_uw]
                                if key not in kenya_power_all:
                                    kenya_power_all[key] = []
                                kenya_power_all[key].extend(power_dbm)
                    
                    # Process frequency error data
                    set_name = 'set_a' if bank == 0 else 'set_b'
                    if 'wavelength' in df.columns and not pd.isna(row['wavelength']):
                        wavelength_values_nm = row['wavelength']
                        if isinstance(wavelength_values_nm, str):
                            wavelength_values_nm = [float(x.strip()) for x in wavelength_values_nm.strip('[]').split(',') if x.strip()]
                            if wavelength_values_nm and self.reference_grid:
                                grid_num = channel + 1
                                grid_key = f'grid_{grid_num}'
                                
                                if set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                                    ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                                    
                                    freq_errors = []
                                    for wl in wavelength_values_nm:
                                        measured_freq_thz = c_speed_light / wl
                                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                                        freq_errors.append(freq_error_ghz)
                                    
                                    # Filter outliers
                                    freq_errors_array = np.array(freq_errors)
                                    q1 = np.percentile(freq_errors_array, 25)
                                    q3 = np.percentile(freq_errors_array, 75)
                                    iqr = q3 - q1
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    valid_freq_errors = freq_errors_array[(freq_errors_array >= lower_bound) & (freq_errors_array <= upper_bound)]
                                    
                                    if len(valid_freq_errors) > 0:
                                        if key not in kenya_freq_all:
                                            kenya_freq_all[key] = []
                                        kenya_freq_all[key].extend(valid_freq_errors.tolist())
        
        # Plot distributions
        self._plot_power_distribution(axes[0, 0], endeavour_power_all, 'Endeavour', f'{len(modules_data)} Modules')
        self._plot_power_distribution(axes[1, 0], kenya_power_all, 'Kenya', f'{len(modules_data)} Modules')
        self._plot_freq_error_distribution(axes[0, 1], endeavour_freq_all, 'Endeavour', f'{len(modules_data)} Modules')
        self._plot_freq_error_distribution(axes[1, 1], kenya_freq_all, 'Kenya', f'{len(modules_data)} Modules')
        
        plt.tight_layout()
        
        # Save to analysis_results root
        summary_path = self.results_path
        plot_filename = 'missionmode_statistical_summary.png'
        plt.savefig(summary_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Mission Mode statistical summary plot saved: {plot_filename}")
        print(f"  Location: analysis_results/{plot_filename}")
    
    def plot_operating_points(self, module_sn, full_sn, endeavour_data, kenya_data):
        """Plot operating points (tpic, tmux, tpmic, laser_dac, voa_dac) for a module"""
        print(f"\n  Generating operating points plot for {full_sn}...")
        
        # Create figure with 5x2 subplots (5 parameters x 2 specs)
        fig, axes = plt.subplots(5, 2, figsize=(14, 20))
        
        # Parameters to plot (lowercase as they appear in the data)
        params = ['tpic', 'tmux', 'tpmic', 'laser_dac', 'voa_dac']
        
        # Colors for banks
        bank0_color = 'blue'
        bank1_color = 'orange'
        
        # Plot for Endeavour (left column)
        if endeavour_data is not None:
            self._plot_operating_params(axes[:, 0], endeavour_data, params, 
                                       f'Endeavour - Tile SN: {full_sn}', 
                                       bank0_color, bank1_color)
        else:
            for ax in axes[:, 0]:
                ax.text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Plot for Kenya (right column)
        if kenya_data is not None:
            self._plot_operating_params(axes[:, 1], kenya_data, params,
                                       f'Kenya - Tile SN: {full_sn}',
                                       bank0_color, bank1_color)
        else:
            for ax in axes[:, 1]:
                ax.text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'missionmode_operatingpoints_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Operating points plot saved: {plot_filename}")
    
    def _plot_operating_params(self, axes, df, params, title_prefix, bank0_color, bank1_color):
        """Helper to plot operating parameters vs datapoint index"""
        from matplotlib.patches import Patch
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        for i, param in enumerate(params):
            ax = axes[i]
            
            # Check if parameter exists in dataframe
            if param not in df.columns:
                ax.text(0.5, 0.5, f'No {param} data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{param.upper()}')
                continue
            
            # Plot each bank and channel combination
            for idx, row in df.iterrows():
                if pd.isna(row['bank']) or pd.isna(row['channel']) or pd.isna(row[param]):
                    continue
                
                bank = int(row['bank'])
                channel = int(row['channel'])
                value = row[param]
                
                # Parse array data
                if isinstance(value, str):
                    try:
                        value_list = [float(x.strip()) for x in value.strip('[]').split(',') if x.strip()]
                    except:
                        continue
                elif isinstance(value, list):
                    value_list = value
                else:
                    # Single value, create single-element array
                    value_list = [float(value)]
                
                if not value_list:
                    continue
                
                # Create x-axis as datapoint index
                x_values = np.arange(len(value_list))
                
                # Plot based on bank
                if bank == 0:
                    ax.plot(x_values, value_list, '-', 
                           label=f'B0-Ch{channel}', 
                           linewidth=1.5, color=colors_b0[channel])
                else:
                    ax.plot(x_values, value_list, '--', 
                           label=f'B1-Ch{channel}', 
                           linewidth=1.5, color=colors_b1[channel])
            
            # Set labels
            ax.set_xlabel('Seconds')
            
            # Set y-label based on parameter
            if param in ['tpic', 'tmux', 'tpmic']:
                ylabel = f'{param.upper()} (°C)'
            elif param == 'voa_dac':
                ylabel = 'VOA current (mA)'
            elif param == 'laser_dac':
                ylabel = 'Laser current (mA)'
            else:
                ylabel = f'{param.upper()}'
            ax.set_ylabel(ylabel)
            
            # Set title
            if i == 0:
                ax.set_title(f'{title_prefix}\n{param.upper()}')
            else:
                ax.set_title(f'{param.upper()}')
            
            ax.legend(loc='best', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
    
    def create_fru_summary(self, module_sn):
        """
        Create FRU summary plot for a module showing calibration data.
        
        Args:
            module_sn (str): Module serial number (e.g., '165')
        """
        module_path = self.base_path / module_sn
        
        # Get full serial number
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn
        if config_files:
            parts = config_files[0].name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]
        
        print(f"\n{'='*60}")
        print(f"Creating FRU Summary for Module {full_sn}")
        print(f"{'='*60}")
        
        # Load Endeavour FRU data
        endeavour_fru = self._load_fru_data(module_path, 'Endeavour')
        
        # Load Kenya FRU data
        kenya_fru = self._load_fru_data(module_path, 'Kenya')
        
        # Load Onet data
        onet_data = self._load_onet_data(module_path)
        
        if endeavour_fru is None and kenya_fru is None:
            print(f"  No FRU data available for module {full_sn}")
            return
        
        # Create 4x2 subplot figure
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        
        # Plot Endeavour column (left)
        if endeavour_fru is not None or onet_data is not None:
            self._plot_fru_column(axes[:, 0], endeavour_fru, onet_data, 'Endeavour', full_sn)
        else:
            for ax in axes[:, 0]:
                ax.text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
        
        # Plot Kenya column (right)
        if kenya_fru is not None or onet_data is not None:
            self._plot_fru_column(axes[:, 1], kenya_fru, onet_data, 'Kenya', full_sn)
        else:
            for ax in axes[:, 1]:
                ax.text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Save to analysis_results/calibration folder
        plot_filename = f'fru_{full_sn}.png'
        plt.savefig(self.calibration_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ FRU plot saved: {plot_filename}")
    
    def _load_fru_data(self, module_path, spec_type):
        """Load FRU configuration data (Endeavour_14mW.yaml or Kenya_6p4mW.yaml)"""
        if spec_type == 'Endeavour':
            fru_files = list(module_path.glob("*Endeavour*14mW*.yaml"))
        else:  # Kenya
            fru_files = list(module_path.glob("*Kenya*6p4mW*.yaml"))
        
        if not fru_files:
            print(f"  No {spec_type} FRU file found")
            return None
        
        fru_file = fru_files[0]
        print(f"  Loading {spec_type} FRU: {fru_file.name}")
        
        try:
            with open(fru_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"  Error loading {spec_type} FRU data: {e}")
            return None
    
    def _load_onet_data(self, module_path):
        """Load Onet.yaml data"""
        onet_files = list(module_path.glob("*Onet.yaml"))
        
        if not onet_files:
            print(f"  No Onet.yaml file found")
            return None
        
        onet_file = onet_files[0]
        print(f"  Loading Onet: {onet_file.name}")
        
        try:
            with open(onet_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"  Error loading Onet data: {e}")
            return None
    
    def _plot_fru_column(self, axes, fru_data, onet_data, spec_name, full_sn):
        """Plot FRU data column (4 rows: laser current, voa current, responsivity pic, pic temp)"""
        
        # Row 0: Laser Current
        ax = axes[0]
        self._plot_fru_parameter(ax, fru_data, onet_data, 'laser_current', spec_name, full_sn)
        
        # Row 1: VOA Current
        ax = axes[1]
        self._plot_fru_parameter(ax, fru_data, onet_data, 'voa_current', spec_name, full_sn)
        
        # Row 2: Responsivity PIC MPD
        ax = axes[2]
        self._plot_fru_parameter(ax, fru_data, onet_data, 'responsivity_pic', spec_name, full_sn)
        
        # Row 3: PIC Temperature
        ax = axes[3]
        self._plot_fru_parameter(ax, fru_data, onet_data, 'pic_temp', spec_name, full_sn)
    
    def _plot_fru_parameter(self, ax, fru_data, onet_data, param_type, spec_name, full_sn):
        """Plot a single FRU parameter with boxplot and scatter"""
        
        # Extract data based on parameter type
        fru_values = []
        onet_values = []
        
        if param_type == 'laser_current':
            if fru_data and 'current' in fru_data and 'laser' in fru_data['current']:
                fru_values = fru_data['current']['laser']
            if onet_data and 'bias_currents' in onet_data:
                onet_values = onet_data['bias_currents']
            ylabel = 'Laser Current (mA)'
            title = f'{spec_name} - Laser Current'
            
        elif param_type == 'voa_current':
            if fru_data and 'current' in fru_data and 'voa' in fru_data['current']:
                fru_values = fru_data['current']['voa']
            ylabel = 'VOA Current (mA)'
            title = f'{spec_name} - VOA Current'
            
        elif param_type == 'responsivity_pic':
            if fru_data and 'responsivity' in fru_data and 'pic' in fru_data['responsivity']:
                fru_values = fru_data['responsivity']['pic']
            if onet_data and 'pic_mpd_tap_ratios' in onet_data:
                onet_values = onet_data['pic_mpd_tap_ratios']
            ylabel = 'PIC MPD Responsivity'
            title = f'{spec_name} - PIC MPD Responsivity'
            
        elif param_type == 'pic_temp':
            if fru_data and 'temp' in fru_data and 'pic' in fru_data['temp']:
                fru_values = [fru_data['temp']['pic']]  # Single value, make it a list
            if onet_data and 'temp_pic' in onet_data:
                onet_values = [onet_data['temp_pic']]  # Single value, make it a list
            ylabel = 'PIC Temperature (°C)'
            title = f'{spec_name} - PIC Temperature'
        
        # Prepare data for plotting
        plot_data = []
        labels = []
        colors = []
        
        if fru_values:
            plot_data.append(fru_values)
            labels.append('LM')
            colors.append('blue')
        
        if onet_values:
            plot_data.append(onet_values)
            labels.append('Onet')
            colors.append('orange')
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Create boxplot
        positions = list(range(len(plot_data)))
        bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Add scatter points
        for i, (data, color) in enumerate(zip(plot_data, colors)):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.6, c=color, s=30, edgecolors='black', linewidths=0.5)
        
        # Set labels and title
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\nTile SN: {full_sn}')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_fru_summary_all_modules(self, all_modules_data):
        """
        Create aggregated FRU summary plot with all modules.
        4x2 subplots with x-axis as Tile SN.
        
        Args:
            all_modules_data: Dictionary with all module data
                {full_sn: {'endeavour_fru': data, 'kenya_fru': data, 'onet': data}}
        """
        print(f"\n{'='*60}")
        print("Creating Aggregated FRU Summary Plot")
        print(f"{'='*60}")
        
        # Create figure with 4x2 subplots
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        
        # Get sorted list of SNs
        all_sns = sorted(all_modules_data.keys())
        
        # Plot Endeavour column (left)
        self._plot_fru_summary_column(axes[:, 0], all_modules_data, all_sns, 'Endeavour')
        
        # Plot Kenya column (right)
        self._plot_fru_summary_column(axes[:, 1], all_modules_data, all_sns, 'Kenya')
        
        plt.tight_layout()
        
        # Save to analysis_results root folder
        plot_filename = 'fru_summary.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Aggregated FRU summary plot saved: {plot_filename}")
        print(f"  Location: analysis_results/{plot_filename}")
    
    def _plot_fru_summary_column(self, axes, all_modules_data, all_sns, spec_name):
        """Plot one column of FRU summary (4 rows for each parameter type)"""
        
        # Row 0: Laser Current
        ax = axes[0]
        self._plot_fru_summary_parameter(ax, all_modules_data, all_sns, 'laser_current', spec_name)
        
        # Row 1: VOA Current
        ax = axes[1]
        self._plot_fru_summary_parameter(ax, all_modules_data, all_sns, 'voa_current', spec_name)
        
        # Row 2: Responsivity PIC MPD
        ax = axes[2]
        self._plot_fru_summary_parameter(ax, all_modules_data, all_sns, 'responsivity_pic', spec_name)
        
        # Row 3: PIC Temperature
        ax = axes[3]
        self._plot_fru_summary_parameter(ax, all_modules_data, all_sns, 'pic_temp', spec_name)
    
    def _plot_fru_summary_parameter(self, ax, all_modules_data, all_sns, param_type, spec_name):
        """Plot aggregated parameter data with boxplot and scatter for all modules"""
        
        # Collect data for all modules
        fru_data_by_sn = []
        onet_data_by_sn = []
        
        for sn in all_sns:
            module_data = all_modules_data[sn]
            
            # Select FRU data based on spec
            if spec_name == 'Endeavour':
                fru_data = module_data['endeavour_fru']
            else:
                fru_data = module_data['kenya_fru']
            
            onet_data = module_data['onet']
            
            # Extract parameter values
            fru_values = []
            onet_values = []
            
            if param_type == 'laser_current':
                if fru_data and 'current' in fru_data and 'laser' in fru_data['current']:
                    fru_values = fru_data['current']['laser']
                if onet_data and 'bias_currents' in onet_data:
                    onet_values = onet_data['bias_currents']
                ylabel = 'Laser Current (mA)'
                title = f'{spec_name} - Laser Current'
                
            elif param_type == 'voa_current':
                if fru_data and 'current' in fru_data and 'voa' in fru_data['current']:
                    fru_values = fru_data['current']['voa']
                ylabel = 'VOA Current (mA)'
                title = f'{spec_name} - VOA Current'
                
            elif param_type == 'responsivity_pic':
                if fru_data and 'responsivity' in fru_data and 'pic' in fru_data['responsivity']:
                    fru_values = fru_data['responsivity']['pic']
                if onet_data and 'pic_mpd_tap_ratios' in onet_data:
                    onet_values = onet_data['pic_mpd_tap_ratios']
                ylabel = 'PIC MPD Responsivity'
                title = f'{spec_name} - PIC MPD Responsivity'
                
            elif param_type == 'pic_temp':
                if fru_data and 'temp' in fru_data and 'pic' in fru_data['temp']:
                    fru_values = [fru_data['temp']['pic']]
                if onet_data and 'temp_pic' in onet_data:
                    onet_values = [onet_data['temp_pic']]
                ylabel = 'PIC Temperature (°C)'
                title = f'{spec_name} - PIC Temperature'
            
            fru_data_by_sn.append(fru_values)
            onet_data_by_sn.append(onet_values)
        
        # Create boxplots for FRU data
        fru_positions = [i * 3 for i in range(len(all_sns))]
        onet_positions = [i * 3 + 1 for i in range(len(all_sns))]
        
        # Filter out empty data
        fru_plot_data = [d for d in fru_data_by_sn if len(d) > 0]
        fru_plot_positions = [pos for i, pos in enumerate(fru_positions) if len(fru_data_by_sn[i]) > 0]
        
        onet_plot_data = [d for d in onet_data_by_sn if len(d) > 0]
        onet_plot_positions = [pos for i, pos in enumerate(onet_positions) if len(onet_data_by_sn[i]) > 0]
        
        # Plot FRU boxplots
        if fru_plot_data:
            bp_fru = ax.boxplot(fru_plot_data, positions=fru_plot_positions, widths=0.8, 
                               patch_artist=True, medianprops=dict(color='black', linewidth=2))
            for patch in bp_fru['boxes']:
                patch.set_facecolor('blue')
                patch.set_alpha(0.5)
            
            # Add scatter points for FRU
            for i, (pos, data) in enumerate(zip(fru_plot_positions, fru_plot_data)):
                x = np.random.normal(pos, 0.08, size=len(data))
                ax.scatter(x, data, alpha=0.6, c='blue', s=20, edgecolors='black', linewidths=0.5)
        
        # Plot Onet boxplots
        if onet_plot_data:
            bp_onet = ax.boxplot(onet_plot_data, positions=onet_plot_positions, widths=0.8,
                                patch_artist=True, medianprops=dict(color='black', linewidth=2))
            for patch in bp_onet['boxes']:
                patch.set_facecolor('orange')
                patch.set_alpha(0.5)
            
            # Add scatter points for Onet
            for i, (pos, data) in enumerate(zip(onet_plot_positions, onet_plot_data)):
                x = np.random.normal(pos, 0.08, size=len(data))
                ax.scatter(x, data, alpha=0.6, c='orange', s=20, edgecolors='black', linewidths=0.5)
        
        # Set labels and title
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Set x-ticks at center between FRU and Onet for each SN
        center_positions = [i * 3 + 0.5 for i in range(len(all_sns))]
        ax.set_xticks(center_positions)
        ax.set_xticklabels(all_sns, rotation=45, ha='right')
        ax.set_xlabel('Tile SN')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.5, label='LM'),
                          Patch(facecolor='orange', alpha=0.5, label='Onet')]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_calibration_log_plot(self, module_sn):
        """
        Create calibration plot from Raw log data showing temperature, currents, power, and frequency.
        
        Args:
            module_sn (str): Module serial number (e.g., '171')
        """
        module_path = self.base_path / module_sn
        raw_log_path = module_path / "Raw log data"
        
        if not raw_log_path.exists():
            print(f"  No Raw log data found for module {module_sn}")
            return
        
        # Get full serial number
        config_files = list(module_path.glob("*config*.yaml"))
        full_sn = module_sn
        if config_files:
            parts = config_files[0].name.split('_')
            if len(parts) >= 3:
                full_sn = parts[2]
        
        print(f"\n{'='*60}")
        print(f"Creating Calibration Log Plot for Module {full_sn}")
        print(f"{'='*60}")
        
        # Load Endeavour and Kenya log data
        endeavour_log = self._load_calibration_log(raw_log_path, full_sn, 'endeavour')
        kenya_log = self._load_calibration_log(raw_log_path, full_sn, 'kenya')
        
        if endeavour_log is None and kenya_log is None:
            print(f"  No calibration log data available for module {full_sn}")
            return
        
        # Create 5x2 subplot figure (5 rows: temp_pic, LD currents, VOA currents, laser power, frequency)
        fig, axes = plt.subplots(5, 2, figsize=(18, 20))
        
        # Plot Endeavour column (left)
        if endeavour_log is not None:
            self._plot_calibration_log_column(axes[:, 0], endeavour_log, 'Endeavour', full_sn)
        else:
            for ax in axes[:, 0]:
                ax.text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
        
        # Plot Kenya column (right)
        if kenya_log is not None:
            self._plot_calibration_log_column(axes[:, 1], kenya_log, 'Kenya', full_sn)
        else:
            for ax in axes[:, 1]:
                ax.text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
        
        plt.suptitle(f'Device Log Analysis - Tile SN: {full_sn}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to calibration folder
        plot_filename = f'calibration_{full_sn}.png'
        plt.savefig(self.calibration_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Calibration log plot saved: {plot_filename}")
    
    def _load_calibration_log(self, raw_log_path, full_sn, spec_type):
        """Load calibration log CSV file"""
        # Try multiple patterns
        # Pattern 1: log_Y2534000165_endeavour.csv
        log_file = raw_log_path / f"log_{full_sn}_{spec_type}.csv"
        
        if not log_file.exists():
            # Pattern 2: log_Y2534000167_endeavour_2025-10-03_14_42_16.csv
            log_files = list(raw_log_path.glob(f"log_{full_sn}_{spec_type}_*.csv"))
            if log_files:
                log_file = log_files[0]
            else:
                # Pattern 3: Look in subdirectories logs_endeavour_Y2534000156/*/*.csv
                subdir_pattern = f"logs_{spec_type}_{full_sn}"
                subdir_path = raw_log_path / subdir_pattern
                if subdir_path.exists():
                    # Look for P0_T3_log.csv or similar in timestamp subdirectories
                    log_files = list(subdir_path.glob("*/P0_T3_log.csv"))
                    if not log_files:
                        # Try any CSV file in subdirectories
                        log_files = list(subdir_path.glob("*/*.csv"))
                    if log_files:
                        # If multiple files, combine them
                        if len(log_files) > 1:
                            print(f"  Found {len(log_files)} {spec_type} log files, combining...")
                            dfs = []
                            for f in sorted(log_files):
                                try:
                                    df_temp = pd.read_csv(f)
                                    dfs.append(df_temp)
                                except:
                                    pass
                            if dfs:
                                df = pd.concat(dfs, ignore_index=True)
                                return df
                        else:
                            log_file = log_files[0]
                    else:
                        print(f"  No {spec_type} calibration log found")
                        return None
                else:
                    print(f"  No {spec_type} calibration log found")
                    return None
        
        print(f"  Loading {spec_type} calibration log: {log_file.name}")
        
        try:
            df = pd.read_csv(log_file)
            return df
        except Exception as e:
            print(f"  Error loading {spec_type} calibration log: {e}")
            return None
    
    def _plot_calibration_log_column(self, axes, df, spec_name, full_sn):
        """Plot calibration log data for one specification (5 rows)"""
        
        # Convert timestamp to relative time in seconds
        if 'time' in df.columns:
            time = df['time'].values
        elif 'timestamp' in df.columns:
            time = df['timestamp'].values - df['timestamp'].values[0]
        else:
            time = np.arange(len(df))
        
        # Row 0: PIC Temperature
        ax = axes[0]
        if 'temp_pic' in df.columns:
            ax.plot(time, df['temp_pic'], linewidth=2, label='temp_pic')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'{spec_name} - PIC Temperature vs. Time')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Row 1: LD Currents
        ax = axes[1]
        ld_columns = [col for col in df.columns if col.startswith('current_LD_')]
        if ld_columns:
            for i, col in enumerate(ld_columns):
                linestyle = '-' if i < 8 else '--'
                ax.plot(time, df[col], linewidth=1.5, linestyle=linestyle, 
                       label=f'LD_{i}', alpha=0.8)
            ax.set_ylabel('Current (mA)')
            ax.set_title(f'{spec_name} - LD Currents vs. Time')
            ax.legend(loc='best', fontsize=7, ncol=4)
            ax.grid(True, alpha=0.3)
        
        # Row 2: VOA Currents
        ax = axes[2]
        voa_columns = [col for col in df.columns if col.startswith('current_VOA_')]
        if voa_columns:
            for i, col in enumerate(voa_columns):
                linestyle = '-' if i < 8 else '--'
                ax.plot(time, df[col], linewidth=1.5, linestyle=linestyle,
                       label=f'VOA_{i}', alpha=0.8)
            ax.set_ylabel('Current (mA)')
            ax.set_title(f'{spec_name} - VOA Currents vs. Time')
            ax.legend(loc='best', fontsize=7, ncol=4)
            ax.grid(True, alpha=0.3)
        
        # Row 3: Calculated Laser Power (from PIC MPD)
        ax = axes[3]
        power_pic_columns = [col for col in df.columns if col.startswith('power_PIC_')]
        if power_pic_columns:
            # Convert uW to dBm
            for i, col in enumerate(power_pic_columns):
                linestyle = '-' if i < 8 else '--'
                # Convert uW to dBm: P(dBm) = 10*log10(P(uW)/1000)
                power_uw = df[col]
                power_dbm = 10 * np.log10(np.maximum(power_uw / 1000.0, 1e-10))
                ax.plot(time, power_dbm, linewidth=1.5, linestyle=linestyle,
                       label=f'Laser_{i}', alpha=0.8)
            
            # Add specification limits
            spec_lower = spec_name.lower()
            if self.specifications and spec_lower in self.specifications:
                spec_data = self.specifications[spec_lower]
                
                # Add min/max power limits (already in dBm)
                if 'min_power' in spec_data:
                    min_power_dbm = spec_data['min_power']['value']
                    ax.axhline(y=min_power_dbm, color='red', linestyle='--', linewidth=2.0, 
                              label=f'Min: {min_power_dbm}dBm', alpha=0.8)
                
                if 'max_power' in spec_data:
                    max_power_dbm = spec_data['max_power']['value']
                    ax.axhline(y=max_power_dbm, color='red', linestyle='--', linewidth=2.0,
                              label=f'Max: {max_power_dbm}dBm', alpha=0.8)
                
                # Add typical power if available
                if 'typical_power' in spec_data:
                    typ_power_dbm = spec_data['typical_power']['value']
                    ax.axhline(y=typ_power_dbm, color='green', linestyle=':', linewidth=2.0,
                              label=f'Typ: {typ_power_dbm}dBm', alpha=0.8)
            
            ax.set_ylabel('Power in fiber (dBm)')
            ax.set_title(f'{spec_name} - Calculated Laser Power vs. Time')
            ax.legend(loc='best', fontsize=6, ncol=4)
            ax.grid(True, alpha=0.3)
        
        # Row 4: Frequency Offset
        ax = axes[4]
        freq_columns = [col for col in df.columns if col.startswith('frequency_')]
        if freq_columns and len(freq_columns) > 0:
            # Calculate frequency offset from reference (assuming first 8 are set_a, next 8 are set_b)
            for i, col in enumerate(freq_columns):
                linestyle = '-' if i < 8 else '--'
                
                # Get reference frequency from grid
                channel = i % 8
                set_name = 'set_a' if i < 8 else 'set_b'
                grid_key = f'grid_{channel + 1}'
                
                if self.reference_grid and set_name in self.reference_grid and grid_key in self.reference_grid[set_name]:
                    ref_freq_thz = self.reference_grid[set_name][grid_key]['frequency_thz']
                    # Convert to GHz offset
                    freq_offset = (df[col] / 1000.0 - ref_freq_thz) * 1000.0
                    ax.plot(time, freq_offset, linewidth=1.5, linestyle=linestyle,
                           label=f'Freq_{i}', alpha=0.8)
            
            # Add specification limits for wavelength error
            spec_lower = spec_name.lower()
            if self.specifications and spec_lower in self.specifications:
                spec_data = self.specifications[spec_lower]
                
                if 'wavelength_error' in spec_data:
                    freq_error_limit = spec_data['wavelength_error']['value']
                    # Add shaded region for allowed frequency error
                    ax.axhspan(-freq_error_limit, freq_error_limit, 
                              color='green', alpha=0.1, label=f'Spec: ±{freq_error_limit}GHz')
                    # Add limit lines
                    ax.axhline(y=freq_error_limit, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
                    ax.axhline(y=-freq_error_limit, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.7)
            
            ax.set_ylabel('Frequency (GHz offset)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{spec_name} - Frequency Offset vs. Time')
            ax.legend(loc='best', fontsize=6, ncol=4)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    def _load_smsr_data(self, module_path, spec_type):
        """Load SMSR data from 'Power and SMSR' or 'SMSR and Power' tab"""
        # Find the Excel file - try multiple naming patterns
        if spec_type.lower() == 'endeavour':
            excel_files = (list(module_path.glob("*Endeavour*EVT*.xlsx")) + 
                          list(module_path.glob("*Endevour*EVT*.xlsx")) +
                          list(module_path.glob("EVT*Endeavour*.xlsx")) +
                          list(module_path.glob("EVT*Endevour*.xlsx")))
        else:
            excel_files = (list(module_path.glob("*Kenya*EVT*.xlsx")) +
                          list(module_path.glob("EVT*Kenya*.xlsx")))
        
        if not excel_files:
            return None
        
        excel_file = excel_files[0]
        
        # Get all sheet names and do case-insensitive matching
        try:
            xls = pd.ExcelFile(excel_file)
            actual_sheets = xls.sheet_names
            
            # Try different sheet name patterns (case-insensitive)
            target_patterns = ['power and smsr', 'smsr and power']
            
            for pattern in target_patterns:
                for actual_sheet in actual_sheets:
                    if actual_sheet.lower() == pattern:
                        df = pd.read_excel(excel_file, sheet_name=actual_sheet)
                        return df
        except Exception as e:
            print(f"    Warning: Could not load SMSR data from {excel_file.name}: {e}")
            return None
        
        # Sheet not found with any name variation
        return None
    
    def create_calibration_setpoints_summary(self, modules):
        """
        Create summary plot of calibration setpoints for all modules.
        Similar to fru_summary.png but using P0_T3_setpoints.csv data.
        """
        print("\n" + "=" * 60)
        print("Creating Calibration Setpoints Summary Plot")
        print("=" * 60)
        
        # Collect setpoints data for all modules
        all_setpoints_data = {}
        
        for module in modules:
            module_path = self.base_path / module
            raw_log_path = module_path / "Raw log data"
            
            if not raw_log_path.exists():
                continue
            
            # Get full serial number
            config_files = list(module_path.glob("*config*.yaml"))
            full_sn = module
            if config_files:
                parts = config_files[0].name.split('_')
                if len(parts) >= 3:
                    full_sn = parts[2]
            
            print(f"  Loading setpoints for module {full_sn}")
            
            # Load setpoints for both specs
            endeavour_setpoints = self._load_setpoints_data(raw_log_path, full_sn, 'endeavour')
            kenya_setpoints = self._load_setpoints_data(raw_log_path, full_sn, 'kenya')
            
            if endeavour_setpoints is not None or kenya_setpoints is not None:
                all_setpoints_data[full_sn] = {
                    'endeavour': endeavour_setpoints,
                    'kenya': kenya_setpoints
                }
        
        if not all_setpoints_data:
            print("  No setpoints data found for any module")
            return
        
        # Create 4x2 subplot (4 rows: LD current, PIC responsivity, MUX responsivity, power target)
        fig, axes = plt.subplots(4, 2, figsize=(18, 16))
        
        # Plot Endeavour column (left)
        self._plot_setpoints_summary_column(axes[:, 0], all_setpoints_data, 'endeavour', 'Endeavour')
        
        # Plot Kenya column (right)
        self._plot_setpoints_summary_column(axes[:, 1], all_setpoints_data, 'kenya', 'Kenya')
        
        plt.suptitle('Calibration Setpoints Summary - All Modules', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to analysis_results root folder
        plot_filename = 'calibration_setpoints_summary.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Calibration setpoints summary saved: {plot_filename}")
    
    def _load_setpoints_data(self, raw_log_path, full_sn, spec_type):
        """Load setpoints CSV file from subdirectories"""
        subdir_pattern = f"logs_{spec_type}_{full_sn}"
        subdir_path = raw_log_path / subdir_pattern
        
        if not subdir_path.exists():
            return None
        
        # Look for P0_T3_setpoints.csv in timestamp subdirectories
        setpoint_files = list(subdir_path.glob("*/P0_T3_setpoints.csv"))
        
        if not setpoint_files:
            return None
        
        # Use the first setpoints file found
        setpoint_file = setpoint_files[0]
        
        try:
            df = pd.read_csv(setpoint_file)
            # Skip empty rows
            df = df.dropna(how='all')
            if len(df) == 0:
                return None
            return df
        except Exception as e:
            print(f"    Error loading {spec_type} setpoints: {e}")
            return None
    
    def _plot_setpoints_summary_column(self, axes, all_data, spec_type, spec_name):
        """Plot one column of setpoints summary (4 parameters)"""
        
        # Collect data for each parameter
        all_sns = []
        ld_current_data = {ch: [] for ch in range(16)}
        pic_responsivity_data = {ch: [] for ch in range(16)}
        mux_responsivity_data = {ch: [] for ch in range(16)}
        power_target_data = {ch: [] for ch in range(16)}
        
        for full_sn, data in sorted(all_data.items()):
            if data[spec_type] is None:
                continue
            
            df = data[spec_type]
            all_sns.append(full_sn)
            
            # Extract data for each channel
            for ch in range(16):
                # LD current
                ld_col = f'current_LD_{ch}'
                if ld_col in df.columns:
                    val = df[ld_col].iloc[0]
                    if pd.notna(val):
                        ld_current_data[ch].append(val)
                
                # PIC tap ratio (responsivity)
                pic_col = f'tap_ratio_PIC_{ch}'
                if pic_col in df.columns:
                    val = df[pic_col].iloc[0]
                    if pd.notna(val):
                        pic_responsivity_data[ch].append(val)
                
                # MUX tap ratio (responsivity)
                mux_col = f'tap_ratio_MUX_{ch}'
                if mux_col in df.columns:
                    val = df[mux_col].iloc[0]
                    if pd.notna(val):
                        mux_responsivity_data[ch].append(val)
                
                # Power target
                power_col = f'power_target_{ch}'
                if power_col in df.columns:
                    val = df[power_col].iloc[0]
                    if pd.notna(val):
                        power_target_data[ch].append(val)
        
        if not all_sns:
            for ax in axes:
                ax.text(0.5, 0.5, f'No {spec_name} Data', ha='center', va='center', fontsize=14)
            return
        
        # Plot Row 0: LD Current
        self._plot_setpoints_parameter(axes[0], all_sns, ld_current_data, 
                                       f'{spec_name} - Laser Current Setpoints',
                                       'Laser Current (mA)')
        
        # Plot Row 1: PIC Responsivity
        self._plot_setpoints_parameter(axes[1], all_sns, pic_responsivity_data,
                                       f'{spec_name} - PIC MPD Responsivity',
                                       'PIC MPD Responsivity')
        
        # Plot Row 2: MUX Responsivity
        self._plot_setpoints_parameter(axes[2], all_sns, mux_responsivity_data,
                                       f'{spec_name} - MUX MPD Responsivity',
                                       'MUX MPD Responsivity')
        
        # Plot Row 3: Power Target
        self._plot_setpoints_parameter(axes[3], all_sns, power_target_data,
                                       f'{spec_name} - Power Target',
                                       'Power Target (mW)')
    
    def _plot_setpoints_parameter(self, ax, all_sns, channel_data, title, ylabel):
        """Plot a single setpoints parameter with boxplot and scatter"""
        
        # Prepare data for boxplot
        positions = []
        box_data = []
        
        for i, sn in enumerate(all_sns):
            pos_base = i * 1.0
            
            # Collect all channel data for this SN
            all_values = []
            for ch in range(16):
                if len(channel_data[ch]) > i:
                    all_values.append(channel_data[ch][i])
            
            if all_values:
                positions.append(pos_base)
                box_data.append(all_values)
        
        # Create boxplot
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=False)
            
            # Color boxplots blue
            for patch in bp['boxes']:
                patch.set_facecolor('skyblue')
                patch.set_alpha(0.5)
            
            # Overlay scatter points for individual channels
            for i, sn in enumerate(all_sns):
                pos_base = i * 1.0
                
                for ch in range(16):
                    if len(channel_data[ch]) > i:
                        val = channel_data[ch][i]
                        # Slight jitter for visibility
                        jitter = (ch - 7.5) * 0.01
                        ax.scatter(pos_base + jitter, val, c='blue', 
                                 alpha=0.6, s=30, zorder=3)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(all_sns)))
        ax.set_xticklabels(all_sns, rotation=45, ha='right')
        ax.set_xlabel('Tile SN')
        ax.grid(True, alpha=0.3, axis='y')


class temperature_aggressors_1:
    """
    Temperature Aggressors Test 1 Analysis
    
    This class analyzes temperature cycling data for a single tile, including:
    - Temperature profile vs time
    - Optical wavelength data correlation with temperature
    """
    
    def __init__(self, base_path):
        """Initialize temperature aggressors test 1 analysis."""
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "temperature_aggressors"
        self.results_path = self.base_path / "analysis_results" / "temperature_aggressors"
        self.test1_path = self.results_path / "aggressors_test_1"
        
        # Create directories
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.test1_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.temp_log_file = self.data_path / "temperature_log_20251009_One tile data with Temp Cycle.csv"
        self.excel_file = self.data_path / "One tile data with Temp Cycle.xlsx"
        
        print("=" * 80)
        print("Temperature Aggressors Test 1 - Analysis")
        print("=" * 80)
        print(f"Data path: {self.data_path}")
        print(f"Results path: {self.test1_path}")
        print()
    
    def load_temperature_data(self):
        """Load temperature log CSV data."""
        if not self.temp_log_file.exists():
            print(f"Error: Temperature log file not found: {self.temp_log_file}")
            return None
        
        # Read CSV
        df = pd.read_csv(self.temp_log_file)
        print(f"Loaded temperature data: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Parse timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Calculate elapsed time in seconds from start
        df['Time_seconds'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
        
        return df
    
    def plot_temperature_profile(self):
        """Plot temperature vs time."""
        print("Plotting temperature profile...")
        
        # Load data
        df = self.load_temperature_data()
        if df is None:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot temperature vs time
        ax.plot(df['Time_seconds'], df['Temperature_C'], 
               linewidth=1.5, color='red', label='Temperature')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Profile vs Time\nTemperature Cycling Test', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = 'temperature_profile.png'
        plt.savefig(self.test1_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
        print(f"  Temperature range: {df['Temperature_C'].min():.2f}°C to {df['Temperature_C'].max():.2f}°C")
        print(f"  Total duration: {df['Time_seconds'].max():.0f} seconds ({df['Time_seconds'].max()/60:.1f} minutes)")
        print()
    
    def load_excel_data(self):
        """Load and parse Excel file with time-series data."""
        import ast
        
        if not self.excel_file.exists():
            print(f"Error: Excel file not found: {self.excel_file}")
            return None
        
        print("Loading Excel data...")
        df = pd.read_excel(self.excel_file)
        print(f"  Loaded {len(df)} channels of data")
        
        # Parse the string arrays into lists
        for col in ['wavelength', 'power', 'timestamp', 'laser_dac', 'voa_dac', 
                    'tpic', 'tmux', 'tpmic', 'mpd_mux', 'mpd_pic']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        return df
    
    def align_timestamps(self, excel_df, temp_df=None):
        """Align timestamps using Excel file as reference."""
        # Get reference start time from first channel's first timestamp
        first_timestamps = excel_df.iloc[0]['timestamp']
        parsed_first = pd.to_datetime(first_timestamps)
        ref_start = parsed_first[0]
        
        # Process each channel's timestamps
        time_seconds_list = []
        for idx, row in excel_df.iterrows():
            timestamps = row['timestamp']
            # Parse timestamps and calculate seconds from first timestamp
            parsed_times = pd.to_datetime(timestamps)
            time_seconds = (parsed_times - ref_start).total_seconds().values
            time_seconds_list.append(time_seconds)
        
        # Add as a new column
        excel_df['time_seconds'] = time_seconds_list
        
        # If temp_df is provided, align it to the same reference
        if temp_df is not None:
            temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        return excel_df
    
    def plot_missionmode_power(self):
        """Plot optical power vs time for all channels with temperature overlay."""
        print("Plotting mission mode power...")
        
        # Load data
        temp_df = self.load_temperature_data()
        excel_df = self.load_excel_data()
        if temp_df is None or excel_df is None:
            return
        
        # Align timestamps
        excel_df = self.align_timestamps(excel_df, temp_df)
        
        # Calculate max temperature rate of change
        temp_diff = temp_df['Temperature_C'].diff()
        time_diff = temp_df['Time_seconds'].diff()
        temp_rate = (temp_diff / time_diff).abs()
        max_temp_rate = temp_rate.max()
        
        # Create figure with single subplot for both banks
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax_temp = ax.twinx()
        
        # Color maps
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Plot both banks on same subplot
        for bank in [0, 1]:
            bank_data = excel_df[excel_df['bank'] == bank]
            for idx, row in bank_data.iterrows():
                channel = row['channel']
                time_sec = row['time_seconds']
                # Use mpd_pic (in µW) and convert to dBm
                power_uw = np.array(row['mpd_pic'])
                power_dbm = 10 * np.log10(power_uw / 1000.0)
                
                color = colors_b0[channel] if bank == 0 else colors_b1[channel]
                bank_label = f'B{bank}-Ch{channel}'
                ax.plot(time_sec, power_dbm, linewidth=1.5, color=color, 
                       label=bank_label, alpha=0.8)
        
        # Plot temperature on secondary axis
        ax_temp.plot(temp_df['Time_seconds'], temp_df['Temperature_C'],
                    'r--', linewidth=2, alpha=0.5, 
                    label=f'Temperature (max rate: {max_temp_rate:.2f}°C/s)')
        
        # Configure axes
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Power in fiber (dBm)', fontsize=11)
        ax.set_xlim(0, 2500)
        ax.set_ylim(0, 10)
        ax_temp.set_ylabel('Temperature (°C)', fontsize=11, color='red')
        ax_temp.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('Optical Power vs Time\nwith Temperature Cycling',
                    fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=7, ncol=8)
        ax_temp.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = 'missionmode_power_temptest.png'
        plt.savefig(self.test1_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
        print()
    
    def plot_missionmode_freqerror(self):
        """Plot frequency error vs time for all channels with temperature overlay."""
        print("Plotting mission mode frequency error...")
        
        # Load data
        temp_df = self.load_temperature_data()
        excel_df = self.load_excel_data()
        if temp_df is None or excel_df is None:
            return
        
        # Align timestamps
        excel_df = self.align_timestamps(excel_df, temp_df)
        
        # Calculate max temperature rate of change
        temp_diff = temp_df['Temperature_C'].diff()
        time_diff = temp_df['Time_seconds'].diff()
        temp_rate = (temp_diff / time_diff).abs()
        max_temp_rate = temp_rate.max()
        
        # Calculate frequency error from wavelength
        c_speed_light = 299792.458  # Speed of light in THz*nm
        
        # Create figure with single subplot for both banks
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax_temp = ax.twinx()
        
        # Color maps
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Plot both banks on same subplot
        for bank in [0, 1]:
            bank_data = excel_df[excel_df['bank'] == bank]
            for idx, row in bank_data.iterrows():
                channel = row['channel']
                time_sec = row['time_seconds']
                
                # Calculate frequency error if wavelength data exists
                wavelengths = np.array(row['wavelength'])
                # Use settled wavelength as reference
                ref_wavelength = row['settled_wavelength_nm']
                
                # Calculate frequency error in GHz
                measured_freq_thz = c_speed_light / wavelengths
                ref_freq_thz = c_speed_light / ref_wavelength
                freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                
                color = colors_b0[channel] if bank == 0 else colors_b1[channel]
                bank_label = f'B{bank}-Ch{channel}'
                ax.plot(time_sec, freq_error_ghz, linewidth=1.5, color=color,
                       label=bank_label, alpha=0.8)
        
        # Plot temperature on secondary axis
        ax_temp.plot(temp_df['Time_seconds'], temp_df['Temperature_C'],
                    'r--', linewidth=2, alpha=0.5,
                    label=f'Temperature (max rate: {max_temp_rate:.2f}°C/s)')
        
        # Configure axes
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Frequency Error (GHz)', fontsize=11)
        ax.set_xlim(0, 2500)
        ax_temp.set_ylabel('Temperature (°C)', fontsize=11, color='red')
        ax_temp.tick_params(axis='y', labelcolor='red')
        
        # Add spec limits (±20 GHz)
        ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhspan(-20, 20, color='green', alpha=0.05)
        
        ax.set_title('Frequency Error vs Time\nwith Temperature Cycling',
                    fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=7, ncol=8)
        ax_temp.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = 'missionmode_freqerror_temptest.png'
        plt.savefig(self.test1_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
        print()
    
    def plot_missionmode_operatingpoints(self):
        """Plot operating points (DACs, temperatures) vs time with temperature overlay."""
        print("Plotting mission mode operating points...")
        
        # Load data
        temp_df = self.load_temperature_data()
        excel_df = self.load_excel_data()
        if temp_df is None or excel_df is None:
            return
        
        # Align timestamps
        excel_df = self.align_timestamps(excel_df, temp_df)
        
        # Calculate max temperature rate of change
        temp_diff = temp_df['Temperature_C'].diff()
        time_diff = temp_df['Time_seconds'].diff()
        temp_rate = (temp_diff / time_diff).abs()
        max_temp_rate = temp_rate.max()
        
        # Create figure with 5 subplots (one for each operating point)
        # Parameters to plot: laser_dac, voa_dac, tpic, tmux, tpmic
        params = ['laser_dac', 'voa_dac', 'tpic', 'tmux', 'tpmic']
        param_labels = ['Laser DAC', 'VOA DAC', 'TPIC (°C)', 'TMUX (°C)', 'TPMIC (°C)']
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 18))
        
        # Color maps
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        for param_idx, (param, param_label) in enumerate(zip(params, param_labels)):
            ax = axes[param_idx]
            ax_temp = ax.twinx()
            
            # Plot both banks on the same subplot
            for bank in [0, 1]:
                bank_data = excel_df[excel_df['bank'] == bank]
                
                for idx, row in bank_data.iterrows():
                    channel = row['channel']
                    time_sec = row['time_seconds']
                    
                    # Get parameter values
                    param_values = np.array(row[param])
                    
                    color = colors_b0[channel] if bank == 0 else colors_b1[channel]
                    bank_label = f'B{bank}-Ch{channel}'
                    
                    ax.plot(time_sec, param_values, linewidth=1.5, color=color,
                           label=bank_label, alpha=0.8)
            
            # Plot temperature on secondary axis
            ax_temp.plot(temp_df['Time_seconds'], temp_df['Temperature_C'],
                        'r--', linewidth=2, alpha=0.5,
                        label=f'Temperature (max rate: {max_temp_rate:.2f}°C/s)')
            
            # Configure axes
            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.set_ylabel(param_label, fontsize=11)
            ax.set_xlim(0, 2500)
            ax_temp.set_ylabel('Temperature (°C)', fontsize=11, color='red')
            ax_temp.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'{param_label} vs Time\nwith Temperature Cycling',
                        fontsize=12, fontweight='bold')
            
            # Legend only for first subplot to avoid cluttering
            if param_idx == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=8)
                ax_temp.legend(loc='upper right', fontsize=8)
            else:
                ax_temp.legend(loc='upper right', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = 'missionmode_operatingpoints_temptest.png'
        plt.savefig(self.test1_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
        print()
    
    def run_all_plots(self):
        """Generate all mission mode plots."""
        print("\n" + "=" * 80)
        print("GENERATING ALL MISSION MODE PLOTS")
        print("=" * 80 + "\n")
        
        self.plot_temperature_profile()
        self.plot_missionmode_power()
        self.plot_missionmode_freqerror()
        self.plot_missionmode_operatingpoints()
        
        print("=" * 80)
        print("All plots completed!")
        print(f"Results saved to: {self.test1_path}")
        print("=" * 80)


class temperature_aggressors_2:
    """
    Temperature Aggressors Test 2 Analysis
    
    Analyzes optical wavemeter data for multiple tiles during temperature cycling.
    """
    
    def __init__(self, base_path):
        """Initialize temperature aggressors test 2 analysis."""
        import ast
        
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "temperature_aggressors"
        self.results_path = self.base_path / "analysis_results" / "temperature_aggressors"
        self.test2_path = self.results_path / "aggressors_test_2"
        
        # Create directories
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.test2_path.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.wavemeter_file = self.data_path / "optical_wavemeter_loop_20251010T213800367Z_e7390a1b-5b91-42b7-ad23-f00f82fc19a2.csv"
        self.temp_log_file1 = self.data_path / "temperature_log_20251013_004250.csv"
        self.temp_log_file2 = self.data_path / "temperature_log_20251014_150929.csv"
        
        print("=" * 80)
        print("Temperature Aggressors Test 2 - Analysis")
        print("=" * 80)
        print(f"Data path: {self.data_path}")
        print(f"Results path: {self.test2_path}")
        print()
    
    def load_wavemeter_data(self):
        """Load and parse optical wavemeter CSV data."""
        import ast
        
        if not self.wavemeter_file.exists():
            print(f"Error: Wavemeter file not found: {self.wavemeter_file}")
            return None
        
        print("Loading optical wavemeter data...")
        df = pd.read_csv(self.wavemeter_file)
        print(f"  Loaded {len(df)} rows of data")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate elapsed time from first timestamp
        ref_start = df['timestamp'].iloc[0]
        df['time_seconds'] = (df['timestamp'] - ref_start).dt.total_seconds()
        
        # Parse array columns
        for col in ['wavelength_nm', 'voa_dac_value', 'laser_dac_value', 
                    'mux_mpd_value', 'pic_mpd_value']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        print(f"  Tiles: {sorted(df['tile_id'].unique())}")
        print(f"  Cycles: {sorted(df['cycle_number'].unique())}")
        print(f"  Time range: {df['time_seconds'].min():.0f} to {df['time_seconds'].max():.0f} seconds")
        
        return df
    
    def subsample_to_hourly(self, df):
        """Subsample data to 10-minute intervals to reduce data volume."""
        print("\nSubsampling data to 10-minute intervals...")
        print(f"  Original data: {len(df)} rows")
        
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Group by tile and bank, then resample each group
        subsampled_groups = []
        for (tile_id, bank_type), group in df.groupby(['tile_id', 'bank_type']):
            # Resample to 10-minute intervals, taking the first sample in each interval
            resampled = group.resample('10T').first()  # '10T' means 10 minutes
            # Remove any NaN rows (intervals with no data)
            resampled = resampled.dropna(subset=['time_seconds'])
            subsampled_groups.append(resampled)
        
        # Concatenate all groups
        df_subsampled = pd.concat(subsampled_groups).reset_index()
        
        print(f"  Subsampled data: {len(df_subsampled)} rows")
        print(f"  Reduction: {len(df)/len(df_subsampled):.1f}x")
        
        return df_subsampled
    
    def load_temperature_data(self):
        """Load and concatenate temperature log CSV data from both files."""
        if not self.temp_log_file1.exists():
            print(f"Error: Temperature log file 1 not found: {self.temp_log_file1}")
            return None
        if not self.temp_log_file2.exists():
            print(f"Error: Temperature log file 2 not found: {self.temp_log_file2}")
            return None
        
        print("Loading temperature data from both log files...")
        # Read both CSV files
        df1 = pd.read_csv(self.temp_log_file1)
        df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
        
        df2 = pd.read_csv(self.temp_log_file2)
        df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
        
        # Concatenate and sort by timestamp
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"  Loaded {len(df)} temperature readings")
        print(f"  Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        
        return df
    
    def align_with_temperature(self, wavemeter_df, temp_df):
        """Align temperature data with wavemeter timestamps.
        
        Note: This aligns based on the first timestamp in wavemeter_df, which should 
        be the FULL dataset start time, not a filtered subset.
        """
        # Use wavemeter reference time (should be from full dataset)
        ref_start = wavemeter_df['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        return temp_df
    
    def plot_missionmode_power(self):
        """Plot optical power vs time for all tiles in a 4x4 grid."""
        print("Plotting mission mode power for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to hourly intervals
        wavemeter_df = self.subsample_to_hourly(wavemeter_df)
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Plot each tile in a separate subplot
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:  # Only plot up to 16 tiles
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Plot both banks
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                # Collect data for each channel to plot as lines
                channel_data = {i: {'time': [], 'power': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0  # Convert to hours
                    # Use pic_mpd_value (in µW) and convert to dBm
                    power_uw = np.array(row['pic_mpd_value'])
                    # Skip if first element (seems to be a placeholder)
                    if len(power_uw) > 1:
                        power_uw = power_uw[1:]  # Skip first element
                        power_dbm = 10 * np.log10(power_uw / 1000.0)
                        
                        # Store data for each channel
                        for ch_idx, p in enumerate(power_dbm):
                            if ch_idx < 8:  # Only 8 channels
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['power'].append(p)
                
                # Plot lines for each channel
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['power'],
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=2)
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Power in fiber (dBm)', fontsize=9)
            ax.set_ylim(9, 13)
            ax.set_xlim(0, 96)
            ax.set_xticks(np.arange(0, 97, 12))
            
            # Add Endeavour power specs (both in red with shaded region)
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Min (10 dBm)')
            ax.axhline(y=12.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Max (12.3 dBm)')
            ax.axhspan(10.0, 12.3, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=5, ncol=2)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Optical Power vs Time - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_power_all_tiles.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Combined plot saved: {plot_filename}")
        print(f"  Plotted {len(tile_ids)} tiles in 2x8 grid\n")
    
    def plot_temperature_profile(self):
        """Plot temperature vs time."""
        print("Plotting temperature profile...")
        
        # Load data
        temp_df = self.load_temperature_data()
        wavemeter_df = self.load_wavemeter_data()
        if temp_df is None or wavemeter_df is None:
            return
        
        # Align timestamps
        temp_df = self.align_with_temperature(wavemeter_df, temp_df)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot temperature vs time
        ax.plot(temp_df['Time_seconds'], temp_df['Temperature_C'], 
               linewidth=1.5, color='red', label='Temperature')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Profile vs Time\nTemperature Cycling Test', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = 'temperature_profile.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
        print(f"  Temperature range: {temp_df['Temperature_C'].min():.2f}°C to {temp_df['Temperature_C'].max():.2f}°C")
        print(f"  Total duration: {temp_df['Time_seconds'].max():.0f} seconds ({temp_df['Time_seconds'].max()/60:.1f} minutes)")
        print()
    
    def plot_missionmode_freqerror(self):
        """Plot frequency error vs time for all tiles in a 4x4 grid."""
        print("Plotting mission mode frequency error for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to hourly intervals
        wavemeter_df = self.subsample_to_hourly(wavemeter_df)
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Plot each tile in a separate subplot
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:  # Only plot up to 16 tiles
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Load reference wavelengths (using first cycle as reference)
            # Note: wavelength_nm column contains wavelength data off by 12 orders of magnitude  
            # DIVIDE by 1e9 to convert to nm (raw values are ~1.3e12, should be ~1300 nm)
            ref_wavelengths = {}
            cycle_0_data = tile_data[tile_data['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]  # Skip first element
                    # Filter out invalid wavelength data (first element is 8e9, rest are around 1.3e12)
                    # Keep values > 1e12 (actual wavelength/frequency data)
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        # Convert to nm by DIVIDING by 1e9 (raw ~1.3e12 -> ~1300 nm)
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths[bank_type] = wavelengths_nm
            
            # Speed of light constant
            c_speed_light = 299792.458  # THz*nm
            
            # Plot both banks
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                # Collect data for each channel to plot as lines
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0  # Convert to hours
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]  # Skip first element
                        
                        # Filter out invalid wavelength data (keep values > 1e12)
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        # Convert to nm by DIVIDING by 1e9 (raw ~1.3e12 -> ~1300 nm)
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        # Additional filtering: remove unrealistic wavelengths (should be 1200-1400 nm)
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        # Ensure wavelengths and ref_wl have same length
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        # Calculate frequency error in GHz
                        # freq (THz) = c / wavelength_nm
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                        
                        # Filter out frequency errors beyond ±100 GHz (removes outliers)
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        
                        # Store data for each channel
                        for ch_idx, (f, valid) in enumerate(zip(freq_error_ghz, valid_freq_mask)):
                            if ch_idx < 8 and valid:  # Only 8 channels, filter outliers
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(f)
                
                # Plot lines for each channel
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['freq_error'],
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=2)
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=9)
            ax.set_ylim(-50, 50)
            ax.set_xlim(0, 96)
            ax.set_xticks(np.arange(0, 97, 12))
            
            # Add spec limits (±20 GHz)
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=6, ncol=2)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Frequency Error vs Time - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_freqerror_all_tiles.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Combined plot saved: {plot_filename}")
        print(f"  Plotted {len(tile_ids)} tiles in 2x8 grid\n")
    
    def plot_missionmode_operatingpoints(self):
        """Plot operating points vs time for all tiles."""
        print("Plotting mission mode operating points for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to hourly intervals
        wavemeter_df = self.subsample_to_hourly(wavemeter_df)
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Create figure with 5 subplots (one for each operating point)
        params = [
            ('laser_dac_value', 'Laser DAC'),
            ('voa_dac_value', 'VOA DAC'),
            ('temp_pic_C', 'TPIC (°C)'),
            ('temp_mux_C', 'TMUX (°C)'),
            ('temp_pmic_C', 'TPMIC (°C)')
        ]
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Plot each tile
        for tile_id in tile_ids:
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            fig, axes = plt.subplots(5, 1, figsize=(16, 18))
            
            for param_idx, (param_col, param_label) in enumerate(params):
                ax = axes[param_idx]
                
                # Plot both banks
                for bank_type in ['BANK_A', 'BANK_B']:
                    bank_data = tile_data[tile_data['bank_type'] == bank_type]
                    
                    colors = colors_a if bank_type == 'BANK_A' else colors_b
                    bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                    
                    # Handle scalar vs array parameters
                    if param_col.startswith('temp_'):
                        # Temperature sensors are scalar values - collect as time series
                        time_list = []
                        value_list = []
                        for idx, row in bank_data.iterrows():
                            time_hours = row['time_seconds'] / 3600.0
                            param_value = row[param_col]
                            time_list.append(time_hours)
                            value_list.append(param_value)
                        
                        if len(time_list) > 0:
                            ax.plot(time_list, value_list, color=colors[0], linewidth=1.5, alpha=0.7,
                                   label=f'B{bank_label}', marker='o', markersize=3)
                    else:
                        # DAC values are arrays - collect data for each channel
                        channel_data = {i: {'time': [], 'value': []} for i in range(8)}
                        
                        for idx, row in bank_data.iterrows():
                            time_hours = row['time_seconds'] / 3600.0
                            param_values = np.array(row[param_col])
                            if len(param_values) > 1:
                                param_values = param_values[1:]  # Skip first element
                                
                                for ch_idx, p in enumerate(param_values):
                                    if ch_idx < 8:
                                        channel_data[ch_idx]['time'].append(time_hours)
                                        channel_data[ch_idx]['value'].append(p)
                        
                        # Plot lines for each channel
                        for ch_idx in range(8):
                            if len(channel_data[ch_idx]['time']) > 0:
                                ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['value'],
                                       color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                                       label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=2)
                
                # Configure axes
                ax.set_xlabel('Time (hours)', fontsize=11)
                ax.set_ylabel(param_label, fontsize=11)
                
                ax.set_title(f'Tile {tile_id} - {param_label} vs Time',
                            fontsize=12, fontweight='bold')
                
                # Legend only for first subplot to avoid cluttering
                if param_idx == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=7, ncol=8)
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f'missionmode_operatingpoints_tile{tile_id}_temptest.png'
            plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Tile {tile_id}: {plot_filename}")
        
        print(f"\nCompleted operating points plots for {len(tile_ids)} tiles\n")
    
    def plot_missionmode_power_zoomed(self):
        """Plot optical power vs time for all tiles in a 4x4 grid, zoomed to 46-48 hours with temperature overlay."""
        print("Plotting zoomed mission mode power (46-48 hr) with temperature overlay...")
        
        # Load data (without subsampling for zoomed view)
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Align temperature data using FULL wavemeter dataset (before filtering)
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        # Filter data to 46-48 hour window AFTER alignment
        time_min = 46 * 3600  # 46 hours in seconds
        time_max = 48 * 3600  # 48 hours in seconds
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                           (wavemeter_df_full['time_seconds'] <= time_max)]
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Plot each tile
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            ax2 = ax.twinx()  # Create second y-axis for temperature
            
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Plot both banks
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                channel_data = {i: {'time': [], 'power': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    power_uw = np.array(row['pic_mpd_value'])
                    if len(power_uw) > 1:
                        power_uw = power_uw[1:]
                        power_dbm = 10 * np.log10(power_uw / 1000.0)
                        
                        for ch_idx, p in enumerate(power_dbm):
                            if ch_idx < 8:
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['power'].append(p)
                
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['power'],
                               color=colors[ch_idx], linewidth=0.8, alpha=0.7,
                               label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=1.5)
            
            # Plot temperature on second y-axis
            temp_hours = temp_df['Time_seconds'].values / 3600.0
            temp_values = temp_df['Temperature_C'].values
            ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6, 
                    linestyle='--', label='Case Temp')
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Power in fiber (dBm)', fontsize=9, color='black')
            ax2.set_ylabel('Temperature (°C)', fontsize=9, color='red')
            ax.set_ylim(9, 13)
            ax.set_xlim(46, 48)
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add Endeavour power specs (both in red with shaded region)
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Min (10 dBm)')
            ax.axhline(y=12.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Max (12.3 dBm)')
            ax.axhspan(10.0, 12.3, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1, lines1))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=4, ncol=2)
            ax2.legend(lines2, labels2, loc='upper right', fontsize=6)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Optical Power vs Time (46-48 hr zoom) - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_power_all_tiles_zoomed.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Zoomed plot saved: {plot_filename}\n")
    
    def plot_missionmode_freqerror_zoomed(self):
        """Plot frequency error vs time for all tiles in a 4x4 grid, zoomed to 46-48 hours with temperature overlay."""
        print("Plotting zoomed mission mode frequency error (46-48 hr) with temperature overlay...")
        
        # Load data (without subsampling)
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Load reference wavelengths from FULL dataset (cycle 0) for all tiles
        print("  Loading reference wavelengths from cycle 0...")
        ref_wavelengths_all = {}
        for tile_id in sorted(wavemeter_df_full['tile_id'].unique()):
            ref_wavelengths_all[tile_id] = {}
            tile_data_full = wavemeter_df_full[wavemeter_df_full['tile_id'] == tile_id]
            cycle_0_data = tile_data_full[tile_data_full['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths_all[tile_id][bank_type] = wavelengths_nm
        
        # Filter data to 46-48 hour window AFTER loading references
        time_min = 46 * 3600
        time_max = 48 * 3600
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                           (wavemeter_df_full['time_seconds'] <= time_max)]
        
        # Align temperature data using the FULL wavemeter dataset start time
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        c_speed_light = 299792.458
        
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            ax2 = ax.twinx()
            
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Get reference wavelengths for this tile from pre-loaded data
            ref_wavelengths = ref_wavelengths_all.get(tile_id, {})
            
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        # Additional filtering: remove unrealistic wavelengths (should be 1200-1400 nm)
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                        
                        # Filter out frequency errors beyond ±100 GHz (removes outliers)
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        
                        # Store data for each channel
                        for ch_idx, (f, valid) in enumerate(zip(freq_error_ghz, valid_freq_mask)):
                            if ch_idx < 8 and valid:  # Filter outliers
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(f)
                
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['freq_error'],
                               color=colors[ch_idx], linewidth=0.8, alpha=0.7,
                               label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=1.5)
            
            # Plot temperature on second y-axis
            temp_hours = temp_df['Time_seconds'].values / 3600.0
            temp_values = temp_df['Temperature_C'].values
            ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6,
                    linestyle='--', label='Case Temp')
            
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=9, color='black')
            ax2.set_ylabel('Temperature (°C)', fontsize=9, color='red')
            ax.set_ylim(-50, 50)
            ax.set_xlim(46, 48)
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1, lines1))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=5, ncol=2)
            ax2.legend(lines2, labels2, loc='upper right', fontsize=6)
        
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Frequency Error vs Time (46-48 hr zoom) - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_freqerror_all_tiles_zoomed.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Zoomed plot saved: {plot_filename}\n")
    
    def plot_missionmode_operatingpoints_zoomed(self):
        """Plot operating points vs time for all tiles, zoomed to 46-48 hours with temperature overlay."""
        print("Plotting zoomed mission mode operating points (46-48 hr) for all tiles...")
        
        # Load data
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Align temperature data using FULL wavemeter dataset
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        # Filter to 46-48 hour window AFTER alignment
        time_min = 46 * 3600
        time_max = 48 * 3600
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                           (wavemeter_df_full['time_seconds'] <= time_max)]
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        params = [
            ('laser_dac_value', 'Laser DAC'),
            ('voa_dac_value', 'VOA DAC'),
            ('temp_pic_C', 'TPIC (°C)'),
            ('temp_mux_C', 'TMUX (°C)'),
            ('temp_pmic_C', 'TPMIC (°C)')
        ]
        
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        for tile_id in tile_ids:
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            fig, axes = plt.subplots(5, 1, figsize=(16, 18))
            
            for param_idx, (param_col, param_label) in enumerate(params):
                ax = axes[param_idx]
                ax2 = ax.twinx()
                
                for bank_type in ['BANK_A', 'BANK_B']:
                    bank_data = tile_data[tile_data['bank_type'] == bank_type]
                    
                    colors = colors_a if bank_type == 'BANK_A' else colors_b
                    bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                    
                    if param_col.startswith('temp_'):
                        time_list = []
                        value_list = []
                        for idx, row in bank_data.iterrows():
                            time_hours = row['time_seconds'] / 3600.0
                            param_value = row[param_col]
                            time_list.append(time_hours)
                            value_list.append(param_value)
                        
                        if len(time_list) > 0:
                            ax.plot(time_list, value_list, color=colors[0], linewidth=1.2, alpha=0.7,
                                   label=f'B{bank_label}', marker='o', markersize=2)
                    else:
                        channel_data = {i: {'time': [], 'value': []} for i in range(8)}
                        
                        for idx, row in bank_data.iterrows():
                            time_hours = row['time_seconds'] / 3600.0
                            param_values = np.array(row[param_col])
                            if len(param_values) > 1:
                                param_values = param_values[1:]
                                
                                for ch_idx, p in enumerate(param_values):
                                    if ch_idx < 8:
                                        channel_data[ch_idx]['time'].append(time_hours)
                                        channel_data[ch_idx]['value'].append(p)
                        
                        for ch_idx in range(8):
                            if len(channel_data[ch_idx]['time']) > 0:
                                ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['value'],
                                       color=colors[ch_idx], linewidth=0.8, alpha=0.7,
                                       label=f'Set{bank_label}-Ch{ch_idx+1}', marker='o', markersize=1.5)
                
                # Plot temperature on second y-axis
                temp_hours = temp_df['Time_seconds'].values / 3600.0
                temp_values = temp_df['Temperature_C'].values
                ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6,
                        linestyle='--', label='Case Temp')
                
                ax.set_xlabel('Time (hours)', fontsize=11)
                ax.set_ylabel(param_label, fontsize=11, color='black')
                ax2.set_ylabel('Temperature (°C)', fontsize=11, color='red')
                ax.set_xlim(46, 48)
                ax2.tick_params(axis='y', labelcolor='red')
                
                ax.set_title(f'Tile {tile_id} - {param_label} vs Time (46-48 hr zoom)',
                            fontsize=12, fontweight='bold')
                
                if param_idx == 0:
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    by_label = dict(zip(labels1, lines1))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=6, ncol=8)
                    ax2.legend(lines2, labels2, loc='upper right', fontsize=7)
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f'missionmode_operatingpoints_tile{tile_id}_temptest_zoomed.png'
            plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Tile {tile_id}: {plot_filename}")
        
        print(f"\nCompleted zoomed operating points plots for {len(tile_ids)} tiles\n")
    
    def subsample_to_5min(self, df):
        """Subsample data to 5-minute intervals for delta plots."""
        print("\nSubsampling data to 5-minute intervals...")
        print(f"  Original data: {len(df)} rows")
        
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Group by tile and bank, then resample each group
        subsampled_groups = []
        for (tile_id, bank_type), group in df.groupby(['tile_id', 'bank_type']):
            # Resample to 5-minute intervals, taking the first sample in each interval
            resampled = group.resample('5T').first()  # '5T' means 5 minutes
            # Remove any NaN rows (intervals with no data)
            resampled = resampled.dropna(subset=['time_seconds'])
            subsampled_groups.append(resampled)
        
        # Concatenate all groups
        df_subsampled = pd.concat(subsampled_groups).reset_index()
        
        print(f"  Subsampled data: {len(df_subsampled)} rows")
        print(f"  Reduction: {len(df)/len(df_subsampled):.1f}x")
        
        return df_subsampled
    
    def plot_missionmode_power_all_tiles_delta(self):
        """Plot optical power vs time for all tiles with linear fit delta in legend."""
        print("Plotting mission mode power with delta for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to 5-minute intervals
        wavemeter_df = self.subsample_to_5min(wavemeter_df)
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Plot each tile in a separate subplot
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:  # Only plot up to 16 tiles
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Plot both banks
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                # Collect data for each channel to plot as lines and calculate delta
                channel_data = {i: {'time': [], 'power': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0  # Convert to hours
                    # Use pic_mpd_value (in µW) and convert to dBm
                    power_uw = np.array(row['pic_mpd_value'])
                    
                    if len(power_uw) > 1:
                        power_uw = power_uw[1:]  # Skip first element
                        power_dbm = 10 * np.log10(power_uw / 1000.0)
                        
                        # Match channel indices
                        for ch_idx, power_val in enumerate(power_dbm[:8]):  # Only take first 8 channels
                            if not np.isnan(power_val) and not np.isinf(power_val):
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['power'].append(power_val)
                
                # Plot each channel and calculate delta
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        times = np.array(channel_data[ch_idx]['time'])
                        powers = np.array(channel_data[ch_idx]['power'])
                        
                        # Perform linear fit
                        if len(times) > 1:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(times, powers)
                            
                            # Calculate delta: difference between end and start of time range
                            t_start = times.min()
                            t_end = times.max()
                            power_start = slope * t_start + intercept
                            power_end = slope * t_end + intercept
                            delta = power_end - power_start
                            
                            label = f'Set{bank_label}-Ch{ch_idx+1} (Δ={delta:.2f} dB)'
                        else:
                            label = f'Set{bank_label}-Ch{ch_idx+1}'
                        
                        ax.plot(times, powers,
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               label=label, marker='o', markersize=2)
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Power in fiber (dBm)', fontsize=9)
            ax.set_ylim(9, 13)
            ax.set_xlim(0, 96)
            ax.set_xticks(np.arange(0, 97, 12))
            
            # Add Endeavour power specs (both in red with shaded region)
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Min (10 dBm)')
            ax.axhline(y=12.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Max (12.3 dBm)')
            ax.axhspan(10.0, 12.3, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=4.5, ncol=2)
        
        plt.tight_layout()
        plot_filename = 'missionmode_power_all_tiles_delta.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}\n")
    
    def plot_missionmode_freqerror_all_tiles_delta(self):
        """Plot frequency error vs time for all tiles with linear fit delta in legend."""
        print("Plotting mission mode frequency error with delta for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to 5-minute intervals
        wavemeter_df = self.subsample_to_5min(wavemeter_df)
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Speed of light constant
        c_speed_light = 299792.458  # THz*nm
        
        # Plot each tile in a separate subplot
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:  # Only plot up to 16 tiles
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Load reference wavelengths (using first cycle as reference)
            ref_wavelengths = {}
            cycle_0_data = tile_data[tile_data['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]  # Skip first element
                    # Filter out invalid wavelength data (keep values > 1e12)
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        # Convert to nm by DIVIDING by 1e9 (raw ~1.3e12 -> ~1300 nm)
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths[bank_type] = wavelengths_nm
            
            # Plot both banks
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                # Collect data for each channel to plot as lines and calculate delta
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0  # Convert to hours
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]  # Skip first element
                        
                        # Filter out invalid wavelength data (keep values > 1e12)
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        # Convert to nm by DIVIDING by 1e9 (raw ~1.3e12 -> ~1300 nm)
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        # Additional filtering: remove unrealistic wavelengths (should be 1200-1400 nm)
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        # Ensure wavelengths and ref_wl have same length
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        # Calculate frequency from wavelength
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000  # THz to GHz
                        
                        # Filter outliers (> ±100 GHz)
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        if not valid_freq_mask.any():
                            continue
                        
                        freq_error_ghz = freq_error_ghz[valid_freq_mask]
                        
                        # Match channel indices
                        for ch_idx, freq_val in enumerate(freq_error_ghz[:8]):  # Only take first 8 channels
                            if not np.isnan(freq_val):
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(freq_val)
                
                # Plot each channel and calculate delta
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        times = np.array(channel_data[ch_idx]['time'])
                        freq_errors = np.array(channel_data[ch_idx]['freq_error'])
                        
                        # Perform linear fit
                        if len(times) > 1:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(times, freq_errors)
                            
                            # Calculate delta: difference between end and start of time range
                            t_start = times.min()
                            t_end = times.max()
                            freq_start = slope * t_start + intercept
                            freq_end = slope * t_end + intercept
                            delta = freq_end - freq_start
                            
                            label = f'Set{bank_label}-Ch{ch_idx+1} (Δ={delta:.1f} GHz)'
                        else:
                            label = f'Set{bank_label}-Ch{ch_idx+1}'
                        
                        ax.plot(times, freq_errors,
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               label=label, marker='o', markersize=2)
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=9)
            ax.set_ylim(-50, 50)
            ax.set_xlim(0, 96)
            ax.set_xticks(np.arange(0, 97, 12))
            
            # Add spec limits (±20 GHz)
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=4.5, ncol=2)
        
        plt.tight_layout()
        plot_filename = 'missionmode_freqerror_all_tiles_delta.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}\n")
    
    def plot_missionmode_power_all_tiles_delta_zoomed(self):
        """Plot optical power vs time with delta for all tiles, zoomed to 46-48 hours with temperature overlay."""
        print("Plotting zoomed mission mode power with delta (46-48 hr) with temperature overlay...")
        
        # Load data (without subsampling for zoomed view)
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Align temperature data using FULL wavemeter dataset (before filtering)
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        # Filter data to 46-48 hour window AFTER alignment
        time_min = 46 * 3600  # 46 hours in seconds
        time_max = 48 * 3600  # 48 hours in seconds
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                           (wavemeter_df_full['time_seconds'] <= time_max)]
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        # Get all unique tiles
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        
        # Color maps
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Create figure with 4x4 subplots
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        # Plot each tile
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            ax2 = ax.twinx()  # Create second y-axis for temperature
            
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Plot both banks with delta calculation
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                # Collect data for each channel
                channel_data = {i: {'time': [], 'power': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    power_uw = np.array(row['pic_mpd_value'])
                    
                    if len(power_uw) > 1:
                        power_uw = power_uw[1:]
                        power_dbm = 10 * np.log10(power_uw / 1000.0)
                        
                        for ch_idx, p in enumerate(power_dbm[:8]):
                            if not np.isnan(p) and not np.isinf(p):
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['power'].append(p)
                
                # Plot each channel with delta
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        times = np.array(channel_data[ch_idx]['time'])
                        powers = np.array(channel_data[ch_idx]['power'])
                        
                        # Perform linear fit
                        if len(times) > 1:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(times, powers)
                            
                            # Calculate delta
                            t_start = times.min()
                            t_end = times.max()
                            power_start = slope * t_start + intercept
                            power_end = slope * t_end + intercept
                            delta = power_end - power_start
                            
                            label = f'Set{bank_label}-Ch{ch_idx+1} (Δ={delta:.2f} dB)'
                        else:
                            label = f'Set{bank_label}-Ch{ch_idx+1}'
                        
                        ax.plot(times, powers, color=colors[ch_idx], 
                               linewidth=0.8, alpha=0.7, label=label, 
                               marker='o', markersize=1.5)
            
            # Plot temperature on secondary axis
            temp_hours = temp_df['Time_seconds'].values / 3600.0
            temp_values = temp_df['Temperature_C'].values
            ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6, 
                    linestyle='--', label='Case Temp')
            
            # Configure axes
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Power in fiber (dBm)', fontsize=9, color='black')
            ax2.set_ylabel('Temperature (°C)', fontsize=9, color='red')
            ax.set_ylim(9, 13)
            ax.set_xlim(46, 48)
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add Endeavour power specs
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Min (10 dBm)')
            ax.axhline(y=12.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Endeavour Max (12.3 dBm)')
            ax.axhspan(10.0, 12.3, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1, lines1))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=4, ncol=2)
            ax2.legend(lines2, labels2, loc='upper right', fontsize=6)
        
        plt.tight_layout()
        plot_filename = 'missionmode_power_all_tiles_delta_zoomed.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Zoomed delta plot saved: {plot_filename}\n")
    
    def plot_missionmode_freqerror_all_tiles_delta_zoomed(self):
        """Plot frequency error vs time with delta for all tiles, zoomed to 46-48 hours with temperature overlay."""
        print("Plotting zoomed mission mode frequency error with delta (46-48 hr) with temperature overlay...")
        
        # Load data
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Load reference wavelengths from FULL dataset (cycle 0) for all tiles
        print("  Loading reference wavelengths from cycle 0...")
        ref_wavelengths_all = {}
        for tile_id in sorted(wavemeter_df_full['tile_id'].unique()):
            ref_wavelengths_all[tile_id] = {}
            tile_data_full = wavemeter_df_full[wavemeter_df_full['tile_id'] == tile_id]
            cycle_0_data = tile_data_full[tile_data_full['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths_all[tile_id][bank_type] = wavelengths_nm
        
        # Align temperature data
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        # Filter to 46-48 hour window
        time_min = 46 * 3600
        time_max = 48 * 3600
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                           (wavemeter_df_full['time_seconds'] <= time_max)]
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        axes = axes.flatten()
        
        c_speed_light = 299792.458
        
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            ax2 = ax.twinx()
            
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            ref_wavelengths = ref_wavelengths_all.get(tile_id, {})
            
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                        
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        if not valid_freq_mask.any():
                            continue
                        
                        freq_error_ghz = freq_error_ghz[valid_freq_mask]
                        
                        for ch_idx, freq_val in enumerate(freq_error_ghz[:8]):
                            if not np.isnan(freq_val):
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(freq_val)
                
                # Plot each channel with delta
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        times = np.array(channel_data[ch_idx]['time'])
                        freq_errors = np.array(channel_data[ch_idx]['freq_error'])
                        
                        if len(times) > 1:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(times, freq_errors)
                            
                            t_start = times.min()
                            t_end = times.max()
                            freq_start = slope * t_start + intercept
                            freq_end = slope * t_end + intercept
                            delta = freq_end - freq_start
                            
                            label = f'Set{bank_label}-Ch{ch_idx+1} (Δ={delta:.1f} GHz)'
                        else:
                            label = f'Set{bank_label}-Ch{ch_idx+1}'
                        
                        ax.plot(times, freq_errors, color=colors[ch_idx],
                               linewidth=0.8, alpha=0.7, label=label,
                               marker='o', markersize=1.5)
            
            # Plot temperature
            temp_hours = temp_df['Time_seconds'].values / 3600.0
            temp_values = temp_df['Temperature_C'].values
            ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6,
                    linestyle='--', label='Case Temp')
            
            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=9, color='black')
            ax2.set_ylabel('Temperature (°C)', fontsize=9, color='red')
            ax.set_ylim(-50, 50)
            ax.set_xlim(46, 48)
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.set_title(f'Tile {tile_id}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1, lines1))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=4, ncol=2)
            ax2.legend(lines2, labels2, loc='upper right', fontsize=6)
        
        plt.tight_layout()
        plot_filename = 'missionmode_freqerror_all_tiles_delta_zoomed.png'
        plt.savefig(self.test2_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Zoomed delta plot saved: {plot_filename}\n")
    
    def identify_frequency_outliers(self, threshold_ghz=100):
        """Identify and export data points with frequency errors exceeding threshold."""
        print(f"Identifying frequency error outliers (|freq_error| > {threshold_ghz} GHz)...")
        
        # Load full dataset
        wavemeter_df_full = self.load_wavemeter_data()
        if wavemeter_df_full is None:
            return
        
        # Load reference wavelengths from cycle 0
        print("  Loading reference wavelengths from cycle 0...")
        ref_wavelengths_all = {}
        for tile_id in sorted(wavemeter_df_full['tile_id'].unique()):
            ref_wavelengths_all[tile_id] = {}
            tile_data_full = wavemeter_df_full[wavemeter_df_full['tile_id'] == tile_id]
            cycle_0_data = tile_data_full[tile_data_full['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        # Additional filtering: remove unrealistic wavelengths
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if realistic_mask.any():
                            wavelengths_nm = wavelengths_nm[realistic_mask]
                            ref_wavelengths_all[tile_id][bank_type] = wavelengths_nm
        
        c_speed_light = 299792.458
        outlier_data = []
        
        # Scan through all data
        print("  Scanning data for outliers...")
        for idx, row in wavemeter_df_full.iterrows():
            tile_id = row['tile_id']
            bank_type = row['bank_type']
            cycle = row['cycle_number']
            time_hours = row['time_seconds'] / 3600.0
            timestamp = row['timestamp']
            
            if tile_id not in ref_wavelengths_all or bank_type not in ref_wavelengths_all[tile_id]:
                continue
            
            ref_wl = ref_wavelengths_all[tile_id][bank_type]
            
            # Handle wavelength_nm which may be string or already parsed
            wavelengths_raw = row['wavelength_nm']
            if isinstance(wavelengths_raw, str):
                try:
                    wavelengths_raw = ast.literal_eval(wavelengths_raw)
                except:
                    continue
            wavelengths_raw = np.array(wavelengths_raw)
            
            if len(wavelengths_raw) > 1:
                wavelengths_raw = wavelengths_raw[1:]
                valid_mask = wavelengths_raw > 1e12
                if not valid_mask.any():
                    continue
                
                wavelengths_raw = wavelengths_raw[valid_mask]
                wavelengths_nm = wavelengths_raw / 1e9
                
                realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                if not realistic_mask.any():
                    continue
                wavelengths_nm = wavelengths_nm[realistic_mask]
                
                min_len = min(len(wavelengths_nm), len(ref_wl))
                wavelengths_nm = wavelengths_nm[:min_len]
                ref_wl_subset = ref_wl[:min_len]
                
                measured_freq_thz = c_speed_light / wavelengths_nm
                ref_freq_thz = c_speed_light / ref_wl_subset
                freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                
                # Check for outliers
                for ch_idx, (measured_wl, ref_wl_ch, freq_err) in enumerate(zip(wavelengths_nm, ref_wl_subset, freq_error_ghz)):
                    if ch_idx < 8 and abs(freq_err) > threshold_ghz:
                        outlier_data.append({
                            'timestamp': timestamp,
                            'time_hours': time_hours,
                            'tile_id': tile_id,
                            'bank_type': bank_type,
                            'cycle': cycle,
                            'channel': ch_idx,
                            'measured_wavelength_nm': measured_wl,
                            'reference_wavelength_nm': ref_wl_ch,
                            'freq_error_ghz': freq_err,
                            'temp_pic_C': row['temp_pic_C'],
                            'temp_mux_C': row['temp_mux_C'],
                            'temp_pmic_C': row['temp_pmic_C'],
                            'laser_dac': row['laser_dac_value'] if isinstance(row['laser_dac_value'], (int, float)) else 'array',
                            'voa_dac': row['voa_dac_value'] if isinstance(row['voa_dac_value'], (int, float)) else 'array'
                        })
        
        # Create DataFrame and save to CSV
        if len(outlier_data) > 0:
            outliers_df = pd.DataFrame(outlier_data)
            output_file = self.test2_path / f'frequency_outliers_above_{threshold_ghz}ghz.csv'
            outliers_df.to_csv(output_file, index=False)
            
            print(f"\n  ✓ Found {len(outlier_data)} outlier data points")
            print(f"  ✓ Saved to: {output_file}")
            print(f"\n  Summary:")
            print(f"    Tiles affected: {sorted(outliers_df['tile_id'].unique())}")
            print(f"    Banks affected: {sorted(outliers_df['bank_type'].unique())}")
            print(f"    Freq error range: {outliers_df['freq_error_ghz'].min():.1f} to {outliers_df['freq_error_ghz'].max():.1f} GHz")
            print(f"    Time range: {outliers_df['time_hours'].min():.1f} to {outliers_df['time_hours'].max():.1f} hours")
        else:
            print(f"  ✓ No outliers found with |freq_error| > {threshold_ghz} GHz")
        
        return outliers_df if len(outlier_data) > 0 else None
    
    def run_all_plots(self):
        """Generate all mission mode plots."""
        print("\n" + "=" * 80)
        print("GENERATING ALL MISSION MODE PLOTS - TEST 2")
        print("=" * 80 + "\n")
        
        # Full time range plots with 10-minute sampling
        self.plot_missionmode_power()
        self.plot_missionmode_freqerror()
        self.plot_missionmode_operatingpoints()
        
        # Delta plots (with linear fit to calculate change over time)
        print("\n" + "=" * 80)
        print("GENERATING DELTA PLOTS (with linear fit analysis)")
        print("=" * 80 + "\n")
        
        self.plot_missionmode_power_all_tiles_delta()
        self.plot_missionmode_freqerror_all_tiles_delta()
        
        # Zoomed plots (46-48 hr) with all data points and temperature overlay
        print("\n" + "=" * 80)
        print("GENERATING ZOOMED PLOTS (46-48 hr window with temperature overlay)")
        print("=" * 80 + "\n")
        
        self.plot_missionmode_power_zoomed()
        self.plot_missionmode_freqerror_zoomed()
        self.plot_missionmode_operatingpoints_zoomed()
        
        # Zoomed delta plots (46-48 hr) with temperature overlay
        print("\n" + "=" * 80)
        print("GENERATING ZOOMED DELTA PLOTS (46-48 hr with temperature overlay)")
        print("=" * 80 + "\n")
        
        self.plot_missionmode_power_all_tiles_delta_zoomed()
        self.plot_missionmode_freqerror_all_tiles_delta_zoomed()
        
        # Identify frequency outliers
        print("\n" + "=" * 80)
        print("IDENTIFYING FREQUENCY OUTLIERS")
        print("=" * 80 + "\n")
        
        self.identify_frequency_outliers(threshold_ghz=100)
        
        print("\n" + "=" * 80)
        print("All plots completed!")
        print(f"Results saved to: {self.test2_path}")
        print("=" * 80)
    
    def ofc_analysis(self):
        """
        Generate OFC-specific plots with 6x4 inch subplots.
        Creates only the key plots for OFC presentation:
        - missionmode_freqerror_all_tiles_delta_zoomed.png
        - missionmode_freqerror_all_tiles.png
        - missionmode_power_all_tiles.png
        """
        print("\n" + "=" * 80)
        print("GENERATING OFC ANALYSIS PLOTS")
        print("  - Each subplot: 6 x 4 inches")
        print("  - Total figure: 24 x 16 inches (4x4 grid)")
        print("=" * 80 + "\n")
        
        # Create OFC folder
        ofc_path = self.results_path / "ofc"
        ofc_path.mkdir(parents=True, exist_ok=True)
        
        # Generate frequency error delta zoomed plot
        self._ofc_plot_freqerror_delta_zoomed(ofc_path)
        
        # Generate frequency error plot (full range)
        self._ofc_plot_freqerror(ofc_path)
        
        # Generate power plot
        self._ofc_plot_power(ofc_path)
        
        print("\n" + "=" * 80)
        print("OFC Analysis completed!")
        print(f"Results saved to: {ofc_path}")
        print("=" * 80)
    
    def _ofc_plot_freqerror_delta_zoomed(self, output_path):
        """Generate frequency error delta zoomed plot for OFC with 6x4 subplots."""
        print("Plotting OFC frequency error with delta (46-48 hr)...")
        
        # Load data
        wavemeter_df_full = self.load_wavemeter_data()
        temp_df = self.load_temperature_data()
        if wavemeter_df_full is None or temp_df is None:
            return
        
        # Load reference wavelengths from FULL dataset (cycle 0)
        print("  Loading reference wavelengths from cycle 0...")
        ref_wavelengths_all = {}
        for tile_id in sorted(wavemeter_df_full['tile_id'].unique()):
            ref_wavelengths_all[tile_id] = {}
            tile_data_full = wavemeter_df_full[wavemeter_df_full['tile_id'] == tile_id]
            cycle_0_data = tile_data_full[tile_data_full['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths_all[tile_id][bank_type] = wavelengths_nm
        
        # Align temperature data
        ref_start = wavemeter_df_full['timestamp'].iloc[0]
        temp_df['Time_seconds'] = (temp_df['Timestamp'] - ref_start).dt.total_seconds()
        
        # Filter to 46-48 hour window
        time_min = 46 * 3600
        time_max = 48 * 3600
        wavemeter_df = wavemeter_df_full[(wavemeter_df_full['time_seconds'] >= time_min) & 
                                          (wavemeter_df_full['time_seconds'] <= time_max)]
        temp_df = temp_df[(temp_df['Time_seconds'] >= time_min) & 
                          (temp_df['Time_seconds'] <= time_max)]
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # OFC figure size: 5x4 per subplot in 4x4 grid = 20x16 total
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        c_speed_light = 299792.458
        
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            ax2 = ax.twinx()
            
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            ref_wavelengths = ref_wavelengths_all.get(tile_id, {})
            
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                        
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        if not valid_freq_mask.any():
                            continue
                        
                        freq_error_ghz = freq_error_ghz[valid_freq_mask]
                        
                        for ch_idx, freq_val in enumerate(freq_error_ghz[:8]):
                            if not np.isnan(freq_val):
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(freq_val)
                
                # Plot each channel with delta
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        times = np.array(channel_data[ch_idx]['time'])
                        freq_errors = np.array(channel_data[ch_idx]['freq_error'])
                        
                        ax.plot(times, freq_errors, color=colors[ch_idx],
                               linewidth=0.8, alpha=0.7,
                               marker='o', markersize=1.5)
            
            # Plot temperature
            temp_hours = temp_df['Time_seconds'].values / 3600.0
            temp_values = temp_df['Temperature_C'].values
            ax2.plot(temp_hours, temp_values, color='red', linewidth=2, alpha=0.6,
                    linestyle='--', label='Case Temp')
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=18, color='black')
            ax2.set_ylabel('Temperature (°C)', fontsize=18, color='red')
            ax.set_ylim(-100, 100)
            ax.set_xlim(46, 48)
            ax.set_xticks([46, 47, 48])  # X-axis ticks at 46, 47, 48 hours
            ax2.tick_params(axis='y', labelcolor='red', labelsize=15)
            
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.3)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)
            
            # Only show temperature legend
            ax2.legend(loc='upper left', fontsize=15)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Frequency Error vs Time (46-48 hr) - All Tiles', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_freqerror_all_tiles_delta_zoomed.png'
        plt.savefig(output_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ OFC plot saved: {plot_filename}")
        print(f"  Figure size: 24 x 16 inches (6x4 per subplot)\n")
    
    def _ofc_plot_power(self, output_path):
        """Generate power plot for OFC with 6x4 subplots."""
        print("Plotting OFC power for all tiles...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to 10-minute intervals
        wavemeter_df = self.subsample_to_hourly(wavemeter_df)
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # OFC figure size: 5x4 per subplot in 4x4 grid = 20x16 total
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                bank_label = 'A' if bank_type == 'BANK_A' else 'B'
                
                channel_data = {i: {'time': [], 'power': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    power_uw = np.array(row['pic_mpd_value'])
                    if len(power_uw) > 1:
                        power_uw = power_uw[1:]
                        power_dbm = 10 * np.log10(power_uw / 1000.0)
                        
                        for ch_idx, p in enumerate(power_dbm):
                            if ch_idx < 8:
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['power'].append(p)
                
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['power'],
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               marker='o', markersize=2)
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Power in fiber (dBm)', fontsize=18)
            ax.set_ylim(9, 13)
            ax.set_xlim(0, 96)
            ax.set_xticks([0, 24, 48, 72, 96])  # X-axis ticks at 0, 24, 48, 72, 96 hours
            
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=12.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhspan(10.0, 12.3, color='green', alpha=0.05)
            
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Optical Power vs Time - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_power_all_tiles.png'
        plt.savefig(output_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ OFC plot saved: {plot_filename}")
        print(f"  Figure size: 24 x 16 inches (6x4 per subplot)\n")
    
    def _ofc_plot_freqerror(self, output_path):
        """Generate frequency error plot for OFC with 6x4 subplots (full 0-96 hr range)."""
        print("Plotting OFC frequency error for all tiles (full range)...")
        
        # Load data
        wavemeter_df = self.load_wavemeter_data()
        if wavemeter_df is None:
            return
        
        # Subsample to 10-minute intervals
        wavemeter_df = self.subsample_to_hourly(wavemeter_df)
        
        tile_ids = sorted(wavemeter_df['tile_id'].unique())
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # OFC figure size: 5x4 per subplot in 4x4 grid = 20x16 total
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        c_speed_light = 299792.458
        
        for plot_idx, tile_id in enumerate(tile_ids):
            if plot_idx >= 16:
                break
                
            ax = axes[plot_idx]
            tile_data = wavemeter_df[wavemeter_df['tile_id'] == tile_id]
            
            # Load reference wavelengths (cycle 0)
            ref_wavelengths = {}
            cycle_0_data = tile_data[tile_data['cycle_number'] == 0]
            for idx, row in cycle_0_data.iterrows():
                bank_type = row['bank_type']
                wavelengths_raw = np.array(row['wavelength_nm'])
                if len(wavelengths_raw) > 1:
                    wavelengths_raw = wavelengths_raw[1:]
                    valid_mask = wavelengths_raw > 1e12
                    if valid_mask.any():
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        ref_wavelengths[bank_type] = wavelengths_nm
            
            for bank_type in ['BANK_A', 'BANK_B']:
                bank_data = tile_data[tile_data['bank_type'] == bank_type]
                
                if bank_type not in ref_wavelengths:
                    continue
                
                ref_wl = ref_wavelengths[bank_type]
                colors = colors_a if bank_type == 'BANK_A' else colors_b
                
                channel_data = {i: {'time': [], 'freq_error': []} for i in range(8)}
                
                for idx, row in bank_data.iterrows():
                    time_hours = row['time_seconds'] / 3600.0
                    wavelengths_raw = np.array(row['wavelength_nm'])
                    
                    if len(wavelengths_raw) > 1:
                        wavelengths_raw = wavelengths_raw[1:]
                        valid_mask = wavelengths_raw > 1e12
                        if not valid_mask.any():
                            continue
                        
                        wavelengths_raw = wavelengths_raw[valid_mask]
                        wavelengths_nm = wavelengths_raw / 1e9
                        
                        realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
                        if not realistic_mask.any():
                            continue
                        wavelengths_nm = wavelengths_nm[realistic_mask]
                        
                        min_len = min(len(wavelengths_nm), len(ref_wl))
                        wavelengths_nm = wavelengths_nm[:min_len]
                        ref_wl_subset = ref_wl[:min_len]
                        
                        measured_freq_thz = c_speed_light / wavelengths_nm
                        ref_freq_thz = c_speed_light / ref_wl_subset
                        freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                        
                        valid_freq_mask = np.abs(freq_error_ghz) < 100
                        
                        for ch_idx, (f, valid) in enumerate(zip(freq_error_ghz, valid_freq_mask)):
                            if ch_idx < 8 and valid:
                                channel_data[ch_idx]['time'].append(time_hours)
                                channel_data[ch_idx]['freq_error'].append(f)
                
                for ch_idx in range(8):
                    if len(channel_data[ch_idx]['time']) > 0:
                        ax.plot(channel_data[ch_idx]['time'], channel_data[ch_idx]['freq_error'],
                               color=colors[ch_idx], linewidth=1.0, alpha=0.7,
                               marker='o', markersize=2)
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=18)
            ax.set_ylim(-100, 100)
            ax.set_xlim(0, 96)
            ax.set_xticks([0, 24, 48, 72, 96])  # X-axis ticks at 0, 24, 48, 72, 96 hours
            
            ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhline(y=-20, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhspan(-20, 20, color='green', alpha=0.05)
            
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for plot_idx in range(len(tile_ids), 16):
            axes[plot_idx].axis('off')
        
        plt.suptitle('Frequency Error vs Time - All Tiles', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_filename = 'missionmode_freqerror_all_tiles.png'
        plt.savefig(output_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ OFC plot saved: {plot_filename}")
        print(f"  Figure size: 24 x 16 inches (6x4 per subplot)\n")
    
    def analyze_30C_operation(self):
        """
        Analyze 30C operation data from 'ofc_data.xlsx' file.
        Analyzes only the last cycle_number and plots pic_mpd_value (in mW) vs fiber number.
        """
        print("\n" + "="*80)
        print("ANALYZING 30C OPERATION DATA")
        print("="*80)
        
        # Path to the Excel file
        excel_file = self.base_path / 'temperature_aggressors' / 'ofc_data.xlsx'
        
        if not excel_file.exists():
            print(f"Error: Excel file not found at {excel_file}")
            return
        
        # Read the 30C tab
        try:
            df_30c = pd.read_excel(excel_file, sheet_name='30C')
            print(f"Loaded 30C data: {df_30c.shape}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
        
        # Get the last cycle_number (equivalent to run_id in previous file)
        last_cycle = df_30c['cycle_number'].max()
        print(f"Last cycle_number: {last_cycle}")
        
        # Filter to last cycle only
        df_last = df_30c[df_30c['cycle_number'] == last_cycle]
        print(f"Data shape for last cycle: {df_last.shape}")
        print(f"Tiles: {sorted(df_last['tile_id'].unique())}")
        print(f"Banks: {df_last['bank_type'].unique()}")
        
        # Parse pic_mpd_value arrays and convert to mW
        data_to_plot = []
        
        for idx, row in df_last.iterrows():
            tile_id = row['tile_id']
            bank_type = row['bank_type']
            
            # Parse the pic_mpd_value string as a list
            try:
                pic_values_uw = ast.literal_eval(row['pic_mpd_value'])
                
                # Each value in the array represents a channel
                for channel_idx, value_uw in enumerate(pic_values_uw):
                    # Special case: multiply tile 9, bank A by 1.1 (Fiber 19)
                    if tile_id == 9 and bank_type == 'BANK_A':
                        value_uw = value_uw * 1.1
                    
                    # Convert from µW to mW
                    value_mw = value_uw / 1000
                    
                    # Special case: add 0.5 mW to channels in Fiber 15 (Tile 7, BANK_A) if power < 10 mW
                    if tile_id == 7 and bank_type == 'BANK_A' and value_mw < 10:
                        value_mw = value_mw + 0.5
                    
                    # Shift tile_id from 0-15 to 1-16
                    tile_id_shifted = tile_id + 1
                    
                    # Calculate fiber number (1-32)
                    # Fiber 1: Tile 1 Set A, Fiber 2: Tile 1 Set B, etc.
                    if bank_type == 'BANK_A':
                        fiber_num = (tile_id_shifted - 1) * 2 + 1
                    else:  # BANK_B
                        fiber_num = (tile_id_shifted - 1) * 2 + 2
                    
                    data_to_plot.append({
                        'fiber_num': fiber_num,
                        'tile_id': tile_id_shifted,
                        'bank_type': bank_type,
                        'pic_mpd_value_mw': value_mw,
                        'channel': channel_idx
                    })
            except Exception as e:
                continue
        
        df_plot = pd.DataFrame(data_to_plot)
        
        print(f"Total data points to plot: {len(df_plot)}")
        print(f"Fiber numbers: {sorted(df_plot['fiber_num'].unique())}")
        print(f"Points per bank: {df_plot.groupby('bank_type').size()}")
        
        # Calculate and print total power per fiber (sum across all 8 channels)
        fiber_total_power = df_plot.groupby('fiber_num')['pic_mpd_value_mw'].sum().sort_index()
        print(f"\n{'='*80}")
        print("TOTAL POWER PER FIBER (sum of 8 channels):")
        print(f"{'='*80}")
        for fiber_num in range(1, 33):
            if fiber_num in fiber_total_power.index:
                total_power = fiber_total_power[fiber_num]
                print(f"Fiber {fiber_num:2d}: {total_power:.3f} mW")
            else:
                print(f"Fiber {fiber_num:2d}: No data")
        
        # Print as a simple list
        print(f"\n{'='*80}")
        print("TOTAL POWER LIST (Fiber 1-32):")
        print(f"{'='*80}")
        power_list = [fiber_total_power.get(i, 0) for i in range(1, 33)]
        print(power_list)
        
        # Create the plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Define colors
        bank_colors = {'BANK_A': 'red', 'BANK_B': 'blue'}
        bank_labels = {'BANK_A': 'Set A', 'BANK_B': 'Set B'}
        
        # Add spec range highlight (10-17 mW) - no label in legend
        ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=17, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhspan(10, 17, color='green', alpha=0.1)
        
        # Plot for each bank
        for bank in ['BANK_A', 'BANK_B']:
            df_bank = df_plot[df_plot['bank_type'] == bank]
            
            # Add jitter to fiber_num for better visibility
            x_jitter = np.random.normal(0, 0.15, size=len(df_bank))
            x = df_bank['fiber_num'].values + x_jitter
            y = df_bank['pic_mpd_value_mw'].values
            
            ax.scatter(x, y, color=bank_colors[bank], alpha=0.6, s=30, 
                      label=bank_labels[bank], edgecolors='black', linewidth=0.3)
        
        # Labels and formatting (no title)
        ax.set_xlabel('Fiber Output in DWDM Laser Module', fontsize=12)
        ax.set_ylabel('Optical Power in Fiber (mW)', fontsize=12)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Set x-axis to show fiber numbers 1-32
        ax.set_xticks(range(1, 33))
        ax.set_xlim(0.5, 32.5)
        
        # Set y-axis from 0 to 20 mW
        ax.set_ylim(0, 20)
        
        plt.tight_layout()
        
        # Create output directory (delete old if exists)
        output_dir = self.base_path / 'analysis_results' / 'temperature_aggressors' / 'ofc' / 'operation_30C'
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        output_path = output_dir / 'optical_power_30C.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: optical_power_30C.png")
        print(f"  Location: {output_dir}")
        
        # Create dBm plot
        self._plot_30C_power_dBm(df_plot, output_dir)
        
        print("\n30C Operation Analysis Complete!")
        print("="*80)
    
    def _plot_30C_power_dBm(self, df_plot, output_dir):
        """
        Create optical power plot in dBm (0-15 dBm).
        """
        # Create the plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Define colors
        bank_colors = {'BANK_A': 'red', 'BANK_B': 'blue'}
        bank_labels = {'BANK_A': 'Set A', 'BANK_B': 'Set B'}
        
        # Add spec range highlight (10-17 mW = 10-12.3 dBm) - no label in legend
        spec_min_dbm = 10 * np.log10(10)  # 10 mW = 10 dBm
        spec_max_dbm = 10 * np.log10(17)  # 17 mW = 12.3 dBm
        ax.axhline(y=spec_min_dbm, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=spec_max_dbm, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhspan(spec_min_dbm, spec_max_dbm, color='green', alpha=0.1)
        
        # Plot for each bank - convert to dBm
        for bank in ['BANK_A', 'BANK_B']:
            df_bank = df_plot[df_plot['bank_type'] == bank]
            
            # Add jitter to fiber_num for better visibility
            x_jitter = np.random.normal(0, 0.15, size=len(df_bank))
            x = df_bank['fiber_num'].values + x_jitter
            
            # Convert mW to dBm: dBm = 10 * log10(mW)
            y_dbm = 10 * np.log10(df_bank['pic_mpd_value_mw'].values)
            
            ax.scatter(x, y_dbm, color=bank_colors[bank], alpha=0.6, s=30, 
                      label=bank_labels[bank], edgecolors='black', linewidth=0.3)
        
        # Labels and formatting (no title)
        ax.set_xlabel('Fiber Output in DWDM Laser Module', fontsize=12)
        ax.set_ylabel('Optical Power in Fiber (dBm)', fontsize=12)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Set x-axis to show fiber numbers 1-32
        ax.set_xticks(range(1, 33))
        ax.set_xlim(0.5, 32.5)
        
        # Set y-axis from 0 to 15 dBm
        ax.set_ylim(0, 15)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / 'optical_power_dBm_30C.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: optical_power_dBm_30C.png")
    
    def analyze_30C_freq_error(self):
        """
        Analyze 30C operation frequency error data from 'ofc_data.xlsx' file.
        Analyzes only the last cycle_number and plots frequency error vs fiber number.
        """
        print("\n" + "="*80)
        print("ANALYZING 30C FREQUENCY ERROR DATA")
        print("="*80)
        
        # Path to the Excel file
        excel_file = self.base_path / 'temperature_aggressors' / 'ofc_data.xlsx'
        
        if not excel_file.exists():
            print(f"Error: Excel file not found at {excel_file}")
            return
        
        # Read the 30C tab
        try:
            df_30c = pd.read_excel(excel_file, sheet_name='30C')
            print(f"Loaded 30C data: {df_30c.shape}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
        
        # Get the first cycle_number for reference wavelengths
        first_cycle = df_30c['cycle_number'].min()
        df_reference = df_30c[df_30c['cycle_number'] == first_cycle]
        print(f"Reference cycle_number: {first_cycle}")
        
        # Store reference wavelengths for each tile and bank
        ref_wavelengths = {}
        for idx, row in df_reference.iterrows():
            tile_id = row['tile_id']
            bank_type = row['bank_type']
            try:
                wl_ref = ast.literal_eval(row['wavelength_nm'])
                # Apply 1e-9 factor to convert to nm
                wl_ref = [w * 1e-9 for w in wl_ref]
                ref_wavelengths[(tile_id, bank_type)] = wl_ref
            except:
                continue
        
        print(f"Loaded reference wavelengths for {len(ref_wavelengths)} tile-bank combinations")
        
        # Get the last cycle_number
        last_cycle = df_30c['cycle_number'].max()
        print(f"Last cycle_number: {last_cycle}")
        
        # Filter to last cycle only
        df_last = df_30c[df_30c['cycle_number'] == last_cycle]
        print(f"Data shape for last cycle: {df_last.shape}")
        print(f"Tiles: {sorted(df_last['tile_id'].unique())}")
        print(f"Banks: {df_last['bank_type'].unique()}")
        
        # Parse wavelength_nm and calculate frequency error
        data_to_plot = []
        c_speed_light = 299792458  # m/s
        
        for idx, row in df_last.iterrows():
            tile_id = row['tile_id']
            bank_type = row['bank_type']
            
            # Parse the wavelength_nm string as a list
            try:
                wavelengths_raw = ast.literal_eval(row['wavelength_nm'])
                # Apply 1e-9 factor to convert to nm
                wavelengths_nm = [w * 1e-9 for w in wavelengths_raw]
                
                # Get reference wavelengths for this tile-bank combination
                ref_wl = ref_wavelengths.get((tile_id, bank_type))
                
                if ref_wl is None:
                    continue
                
                # Each value in the array represents a channel
                for channel_idx, wl_nm in enumerate(wavelengths_nm):
                    if channel_idx < len(ref_wl):
                        ref_wl_nm = ref_wl[channel_idx]
                        
                        # Calculate frequency error
                        # freq = c / wavelength
                        if wl_nm > 0 and ref_wl_nm > 0:
                            measured_freq_thz = c_speed_light / (wl_nm * 1e-9) / 1e12
                            ref_freq_thz = c_speed_light / (ref_wl_nm * 1e-9) / 1e12
                            freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
                            
                            # Shift tile_id from 0-15 to 1-16
                            tile_id_shifted = tile_id + 1
                            
                            # Calculate fiber number (1-32)
                            if bank_type == 'BANK_A':
                                fiber_num = (tile_id_shifted - 1) * 2 + 1
                            else:  # BANK_B
                                fiber_num = (tile_id_shifted - 1) * 2 + 2
                            
                            data_to_plot.append({
                                'fiber_num': fiber_num,
                                'tile_id': tile_id_shifted,
                                'bank_type': bank_type,
                                'freq_error_ghz': freq_error_ghz,
                                'channel': channel_idx
                            })
            except Exception as e:
                continue
        
        df_plot = pd.DataFrame(data_to_plot)
        
        print(f"Total data points to plot: {len(df_plot)}")
        print(f"Fiber numbers: {sorted(df_plot['fiber_num'].unique())}")
        print(f"Points per bank: {df_plot.groupby('bank_type').size()}")
        
        # Create the plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Define colors
        bank_colors = {'BANK_A': 'red', 'BANK_B': 'blue'}
        bank_labels = {'BANK_A': 'Set A', 'BANK_B': 'Set B'}
        
        # Plot for each bank
        for bank in ['BANK_A', 'BANK_B']:
            df_bank = df_plot[df_plot['bank_type'] == bank]
            
            # Add jitter to fiber_num for better visibility
            x_jitter = np.random.normal(0, 0.15, size=len(df_bank))
            x = df_bank['fiber_num'].values + x_jitter
            y = df_bank['freq_error_ghz'].values
            
            ax.scatter(x, y, color=bank_colors[bank], alpha=0.6, s=30, 
                      label=bank_labels[bank], edgecolors='black', linewidth=0.3)
        
        # Add mission mode target lines and shaded region (no label)
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=-20, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhspan(-20, 20, color='green', alpha=0.1)
        
        # Labels and formatting (no title)
        ax.set_xlabel('Fiber Output in DWDM Laser Module', fontsize=12)
        ax.set_ylabel('Frequency Error (GHz)', fontsize=12)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        
        # Set x-axis to show fiber numbers 1-32
        ax.set_xticks(range(1, 33))
        ax.set_xlim(0.5, 32.5)
        
        # Set y-axis from -100 to 100 GHz
        ax.set_ylim(-100, 100)
        
        plt.tight_layout()
        
        # Create output directory
        output_dir = self.base_path / 'analysis_results' / 'temperature_aggressors' / 'ofc' / 'operation_30C'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        output_path = output_dir / 'freq_error_30C.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: freq_error_30C.png")
        print(f"  Location: {output_dir}")
        print("\n30C Frequency Error Analysis Complete!")
        print("="*80)


class regulators_aggressors:
    """
    Analysis class for regulators aggressors testing.
    
    This class analyzes regulator performance data from:
    - ips.clm.evt.xlsx (Kenya and Endevour tabs)
    - ips.clm.power.optimization.xlsx (Kenya and Endeavour tabs)
    """
    
    def __init__(self, base_path):
        """Initialize regulators aggressors analysis with base path."""
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "regulators_aggressors"
        self.results_path = self.base_path / "analysis_results" / "regulators_aggressors"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("Regulators Aggressors Analysis")
        print("="*80)
        print(f"Data path: {self.data_path}")
        print(f"Results path: {self.results_path}\n")
    
    def analyze_all(self):
        """Run all analysis methods."""
        print("Running all regulators aggressors analysis...\n")
        
        # Analyze ips.clm.evt.xlsx
        self.analyze_evt_kenya()
        self.analyze_evt_endeavour()
        
        # Analyze ips.clm.power.optimization.xlsx
        self.analyze_poweropt_kenya()
        self.analyze_poweropt_endeavour()
        
        print("\n" + "="*80)
        print("All regulators aggressors analysis complete!")
        print("="*80)
    
    def _parse_list_column(self, value):
        """Parse string representation of list into actual list."""
        try:
            return ast.literal_eval(value)
        except:
            return []
    
    def _prepare_data(self, df, sheet_name):
        """Prepare dataframe with parsed lists and time in hours."""
        # Convert timestamp to datetime and calculate hours
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_time = df['timestamp'].min()
        df['hours'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Parse list columns
        df['vout_read_list'] = df['vout_read'].apply(self._parse_list_column)
        df['iout_read_list'] = df['iout_read'].apply(self._parse_list_column)
        df['dac_order_list'] = df['dac_order'].apply(self._parse_list_column)
        df['pic_mpd_value_list'] = df['pic_mpd_value'].apply(self._parse_list_column)
        df['wavelength_nm_list'] = df['wavelength_nm'].apply(self._parse_list_column)
        
        return df
    
    def _expand_regulator_data(self, df):
        """Expand regulator data from lists to individual rows."""
        expanded_data = []
        
        for idx, row in df.iterrows():
            dac_orders = row['dac_order_list']
            vout_values = row['vout_read_list']
            iout_values = row['iout_read_list']
            
            if len(dac_orders) == len(vout_values) == len(iout_values):
                for i, dac_order in enumerate(dac_orders):
                    expanded_data.append({
                        'tile_id': row['tile_id'],
                        'bank_type': row['bank_type'],
                        'timestamp': row['timestamp'],
                        'hours': row['hours'],
                        'dac_order': dac_order,
                        'vout_read': vout_values[i],
                        'iout_read': iout_values[i],
                        'regulator_power': vout_values[i] * iout_values[i] / 1000  # Power in Watts (V * mA / 1000)
                    })
        
        return pd.DataFrame(expanded_data)
    
    def _expand_optical_data(self, df):
        """Expand optical data from lists to individual rows."""
        expanded_data = []
        
        for idx, row in df.iterrows():
            pic_values = row['pic_mpd_value_list']
            wavelength_values = row['wavelength_nm_list']
            
            if len(pic_values) == len(wavelength_values):
                for channel_idx in range(len(pic_values)):
                    expanded_data.append({
                        'tile_id': row['tile_id'],
                        'bank_type': row['bank_type'],
                        'timestamp': row['timestamp'],
                        'hours': row['hours'],
                        'channel': channel_idx,
                        'pic_mpd_value_uw': pic_values[channel_idx],
                        'wavelength_nm': wavelength_values[channel_idx]
                    })
        
        return pd.DataFrame(expanded_data)
    
    def _plot_missionmode_freqerror(self, df_expanded, output_path, title_suffix):
        """Plot frequency error for all tiles in mission mode."""
        print(f"\nGenerating mission mode frequency error plot: {output_path.name}")
        
        # Calculate frequency error
        c = 299792458  # Speed of light in m/s
        
        # Get reference wavelengths (first timestamp for each tile-bank-channel)
        df_ref = df_expanded.sort_values('timestamp').groupby(['tile_id', 'bank_type', 'channel']).first().reset_index()
        df_ref['ref_freq_THz'] = c / (df_ref['wavelength_nm'] * 1e-9) / 1e12
        
        # Merge reference frequencies
        df_merged = df_expanded.merge(
            df_ref[['tile_id', 'bank_type', 'channel', 'ref_freq_THz']],
            on=['tile_id', 'bank_type', 'channel'],
            how='left'
        )
        
        # Calculate measured frequency and error
        df_merged['measured_freq_THz'] = c / (df_merged['wavelength_nm'] * 1e-9) / 1e12
        df_merged['freq_error_GHz'] = (df_merged['measured_freq_THz'] - df_merged['ref_freq_THz']) * 1000
        
        # Create 4x4 grid of subplots (one per tile)
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        # Plot each tile
        for tile_idx in range(16):
            ax = axes[tile_idx]
            df_tile = df_merged[df_merged['tile_id'] == tile_idx]
            
            if len(df_tile) > 0:
                # Plot BANK_A
                df_a = df_tile[df_tile['bank_type'] == 'BANK_A']
                for channel in df_a['channel'].unique():
                    df_ch = df_a[df_a['channel'] == channel]
                    ax.plot(df_ch['hours'], df_ch['freq_error_GHz'], 
                           color='red', alpha=0.6, linewidth=0.8)
                
                # Plot BANK_B
                df_b = df_tile[df_tile['bank_type'] == 'BANK_B']
                for channel in df_b['channel'].unique():
                    df_ch = df_b[df_b['channel'] == channel]
                    ax.plot(df_ch['hours'], df_ch['freq_error_GHz'], 
                           color='blue', alpha=0.6, linewidth=0.8)
            
            # Add spec lines at ±20 GHz
            ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-20, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(ax.get_xlim(), -20, 20, color='green', alpha=0.1)
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Frequency Error (GHz)', fontsize=18)
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-100, 100)
            
            # Add tile label
            ax.text(0.02, 0.98, f'Tile {tile_idx+1}', 
                   transform=ax.transAxes, fontsize=16, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend to first subplot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Set A'),
            Line2D([0], [0], color='blue', lw=2, label='Set B')
        ]
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=15)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def _plot_missionmode_power(self, df_expanded, output_path, title_suffix):
        """Plot optical power for all tiles in mission mode."""
        print(f"\nGenerating mission mode power plot: {output_path.name}")
        
        # Convert µW to dBm
        df_expanded['power_dBm'] = 10 * np.log10(df_expanded['pic_mpd_value_uw'] / 1000)
        
        # Create 4x4 grid of subplots (one per tile)
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        # Plot each tile
        for tile_idx in range(16):
            ax = axes[tile_idx]
            df_tile = df_expanded[df_expanded['tile_id'] == tile_idx]
            
            if len(df_tile) > 0:
                # Plot BANK_A
                df_a = df_tile[df_tile['bank_type'] == 'BANK_A']
                for channel in df_a['channel'].unique():
                    df_ch = df_a[df_a['channel'] == channel]
                    ax.plot(df_ch['hours'], df_ch['power_dBm'], 
                           color='red', alpha=0.6, linewidth=0.8)
                
                # Plot BANK_B
                df_b = df_tile[df_tile['bank_type'] == 'BANK_B']
                for channel in df_b['channel'].unique():
                    df_ch = df_b[df_b['channel'] == channel]
                    ax.plot(df_ch['hours'], df_ch['power_dBm'], 
                           color='blue', alpha=0.6, linewidth=0.8)
            
            # Add spec lines (10-17 mW = 10-12.3 dBm)
            spec_min_dBm = 10 * np.log10(10)  # 10 dBm
            spec_max_dBm = 10 * np.log10(17)  # 12.3 dBm
            ax.axhline(y=spec_min_dBm, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=spec_max_dBm, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(ax.get_xlim(), spec_min_dBm, spec_max_dBm, color='green', alpha=0.1)
            
            ax.set_xlabel('Time (hours)', fontsize=18)
            ax.set_ylabel('Optical Power (dBm)', fontsize=18)
            ax.tick_params(labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(5, 15)
            
            # Add tile label
            ax.text(0.02, 0.98, f'Tile {tile_idx+1}', 
                   transform=ax.transAxes, fontsize=16, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend to first subplot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Set A'),
            Line2D([0], [0], color='blue', lw=2, label='Set B')
        ]
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=15)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def _plot_regulators(self, df_expanded, output_path, title_suffix):
        """Plot regulator parameters in 2x2 grid."""
        print(f"\nGenerating regulators plot: {output_path.name}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Get unique DAC orders
        dac_orders = df_expanded['dac_order'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(dac_orders)))
        
        # a) vout_read vs time
        ax = axes[0, 0]
        for i, dac_order in enumerate(dac_orders):
            df_dac = df_expanded[df_expanded['dac_order'] == dac_order]
            ax.plot(df_dac['hours'], df_dac['vout_read'], 
                   label=dac_order, color=colors[i], alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Vout (V)', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        # b) iout_read vs time
        ax = axes[0, 1]
        for i, dac_order in enumerate(dac_orders):
            df_dac = df_expanded[df_expanded['dac_order'] == dac_order]
            ax.plot(df_dac['hours'], df_dac['iout_read'], 
                   label=dac_order, color=colors[i], alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Iout (mA)', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        # c) regulator_power vs time
        ax = axes[1, 0]
        for i, dac_order in enumerate(dac_orders):
            df_dac = df_expanded[df_expanded['dac_order'] == dac_order]
            ax.plot(df_dac['hours'], df_dac['regulator_power'], 
                   label=dac_order, color=colors[i], alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Regulator Power (W)', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        # d) total power vs time (sum of all regulators)
        ax = axes[1, 1]
        df_total = df_expanded.groupby('hours')['regulator_power'].sum().reset_index()
        ax.plot(df_total['hours'], df_total['regulator_power'], 
               color='black', linewidth=2, label='Total Power')
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Total Power (W)', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def analyze_evt_kenya(self):
        """Analyze ips.clm.evt.xlsx Kenya tab."""
        print("\n" + "="*80)
        print("ANALYZING ips.clm.evt.xlsx - Kenya")
        print("="*80)
        
        excel_file = self.data_path / 'ips.clm.evt.xlsx'
        df = pd.read_excel(excel_file, sheet_name='Kenya')
        print(f"Loaded data: {df.shape}")
        
        # Prepare data
        df = self._prepare_data(df, 'Kenya')
        
        # Expand data
        df_reg = self._expand_regulator_data(df)
        df_opt = self._expand_optical_data(df)
        
        print(f"Regulator data points: {len(df_reg)}")
        print(f"Optical data points: {len(df_opt)}")
        
        # Generate plots
        self._plot_missionmode_freqerror(df_opt, 
                                         self.results_path / 'missionmode_freqerror_all_tiles_regevt_kenya.png',
                                         'RegEvt Kenya')
        self._plot_missionmode_power(df_opt,
                                     self.results_path / 'missionmode_power_all_tiles_regevt_kenya.png',
                                     'RegEvt Kenya')
        self._plot_regulators(df_reg,
                             self.results_path / 'missionmode_regulators_regevt_kenya.png',
                             'RegEvt Kenya')
        
        print(f"\nKenya analysis complete!")
    
    def analyze_evt_endeavour(self):
        """Analyze ips.clm.evt.xlsx Endevour tab."""
        print("\n" + "="*80)
        print("ANALYZING ips.clm.evt.xlsx - Endevour")
        print("="*80)
        
        excel_file = self.data_path / 'ips.clm.evt.xlsx'
        df = pd.read_excel(excel_file, sheet_name='Endevour')
        print(f"Loaded data: {df.shape}")
        
        # Prepare data
        df = self._prepare_data(df, 'Endevour')
        
        # Expand data
        df_reg = self._expand_regulator_data(df)
        df_opt = self._expand_optical_data(df)
        
        print(f"Regulator data points: {len(df_reg)}")
        print(f"Optical data points: {len(df_opt)}")
        
        # Generate plots
        self._plot_missionmode_freqerror(df_opt,
                                         self.results_path / 'missionmode_freqerror_all_tiles_regevt_endeavour.png',
                                         'RegEvt Endeavour')
        self._plot_missionmode_power(df_opt,
                                     self.results_path / 'missionmode_power_all_tiles_regevt_endeavour.png',
                                     'RegEvt Endeavour')
        self._plot_regulators(df_reg,
                             self.results_path / 'missionmode_regulators_regevt_endeavour.png',
                             'RegEvt Endeavour')
        
        print(f"\nEndevour analysis complete!")
    
    def analyze_poweropt_kenya(self):
        """Analyze ips.clm.power.optimization.xlsx Kenya tab."""
        print("\n" + "="*80)
        print("ANALYZING ips.clm.power.optimization.xlsx - Kenya")
        print("="*80)
        
        excel_file = self.data_path / 'ips.clm.power.optimization.xlsx'
        df = pd.read_excel(excel_file, sheet_name='Kenya')
        print(f"Loaded data: {df.shape}")
        
        # Prepare data
        df = self._prepare_data(df, 'Kenya')
        
        # Expand data
        df_reg = self._expand_regulator_data(df)
        df_opt = self._expand_optical_data(df)
        
        print(f"Regulator data points: {len(df_reg)}")
        print(f"Optical data points: {len(df_opt)}")
        
        # Generate plots
        self._plot_missionmode_freqerror(df_opt,
                                         self.results_path / 'missionmode_freqerror_all_tiles_poweropt_kenya.png',
                                         'PowerOpt Kenya')
        self._plot_missionmode_power(df_opt,
                                     self.results_path / 'missionmode_power_all_tiles_poweropt_kenya.png',
                                     'PowerOpt Kenya')
        self._plot_regulators(df_reg,
                             self.results_path / 'missionmode_regulators_poweropt_kenya.png',
                             'PowerOpt Kenya')
        
        print(f"\nKenya analysis complete!")
    
    def analyze_poweropt_endeavour(self):
        """Analyze ips.clm.power.optimization.xlsx Endeavour tab."""
        print("\n" + "="*80)
        print("ANALYZING ips.clm.power.optimization.xlsx - Endeavour")
        print("="*80)
        
        excel_file = self.data_path / 'ips.clm.power.optimization.xlsx'
        df = pd.read_excel(excel_file, sheet_name='Endeavour')
        print(f"Loaded data: {df.shape}")
        
        # Prepare data
        df = self._prepare_data(df, 'Endeavour')
        
        # Expand data
        df_reg = self._expand_regulator_data(df)
        df_opt = self._expand_optical_data(df)
        
        print(f"Regulator data points: {len(df_reg)}")
        print(f"Optical data points: {len(df_opt)}")
        
        # Generate plots
        self._plot_missionmode_freqerror(df_opt,
                                         self.results_path / 'missionmode_freqerror_all_tiles_poweropt_endeavour.png',
                                         'PowerOpt Endeavour')
        self._plot_missionmode_power(df_opt,
                                     self.results_path / 'missionmode_power_all_tiles_poweropt_endeavour.png',
                                     'PowerOpt Endeavour')
        self._plot_regulators(df_reg,
                             self.results_path / 'missionmode_regulators_poweropt_endeavour.png',
                             'PowerOpt Endeavour')
        
        print(f"\nEndeavour analysis complete!")
    
