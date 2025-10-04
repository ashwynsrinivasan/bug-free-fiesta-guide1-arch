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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
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
        
        print("\n" + "=" * 80)
        print("Mission Mode analysis completed!")
        print(f"Results saved to: analysis_results/mission_mode/")
        print("=" * 80)
    
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
        ax.set_ylabel('Power (dBm)')
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
        ax1.set_ylabel('Optical Power (dBm)')
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
        ax2.set_ylabel('Optical Power (dBm)')
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
        ax1.set_ylabel('Power (dBm)')
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
        ax2.set_ylabel('Power (dBm)')
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
        ax1.set_ylabel('Power (dBm)')
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
        ax2.set_ylabel('Power (dBm)')
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
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Endeavour
        if endeavour_data is not None:
            self._plot_power_spec(axes[0], endeavour_data, 'Endeavour', full_sn)
        else:
            axes[0].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0].set_title(f'Endeavour - Module {full_sn}')
        
        # Plot Kenya
        if kenya_data is not None:
            self._plot_power_spec(axes[1], kenya_data, 'Kenya', full_sn)
        else:
            axes[1].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1].set_title(f'Kenya - Module {full_sn}')
        
        plt.tight_layout()
        plot_filename = f'missionmode_power_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
    
    def _plot_power_spec(self, ax, df, spec_name, full_sn):
        """Plot power for one specification from mpd_pic column"""
        # Find mpd_pic column
        power_col = None
        for col in df.columns:
            if 'mpd_pic' in col.lower():
                power_col = col
                break
        
        if power_col is None:
            ax.text(0.5, 0.5, 'No mpd_pic data found', ha='center', va='center', fontsize=12)
            return
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
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
        ax.set_ylabel('Optical Power (dBm)')
        ax.set_title(f'{spec_name} - Tile SN: {full_sn} - Mission Mode Power')
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, 21, 1))
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_mission_mode_frequency_error(self, endeavour_data, kenya_data, full_sn):
        """Plot frequency error compared to reference grid"""
        print(f"  Generating mission mode frequency error plot...")
        
        if self.reference_grid is None:
            print(f"  Warning: No reference grid available, skipping frequency error analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Endeavour (Bank 0 uses set_a, Bank 1 uses set_b)
        if endeavour_data is not None:
            self._plot_freq_error_spec(axes[0], endeavour_data, 'Endeavour', full_sn)
        else:
            axes[0].text(0.5, 0.5, 'No Endeavour Data', ha='center', va='center', fontsize=14)
            axes[0].set_title(f'Endeavour - Module {full_sn}')
        
        # Plot Kenya (Bank 0 uses set_a, Bank 1 uses set_b)
        if kenya_data is not None:
            self._plot_freq_error_spec(axes[1], kenya_data, 'Kenya', full_sn)
        else:
            axes[1].text(0.5, 0.5, 'No Kenya Data', ha='center', va='center', fontsize=14)
            axes[1].set_title(f'Kenya - Module {full_sn}')
        
        plt.tight_layout()
        plot_filename = f'missionmode_freqerror_{full_sn}.png'
        plt.savefig(self.mission_mode_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_filename}")
    
    def _plot_freq_error_spec(self, ax, df, spec_name, full_sn):
        """Plot frequency error for one specification using wavelength column
        Bank 0 uses set_a, Bank 1 uses set_b"""
        # Find wavelength column
        wl_col = None
        for col in df.columns:
            if col.lower() == 'wavelength':
                wl_col = col
                break
        
        if wl_col is None:
            ax.text(0.5, 0.5, 'No wavelength data', ha='center', va='center', fontsize=12)
            return
        
        # Color maps for channels
        colors_b0 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
        colors_b1 = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))
        
        # Speed of light constant
        c_speed_light = 299792.458  # Speed of light in nm*THz
        
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
        ax.set_ylim(-40, 40)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
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
        
        # Set labels
        ax.set_xlabel('Tile SN')
        ylabel = 'Power (dBm)' if data_type == 'power' else 'Frequency Error (GHz)'
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        center_positions = [i * 2 + 0.25 for i in range(len(all_sns))]
        ax.set_xticks(center_positions)
        ax.set_xticklabels(all_sns, rotation=45, ha='right')
        
        # Legend
        legend_elements = [Patch(facecolor=bank0_color, alpha=0.5, label='Bank 0'),
                          Patch(facecolor=bank1_color, alpha=0.5, label='Bank 1')]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_mission_mode_summary(self, modules_data):
        """Create mission mode summary plot with 2x2 subplots"""
        print(f"\n{'='*60}")
        print("Creating Mission Mode Summary Plot")
        print(f"{'='*60}")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Process data for each module
        # Structure: {sn: {bank: {channel: value}}}
        endeavour_power_data = {}
        endeavour_freq_data = {}
        kenya_power_data = {}
        kenya_freq_data = {}
        
        # Speed of light constant for frequency calculation
        c_speed_light = 299792.458
        
        for full_sn, data in sorted(modules_data.items()):
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
        
        # Plot 1: Endeavour Power (top-left)
        ax = axes[0, 0]
        self._plot_boxplot_with_scatter(ax, endeavour_power_data, all_sns, 'endeavour', 'power', 
                                        'Endeavour - Mission Mode Power')
        
        # Plot 2: Kenya Power (top-right)
        ax = axes[0, 1]
        self._plot_boxplot_with_scatter(ax, kenya_power_data, all_sns, 'kenya', 'power',
                                        'Kenya - Mission Mode Power')
        
        # Plot 3: Endeavour Frequency Error (bottom-left)
        ax = axes[1, 0]
        self._plot_boxplot_with_scatter(ax, endeavour_freq_data, all_sns, 'endeavour', 'freq',
                                        'Endeavour - Mission Mode Frequency Error')
        
        # Plot 4: Kenya Frequency Error (bottom-right)
        ax = axes[1, 1]
        self._plot_boxplot_with_scatter(ax, kenya_freq_data, all_sns, 'kenya', 'freq',
                                        'Kenya - Mission Mode Frequency Error')
        
        plt.tight_layout()
        
        # Save to analysis_results root
        summary_path = self.results_path
        plot_filename = 'missionmode_summary.png'
        plt.savefig(summary_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Mission Mode summary plot saved: {plot_filename}")
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
            # Convert uW to mW and calculate power
            for i, col in enumerate(power_pic_columns):
                linestyle = '-' if i < 8 else '--'
                # Convert to mW: power_mw = power_uw / 1000
                power_mw = df[col] / 1000.0
                ax.plot(time, power_mw, linewidth=1.5, linestyle=linestyle,
                       label=f'Laser_{i}', alpha=0.8)
            
            # Add specification limits
            spec_lower = spec_name.lower()
            if self.specifications and spec_lower in self.specifications:
                spec_data = self.specifications[spec_lower]
                
                # Add min/max power limits
                if 'min_power' in spec_data:
                    min_power_dbm = spec_data['min_power']['value']
                    # Convert dBm to mW: P(mW) = 10^(P(dBm)/10)
                    min_power_mw = 10 ** (min_power_dbm / 10.0)
                    ax.axhline(y=min_power_mw, color='red', linestyle='--', linewidth=1.5, 
                              label=f'Min: {min_power_dbm}dBm', alpha=0.7)
                
                if 'max_power' in spec_data:
                    max_power_dbm = spec_data['max_power']['value']
                    max_power_mw = 10 ** (max_power_dbm / 10.0)
                    ax.axhline(y=max_power_mw, color='red', linestyle='--', linewidth=1.5,
                              label=f'Max: {max_power_dbm}dBm', alpha=0.7)
                
                # Add typical power if available
                if 'typical_power' in spec_data:
                    typ_power_dbm = spec_data['typical_power']['value']
                    typ_power_mw = 10 ** (typ_power_dbm / 10.0)
                    ax.axhline(y=typ_power_mw, color='green', linestyle=':', linewidth=1.5,
                              label=f'Typ: {typ_power_dbm}dBm', alpha=0.7)
            
            ax.set_ylabel('Power (mW)')
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
    
