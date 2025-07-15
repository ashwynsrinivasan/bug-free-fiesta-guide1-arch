#!/usr/bin/env python3
"""
TP2-1 Laser Data Import and DeltaWave Analysis Script
====================================================

This script imports all *Laser.csv files from the TP2-1 data folder
and creates DeltaWave plots showing the change in peak wavelength from the first 120mA datapoint for each tile.
"""

import pandas as pd
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.api.types import is_datetime64_any_dtype
import json

class TP2P1CombinerAnalyzers:
    def __init__(self):
        """
        Initialize the TP2-1 Combiner Analyzer.
        """
        self.laser_data = []
        self.laser_files = []
        self.combined_data = None
        self.voa_data = []
        self.voa_files = []
        self.combined_voa_data = None
        self.output_dir = Path(__file__).parent / "plots" / "TP2-1"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def load_laser_files(self, max_files=None):
        """
        Load *Laser.csv files from the TP2-1 directory.
        
        Args:
            max_files (int, optional): Maximum number of files to load. If None, load all files.
            
        Returns:
            list: List of DataFrames, one for each Laser.csv file
        """
        # Get the script directory and construct path to TP2-1 data folder
        script_dir = Path(__file__).parent
        tp21_data_path = script_dir / "../TP2-1"
        
        # Find all Laser.csv files
        all_laser_files = sorted(glob.glob(str(tp21_data_path / "*Laser.csv")))
        
        # Limit files if max_files is specified
        if max_files is not None:
            self.laser_files = all_laser_files[:max_files]
            print(f"Found {len(all_laser_files)} Laser.csv files in TP2-1 directory")
            print(f"Loading first {len(self.laser_files)} files for analysis")
        else:
            self.laser_files = all_laser_files
            print(f"Found {len(all_laser_files)} Laser.csv files in TP2-1 directory")
            print(f"Loading all {len(self.laser_files)} files for analysis")
        
        # Load each file into a DataFrame
        self.laser_data = []
        for file_path in self.laser_files:
            try:
                df = pd.read_csv(file_path)
                filename = Path(file_path).name
                df['filename'] = filename
                print(f"Loaded: {filename} - Shape: {df.shape}")
                self.laser_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Successfully loaded {len(self.laser_data)} Laser.csv files")
        return self.laser_data
    
    def load_voa_files(self, max_files=None):
        """
        Load *VOA.csv files from the TP2-1 directory.
        
        Args:
            max_files (int, optional): Maximum number of files to load. If None, load all files.
            
        Returns:
            list: List of DataFrames, one for each VOA.csv file
        """
        # Get the script directory and construct path to TP2-1 data folder
        script_dir = Path(__file__).parent
        tp21_data_path = script_dir / "../TP2-1"
        
        # Find all VOA.csv files
        all_voa_files = sorted(glob.glob(str(tp21_data_path / "*VOA.csv")))
        
        # Limit files if max_files is specified
        if max_files is not None:
            self.voa_files = all_voa_files[:max_files]
            print(f"Found {len(all_voa_files)} VOA.csv files in TP2-1 directory")
            print(f"Loading first {len(self.voa_files)} files for analysis")
        else:
            self.voa_files = all_voa_files
            print(f"Found {len(all_voa_files)} VOA.csv files in TP2-1 directory")
            print(f"Loading all {len(self.voa_files)} files for analysis")
        
        # Load each file into a DataFrame
        self.voa_data = []
        for file_path in self.voa_files:
            try:
                df = pd.read_csv(file_path)
                filename = Path(file_path).name
                df['filename'] = filename
                print(f"Loaded: {filename} - Shape: {df.shape}")
                self.voa_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Successfully loaded {len(self.voa_data)} VOA.csv files")
        return self.voa_data
    
    def combine_laser_data(self):
        """
        Combine all loaded laser data into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined data with Tile_SN column
        """
        if not self.laser_data:
            print("No laser data loaded!")
            return None
        
        # Combine all DataFrames
        self.combined_data = pd.concat(self.laser_data, ignore_index=True)
        
        # Extract Tile_SN from filename
        self.combined_data['Tile_SN'] = self.combined_data['filename'].apply(self.extract_tile_sn)
        
        # Clean up numeric columns including PeakWave and PeakPower columns
        numeric_columns = ['Set Laser(mA)', 'Power(mW)', 'Set PIC Temp(C)', 'Set MUX Temp(C)', 'Bank', 'Channel']
        peakwave_columns = [f'PeakWave_{i}(nm)' for i in range(8)]
        peakpower_columns = [f'PeakPower_{i}(dBm)' for i in range(8)]
        numeric_columns.extend(peakwave_columns)
        numeric_columns.extend(peakpower_columns)
        
        for col in numeric_columns:
            if col in self.combined_data.columns:
                self.combined_data[col] = self.combined_data[col].replace('-', pd.NA)
                self.combined_data[col] = pd.to_numeric(self.combined_data[col], errors='coerce')
        
        # Convert time column if it exists
        if 'Time' in self.combined_data.columns:
            self.combined_data['Time'] = pd.to_datetime(self.combined_data['Time'], format='mixed', errors='coerce')
        
        print(f"Combined data shape: {self.combined_data.shape}")
        print(f"Combined data columns: {list(self.combined_data.columns)}")
        print(f"Unique tiles: {self.combined_data['Tile_SN'].dropna().unique()}")
        
        return self.combined_data
    
    def combine_voa_data(self):
        """
        Combine all loaded VOA data into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined data with Tile_SN column
        """
        if not self.voa_data:
            print("No VOA data loaded!")
            return None
        
        # Combine all DataFrames
        self.combined_voa_data = pd.concat(self.voa_data, ignore_index=True)
        
        # Extract Tile_SN from filename
        self.combined_voa_data['Tile_SN'] = self.combined_voa_data['filename'].apply(self.extract_tile_sn)
        
        # Clean up numeric columns including PeakWave, PeakPower, MPD_PIC, MPD_MUX columns
        numeric_columns = ['Set VOA(mA)', 'Power(mW)', 'Set PIC Temp(C)', 'Set MUX Temp(C)', 'Bank', 'Channel']
        peakwave_columns = [f'PeakWave_{i}(nm)' for i in range(8)]
        peakpower_columns = [f'PeakPower_{i}(dBm)' for i in range(8)]
        mpd_pic_columns = [f'MPD_PIC_{i}(uA)' for i in range(8)]
        mpd_mux_columns = [f'MPD_MUX_{i}(uA)' for i in range(8)]
        
        numeric_columns.extend(peakwave_columns)
        numeric_columns.extend(peakpower_columns)
        numeric_columns.extend(mpd_pic_columns)
        numeric_columns.extend(mpd_mux_columns)
        
        for col in numeric_columns:
            if col in self.combined_voa_data.columns:
                self.combined_voa_data[col] = self.combined_voa_data[col].replace('-', pd.NA)
                self.combined_voa_data[col] = pd.to_numeric(self.combined_voa_data[col], errors='coerce')
        
        # Convert time column if it exists
        if 'Time' in self.combined_voa_data.columns:
            self.combined_voa_data['Time'] = pd.to_datetime(self.combined_voa_data['Time'], format='mixed', errors='coerce')
        
        print(f"Combined VOA data shape: {self.combined_voa_data.shape}")
        print(f"Combined VOA data columns: {list(self.combined_voa_data.columns)}")
        print(f"Unique tiles in VOA data: {self.combined_voa_data['Tile_SN'].dropna().unique()}")
        
        return self.combined_voa_data
    
    def extract_tile_sn(self, filename):
        """
        Extract tile serial number from filename.
        
        Args:
            filename (str): Filename of the CSV file
            
        Returns:
            str: Tile serial number (e.g., 'Y25170084')
        """
        match = re.search(r'-Y(\d+)-TP2-1', filename)
        if match:
            return f"Y{match.group(1)}"
        return None
    
    def plot_deltawave_per_tile(self):
        """
        Create DeltaWave plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows the change in peak wavelength from the first 120mA datapoint.
        """
        if self.combined_data is None:
            print("No combined data available! Run combine_laser_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating DeltaWave plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_data[self.combined_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile (use PIC temp for wavelength analysis)
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    channel_slopes = {}  # Store individual slopes for each channel
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all PeakWave channels (0-7) for this specific channel setting
                            for peakwave_idx in range(8):
                                peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                                
                                # Check if column exists in the DataFrame
                                if peakwave_col in channel_data.columns.tolist():
                                    # Group by Set Laser current and take mean PeakWave
                                    grouped_data = channel_data.groupby('Set Laser(mA)')[peakwave_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        # Find the baseline (first 120mA datapoint)
                                        baseline_data = grouped_data[grouped_data['Set Laser(mA)'] == 120].copy()
                                        
                                        if len(baseline_data) > 0:
                                            baseline_wavelength = baseline_data[peakwave_col].values[0]
                                            
                                            # Calculate delta from baseline
                                            grouped_data = grouped_data.copy()
                                            grouped_data['delta_wavelength'] = grouped_data[peakwave_col] - baseline_wavelength
                                            
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakWave_{peakwave_idx}: {len(grouped_data)} points, baseline: {baseline_wavelength:.3f} nm, delta range: {grouped_data['delta_wavelength'].min():.3f} to {grouped_data['delta_wavelength'].max():.3f} nm")
                                            
                                            # Calculate individual slope for this channel (only for first temperature to avoid duplicates)
                                            if temp_idx == 0 and len(grouped_data) > 1:
                                                laser_currents = grouped_data['Set Laser(mA)'].values
                                                delta_wavelengths = grouped_data['delta_wavelength'].values
                                                
                                                if len(laser_currents) >= 2:
                                                    slope = np.polyfit(laser_currents, delta_wavelengths, 1)[0]
                                                    slope_pm = slope * 1000  # Convert nm/mA to pm/mA
                                                    channel_slopes[f'Ch{peakwave_idx}'] = slope_pm
                                                    print(f"        Individual slope for Ch{channel}, Bank {bank}, PeakWave_{peakwave_idx}: {slope_pm:.2f} pm/mA")
                                            
                                            # Use different colors for different PeakWave channels
                                            color_idx = peakwave_idx % len(colors)
                                            # Use different line styles for different temperatures
                                            linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                            
                                            ax.plot(grouped_data['Set Laser(mA)'], grouped_data['delta_wavelength'], 
                                                   marker='o', linewidth=1.5, markersize=2, 
                                                   color=colors[color_idx], linestyle=linestyle,
                                                   label=f'Channel {peakwave_idx}' if temp_idx == 0 else "")
                                            plotted = True
                                        else:
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakWave_{peakwave_idx}: No 120mA baseline found")
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set Laser Current (mA)')
                    ax.set_ylabel('ΔWavelength (nm)')
                    ax.set_xlim(120, 170)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Add zero reference line
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper left', fontsize=8)
                        
                        # Add individual channel slopes annotation
                        if channel_slopes:
                            slope_text = "Slopes (pm/mA):\n"
                            for ch_name, slope in channel_slopes.items():
                                slope_text += f"{ch_name}: {slope:.2f}\n"
                            
                            ax.text(0.98, 0.98, slope_text.strip(), 
                                   transform=ax.transAxes, fontsize=8, 
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'DeltaWave (Change from 120mA baseline) - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"DeltaWave_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ DeltaWave plot saved: {plot_filename}")
    
    def plot_deltapower_per_tile(self):
        """
        Create DeltaPower plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows the change in peak power from the first 120mA datapoint.
        """
        if self.combined_data is None:
            print("No combined data available! Run combine_laser_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating DeltaPower plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_data[self.combined_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile (use PIC temp for power analysis)
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    channel_slopes = {}  # Store individual slopes for each channel
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all PeakPower channels (0-7) for this specific channel setting
                            for peakpower_idx in range(8):
                                peakpower_col = f'PeakPower_{peakpower_idx}(dBm)'
                                
                                # Check if column exists in the DataFrame
                                if peakpower_col in channel_data.columns.tolist():
                                    # Group by Set Laser current and take mean PeakPower
                                    grouped_data = channel_data.groupby('Set Laser(mA)')[peakpower_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        # Find the baseline (first 120mA datapoint)
                                        baseline_data = grouped_data[grouped_data['Set Laser(mA)'] == 120].copy()
                                        
                                        if len(baseline_data) > 0:
                                            baseline_power = baseline_data[peakpower_col].values[0]
                                            
                                            # Calculate delta from baseline
                                            grouped_data = grouped_data.copy()
                                            grouped_data['delta_power'] = grouped_data[peakpower_col] - baseline_power
                                            
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakPower_{peakpower_idx}: {len(grouped_data)} points, baseline: {baseline_power:.3f} dBm, delta range: {grouped_data['delta_power'].min():.3f} to {grouped_data['delta_power'].max():.3f} dBm")
                                            
                                            # Calculate individual slope for this channel (only for first temperature to avoid duplicates)
                                            if temp_idx == 0 and len(grouped_data) > 1:
                                                laser_currents = grouped_data['Set Laser(mA)'].values
                                                delta_powers = grouped_data['delta_power'].values
                                                
                                                if len(laser_currents) >= 2:
                                                    slope = np.polyfit(laser_currents, delta_powers, 1)[0]
                                                    channel_slopes[f'Ch{peakpower_idx}'] = slope
                                                    print(f"        Individual slope for Ch{channel}, Bank {bank}, PeakPower_{peakpower_idx}: {slope:.4f} dBm/mA")
                                            
                                            # Use different colors for different PeakPower channels
                                            color_idx = peakpower_idx % len(colors)
                                            # Use different line styles for different temperatures
                                            linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                            
                                            ax.plot(grouped_data['Set Laser(mA)'], grouped_data['delta_power'], 
                                                   marker='o', linewidth=1.5, markersize=2, 
                                                   color=colors[color_idx], linestyle=linestyle,
                                                   label=f'Channel {peakpower_idx}' if temp_idx == 0 else "")
                                            plotted = True
                                        else:
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakPower_{peakpower_idx}: No 120mA baseline found")
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set Laser Current (mA)')
                    ax.set_ylabel('ΔPower (dBm)')
                    ax.set_xlim(120, 170)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Add zero reference line
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper left', fontsize=8)
                        
                        # Add individual channel slopes annotation
                        if channel_slopes:
                            slope_text = "Slopes (dBm/mA):\n"
                            for ch_name, slope in channel_slopes.items():
                                slope_text += f"{ch_name}: {slope:.4f}\n"
                            
                            ax.text(0.98, 0.98, slope_text.strip(), 
                                   transform=ax.transAxes, fontsize=8, 
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'DeltaPower (Change from 120mA baseline) - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"DeltaPower_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ DeltaPower plot saved: {plot_filename}")
    
    def plot_crosstalk_summary_per_tile(self):
        """
        Create cross-talk summary plots for each tile showing:
        - Principal DeltaWave slope (main channel)
        - 1st order cross-talk (adjacent channels)
        - 2nd order cross-talk (channels 2 positions away)
        - 3rd order cross-talk (channels 3 positions away)
        """
        if self.combined_data is None:
            print("No combined data available! Run combine_laser_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_data['Tile_SN'].dropna().unique()
        
        for tile_sn in unique_tiles:
            print(f"Creating cross-talk summary plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_data[self.combined_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile (use first temperature for analysis)
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            temp = available_temps[0]  # Use first temperature
            
            # Calculate slopes for all channel/bank/peakwave combinations
            slopes_data = {}
            
            for channel in range(8):  # Channels 0-7
                for bank in [0, 1]:  # Banks 0-1
                    key = f"Ch{channel}_Bank{bank}"
                    slopes_data[key] = {}
                    
                    # Get data for this channel/bank/temperature combination
                    channel_data = tile_data[
                        (tile_data['Channel'] == channel) & 
                        (tile_data['Bank'] == bank) & 
                        (tile_data['Set PIC Temp(C)'] == temp)
                    ].copy()
                    
                    if len(channel_data) > 0:
                        # Calculate slopes for each PeakWave sensor
                        for peakwave_idx in range(8):
                            peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                            
                            if peakwave_col in channel_data.columns.tolist():
                                # Group by Set Laser current and take mean PeakWave
                                grouped_data = channel_data.groupby('Set Laser(mA)')[peakwave_col].mean().reset_index()
                                
                                if len(grouped_data) > 1:
                                    # Find the baseline (first 120mA datapoint)
                                    baseline_data = grouped_data[grouped_data['Set Laser(mA)'] == 120].copy()
                                    
                                    if len(baseline_data) > 0:
                                        baseline_wavelength = baseline_data[peakwave_col].values[0]
                                        
                                        # Calculate delta from baseline
                                        grouped_data = grouped_data.copy()
                                        grouped_data['delta_wavelength'] = grouped_data[peakwave_col] - baseline_wavelength
                                        
                                        # Calculate slope
                                        laser_currents = grouped_data['Set Laser(mA)'].values
                                        delta_wavelengths = grouped_data['delta_wavelength'].values
                                        
                                        if len(laser_currents) >= 2:
                                            slope = np.polyfit(laser_currents, delta_wavelengths, 1)[0]
                                            slope_pm = slope * 1000  # Convert nm/mA to pm/mA
                                            slopes_data[key][f'PeakWave_{peakwave_idx}'] = slope_pm
            
            # Create summary plot: 8 channels × 2 banks = 16 subplots
            fig, axs = plt.subplots(8, 2, figsize=(12, 20))
            
            for channel in range(8):  # Channels 0-7
                for bank in [0, 1]:  # Banks 0-1
                    ax = axs[channel, bank]
                    key = f"Ch{channel}_Bank{bank}"
                    
                    if key in slopes_data and slopes_data[key]:
                        # Categorize slopes by cross-talk order
                        principal_slope = slopes_data[key].get(f'PeakWave_{channel}', 0)
                        
                        # 1st order cross-talk: adjacent channels (±1)
                        first_order = []
                        for offset in [-1, 1]:
                            adjacent_ch = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{adjacent_ch}', 0)
                            first_order.append(slope)
                        
                        # 2nd order cross-talk: channels ±2 positions away
                        second_order = []
                        for offset in [-2, 2]:
                            ch_2nd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_2nd}', 0)
                            second_order.append(slope)
                        
                        # 3rd order cross-talk: channels ±3 positions away
                        third_order = []
                        for offset in [-3, 3]:
                            ch_3rd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_3rd}', 0)
                            third_order.append(slope)
                        
                        # 4th order cross-talk: channel ±4 positions away (opposite)
                        fourth_order = slopes_data[key].get(f'PeakWave_{(channel + 4) % 8}', 0)
                        
                        # Create bar plot
                        categories = ['Principal', '1st Order\n(Mean)', '2nd Order\n(Mean)', '3rd Order\n(Mean)', '4th Order']
                        values = [
                            principal_slope,
                            np.mean(first_order),
                            np.mean(second_order),
                            np.mean(third_order),
                            fourth_order
                        ]
                        
                        # Color coding: Principal=blue, Cross-talk=red gradient
                        colors = ['#1f77b4', '#ff9999', '#ff6666', '#ff3333', '#ff0000']
                        
                        bars = ax.bar(categories, values, color=colors, alpha=0.8)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                                   f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
                        
                        ax.set_title(f'Ch{channel} - Bank {bank}', fontsize=10)
                        ax.set_ylabel('Slope (pm/mA)', fontsize=8)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                        ax.grid(True, alpha=0.3)
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        
                        # Add individual values annotation
                        info_text = f"Principal: {principal_slope:.1f}\n"
                        info_text += f"1st: {first_order[0]:.1f}, {first_order[1]:.1f}\n"
                        info_text += f"2nd: {second_order[0]:.1f}, {second_order[1]:.1f}\n"
                        info_text += f"3rd: {third_order[0]:.1f}, {third_order[1]:.1f}\n"
                        info_text += f"4th: {fourth_order:.1f}"
                        
                        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=6,
                               verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Ch{channel} - Bank {bank}', fontsize=10)
            
            plt.suptitle(f'Cross-talk Analysis Summary - Tile {tile_sn}', fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            
            # Save the plot
            plot_filename = f"CrossTalk_Summary_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Cross-talk summary plot saved: {plot_filename}")
    
    def plot_combined_crosstalk_summary(self):
        """
        Create a combined cross-talk summary plot for all tiles.
        Layout: 5 rows × 2 columns (Principal, 1st, 2nd, 3rd, 4th order vs Bank 0/1)
        Shows box plots and scatter plots of wavelength slopes vs tile SN.
        """
        if self.combined_data is None:
            print("No combined data available! Run combine_laser_data() first.")
            return
        
        print("Creating combined cross-talk summary plot...")
        
        # Get unique tiles and sort them
        unique_tiles = sorted(self.combined_data['Tile_SN'].dropna().unique())
        
        # Collect all cross-talk data
        crosstalk_data = {
            'Principal': {'Bank0': [], 'Bank1': []},
            '1st Order': {'Bank0': [], 'Bank1': []},
            '2nd Order': {'Bank0': [], 'Bank1': []},
            '3rd Order': {'Bank0': [], 'Bank1': []},
            '4th Order': {'Bank0': [], 'Bank1': []}
        }
        
        # Process each tile
        for tile_sn in unique_tiles:
            print(f"  Processing tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_data[self.combined_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile (use first temperature for analysis)
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            temp = available_temps[0]  # Use first temperature
            
            # Calculate slopes for all channel/bank/peakwave combinations
            slopes_data = {}
            
            for channel in range(8):  # Channels 0-7
                for bank in [0, 1]:  # Banks 0-1
                    key = f"Ch{channel}_Bank{bank}"
                    slopes_data[key] = {}
                    
                    # Get data for this channel/bank/temperature combination
                    channel_data = tile_data[
                        (tile_data['Channel'] == channel) & 
                        (tile_data['Bank'] == bank) & 
                        (tile_data['Set PIC Temp(C)'] == temp)
                    ].copy()
                    
                    if len(channel_data) > 0:
                        # Calculate slopes for each PeakWave sensor
                        for peakwave_idx in range(8):
                            peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                            
                            if peakwave_col in channel_data.columns.tolist():
                                # Group by Set Laser current and take mean PeakWave
                                grouped_data = channel_data.groupby('Set Laser(mA)')[peakwave_col].mean().reset_index()
                                
                                if len(grouped_data) > 1:
                                    # Find the baseline (first 120mA datapoint)
                                    baseline_data = grouped_data[grouped_data['Set Laser(mA)'] == 120].copy()
                                    
                                    if len(baseline_data) > 0:
                                        baseline_wavelength = baseline_data[peakwave_col].values[0]
                                        
                                        # Calculate delta from baseline
                                        grouped_data = grouped_data.copy()
                                        grouped_data['delta_wavelength'] = grouped_data[peakwave_col] - baseline_wavelength
                                        
                                        # Calculate slope
                                        laser_currents = grouped_data['Set Laser(mA)'].values
                                        delta_wavelengths = grouped_data['delta_wavelength'].values
                                        
                                        if len(laser_currents) >= 2:
                                            slope = np.polyfit(laser_currents, delta_wavelengths, 1)[0]
                                            slope_pm = slope * 1000  # Convert nm/mA to pm/mA
                                            slopes_data[key][f'PeakWave_{peakwave_idx}'] = slope_pm
            
            # Categorize slopes by cross-talk order for each channel/bank
            for channel in range(8):
                for bank in [0, 1]:
                    key = f"Ch{channel}_Bank{bank}"
                    bank_key = f"Bank{bank}"
                    
                    if key in slopes_data and slopes_data[key]:
                        # Principal slope
                        principal_slope = slopes_data[key].get(f'PeakWave_{channel}', np.nan)
                        if not np.isnan(principal_slope):
                            crosstalk_data['Principal'][bank_key].append({
                                'tile': tile_sn,
                                'channel': channel,
                                'slope': principal_slope
                            })
                        
                        # 1st order cross-talk: adjacent channels (±1)
                        for offset in [-1, 1]:
                            adjacent_ch = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{adjacent_ch}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['1st Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 2nd order cross-talk: channels ±2 positions away
                        for offset in [-2, 2]:
                            ch_2nd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_2nd}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['2nd Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 3rd order cross-talk: channels ±3 positions away
                        for offset in [-3, 3]:
                            ch_3rd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_3rd}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['3rd Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 4th order cross-talk: channel ±4 positions away (opposite)
                        fourth_ch = (channel + 4) % 8
                        slope = slopes_data[key].get(f'PeakWave_{fourth_ch}', np.nan)
                        if not np.isnan(slope):
                            crosstalk_data['4th Order'][bank_key].append({
                                'tile': tile_sn,
                                'channel': channel,
                                'slope': slope
                            })
        
        # Create the combined plot: 5 rows × 2 columns
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        
        orders = ['Principal', '1st Order', '2nd Order', '3rd Order', '4th Order']
        colors = ['#1f77b4', '#ff9999', '#ff6666', '#ff3333', '#ff0000']
        
        for row_idx, (order, color) in enumerate(zip(orders, colors)):
            for col_idx, bank in enumerate(['Bank0', 'Bank1']):
                ax = axes[row_idx, col_idx]
                
                # Get data for this order/bank combination
                data = crosstalk_data[order][bank]
                
                if data:
                    # Convert to DataFrame for easier manipulation
                    df = pd.DataFrame(data)
                    
                    # Create box plot data
                    box_data = []
                    tile_positions = []
                    tile_labels = []
                    
                    for i, tile in enumerate(unique_tiles):
                        tile_data = df[df['tile'] == tile]['slope'].values
                        if len(tile_data) > 0:
                            box_data.append(tile_data)
                            tile_positions.append(i)
                            tile_labels.append(tile)
                    
                    if box_data:
                        # Create box plot
                        bp = ax.boxplot(box_data, positions=tile_positions, widths=0.6, 
                                       patch_artist=True, showfliers=False)
                        
                        # Style box plots
                        for patch in bp['boxes']:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        # Add scatter plot overlay
                        for i, (tile, pos) in enumerate(zip(tile_labels, tile_positions)):
                            tile_slopes = df[df['tile'] == tile]['slope'].values
                            if len(tile_slopes) > 0:
                                # Add jitter for better visibility
                                jitter = np.random.normal(0, 0.1, len(tile_slopes))
                                ax.scatter(pos + jitter, tile_slopes, 
                                          alpha=0.6, s=20, color=color, edgecolors='black', linewidth=0.5)
                                
                                # Add average annotation
                                avg_slope = np.mean(tile_slopes)
                                ax.annotate(f'{avg_slope:.3f}', 
                                           xy=(pos, avg_slope), 
                                           xytext=(pos, avg_slope + 0.5),
                                           ha='center', va='bottom', fontsize=8,
                                           rotation=90, color='red')
                    
                    # Set labels and title
                    ax.set_title(f'{order} - Bank {col_idx}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Tile SN', fontsize=10)
                    ax.set_ylabel('Wavelength Slope (pm/mA)', fontsize=10)
                    
                    # Set x-axis
                    ax.set_xticks(range(len(unique_tiles)))
                    ax.set_xticklabels(unique_tiles, rotation=45, ha='right', fontsize=8)
                    
                    # Set y-axis range
                    if row_idx == 0:  # Principal
                        ax.set_ylim(2, 8)
                    elif row_idx in [2, 3, 4]:  # 2nd, 3rd, 4th Order
                        ax.set_ylim(-1, 1)
                    else:  # 1st Order (keep original range)
                        ax.set_ylim(-10, 10)
                    
                    # Grid
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{order} - Bank {col_idx}', fontsize=12, fontweight='bold')
        
        plt.suptitle('Drive Current Cross-talk vs Tile - Combined Summary', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        # Save the plot to plots folder (not plots/TP2-1)
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = "tp2p1_drive_current_crosstalk_vs_tile_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Combined cross-talk summary plot saved: {plot_filename}")
    
    def export_to_xarray(self):
        """Export all combined data as an xarray DataArray and save as a .nc file with TP2-1 attributes."""
        import xarray as xr
        from datetime import datetime
        
        if self.combined_data is None:
            print("No combined data to export!")
            return
        
        # Prepare combined data
        combined_df = self.combined_data.copy()
        
        # Convert datetime columns to ISO strings
        for col in combined_df.columns:
            if is_datetime64_any_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].astype(str)
        
        # Convert to xarray Dataset
        data_xr = xr.Dataset.from_dataframe(combined_df)
        
        # Add attributes (only serializable types)
        data_xr.attrs["test_point"] = "TP2-1"
        data_xr.attrs["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_xr.attrs["data_points"] = len(self.combined_data)
        data_xr.attrs["unique_tiles"] = len(self.combined_data['Tile_SN'].dropna().unique())
        data_xr.attrs["laser_files_count"] = len(self.laser_files)
        data_xr.attrs["description"] = "TP2-1 Laser data analysis including DeltaWave and DeltaPower measurements"
        
        # Add column information
        data_xr.attrs["columns"] = list(combined_df.columns)
        
        # Save as NetCDF
        nc_path = self.data_dir / "tp2p1_combined_data.nc"
        data_xr.to_netcdf(nc_path, engine='netcdf4')
        print(f"✅ Exported combined data to NetCDF: {nc_path}")
    
    def run_analysis(self):
        """
        Run complete analysis: load files, combine data, and create plots.
        """
        print("Starting TP2-1 DeltaWave and DeltaPower Data Analysis...")
        print("=" * 50)
        
        # Load all laser files
        self.load_laser_files()
        
        # Combine data
        self.combine_laser_data()
        
        # Create DeltaWave plots
        self.plot_deltawave_per_tile()
        
        # Create DeltaPower plots
        self.plot_deltapower_per_tile()
        
        # Create cross-talk summary plots
        self.plot_crosstalk_summary_per_tile()
        
        # Create combined cross-talk summary plot
        self.plot_combined_crosstalk_summary()
        
        # Export combined data to NetCDF
        print("\n" + "=" * 50)
        print("TP2-1 EXPORT TO NETCDF")
        print("=" * 50)
        self.export_to_xarray()
        
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print(f"Individual plots saved in: {self.output_dir}")
        print(f"Combined summary saved in: {Path(__file__).parent / 'plots'}")
        print(f"Data exported to: {self.data_dir}")
        
        # List generated files
        unique_tiles = self.combined_data['Tile_SN'].dropna().unique() if self.combined_data is not None else []
        print("\n📋 Generated files:")
        for tile_sn in unique_tiles:
            print(f"   • DeltaWave_{tile_sn}.png")
            print(f"   • DeltaPower_{tile_sn}.png")
            print(f"   • CrossTalk_Summary_{tile_sn}.png")
        print(f"   • tp2p1_drive_current_crosstalk_vs_tile_combined.png")
        print(f"   • tp2p1_combined_data.nc (in data folder)")
        print("=" * 50)

    def plot_voa_peakpower_per_tile(self):
        """
        Create VOA PeakPower plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows peak power vs Set VOA current.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA PeakPower plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all PeakPower channels (0-7) for this specific channel setting
                            for peakpower_idx in range(8):
                                peakpower_col = f'PeakPower_{peakpower_idx}(dBm)'
                                
                                # Check if column exists in the DataFrame
                                if peakpower_col in channel_data.columns.tolist():
                                    # Group by Set VOA current and take mean PeakPower
                                    grouped_data = channel_data.groupby('Set VOA(mA)')[peakpower_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakPower_{peakpower_idx}: {len(grouped_data)} points, power range: {grouped_data[peakpower_col].min():.3f} to {grouped_data[peakpower_col].max():.3f} dBm")
                                        
                                        # Use different colors for different PeakPower channels
                                        color_idx = peakpower_idx % len(colors)
                                        # Use different line styles for different temperatures
                                        linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                        
                                        ax.plot(grouped_data['Set VOA(mA)'], grouped_data[peakpower_col], 
                                               marker='o', linewidth=1.5, markersize=2, 
                                               color=colors[color_idx], linestyle=linestyle,
                                               label=f'Channel {peakpower_idx}' if temp_idx == 0 else "")
                                        plotted = True
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set VOA Current (mA)')
                    ax.set_ylabel('Peak Power (dBm)')
                    ax.set_xlim(0, 20)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper right', fontsize=8)
            
            plt.suptitle(f'VOA PeakPower vs Set VOA Current - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"VOA_PeakPower_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA PeakPower plot saved: {plot_filename}")

    def plot_voa_peakwave_per_tile(self):
        """
        Create VOA PeakWave plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows peak wavelength vs Set VOA current.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA PeakWave plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all PeakWave channels (0-7) for this specific channel setting
                            for peakwave_idx in range(8):
                                peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                                
                                # Check if column exists in the DataFrame
                                if peakwave_col in channel_data.columns.tolist():
                                    # Group by Set VOA current and take mean PeakWave
                                    grouped_data = channel_data.groupby('Set VOA(mA)')[peakwave_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakWave_{peakwave_idx}: {len(grouped_data)} points, wavelength range: {grouped_data[peakwave_col].min():.3f} to {grouped_data[peakwave_col].max():.3f} nm")
                                        
                                        # Use different colors for different PeakWave channels
                                        color_idx = peakwave_idx % len(colors)
                                        # Use different line styles for different temperatures
                                        linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                        
                                        ax.plot(grouped_data['Set VOA(mA)'], grouped_data[peakwave_col], 
                                               marker='o', linewidth=1.5, markersize=2, 
                                               color=colors[color_idx], linestyle=linestyle,
                                               label=f'Channel {peakwave_idx}' if temp_idx == 0 else "")
                                        plotted = True
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set VOA Current (mA)')
                    ax.set_ylabel('Peak Wavelength (nm)')
                    ax.set_xlim(0, 20)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper right', fontsize=8)
            
            plt.suptitle(f'VOA PeakWave vs Set VOA Current - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"VOA_PeakWave_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA PeakWave plot saved: {plot_filename}")

    def plot_voa_deltawave_per_tile(self):
        """
        Create VOA DeltaWave plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows the change in peak wavelength from the first 0mA VOA datapoint.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA DeltaWave plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    channel_slopes = {}  # Store individual slopes for each channel
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all PeakWave channels (0-7) for this specific channel setting
                            for peakwave_idx in range(8):
                                peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                                
                                # Check if column exists in the DataFrame
                                if peakwave_col in channel_data.columns.tolist():
                                    # Group by Set VOA current and take mean PeakWave
                                    grouped_data = channel_data.groupby('Set VOA(mA)')[peakwave_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        # Find the baseline (first 0mA datapoint)
                                        baseline_data = grouped_data[grouped_data['Set VOA(mA)'] == 0].copy()
                                        
                                        if len(baseline_data) > 0:
                                            baseline_wavelength = baseline_data[peakwave_col].values[0]
                                            
                                            # Calculate delta from baseline
                                            grouped_data = grouped_data.copy()
                                            grouped_data['delta_wavelength'] = grouped_data[peakwave_col] - baseline_wavelength
                                            
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakWave_{peakwave_idx}: {len(grouped_data)} points, baseline: {baseline_wavelength:.3f} nm, delta range: {grouped_data['delta_wavelength'].min():.3f} to {grouped_data['delta_wavelength'].max():.3f} nm")
                                            
                                            # Calculate individual slope for this channel (only for first temperature to avoid duplicates)
                                            if temp_idx == 0 and len(grouped_data) > 1:
                                                voa_currents = grouped_data['Set VOA(mA)'].values
                                                delta_wavelengths = grouped_data['delta_wavelength'].values
                                                
                                                if len(voa_currents) >= 2:
                                                    slope = np.polyfit(voa_currents, delta_wavelengths, 1)[0]
                                                    slope_pm = slope * 1000  # Convert nm/mA to pm/mA
                                                    channel_slopes[f'Ch{peakwave_idx}'] = slope_pm
                                                    print(f"        Individual slope for Ch{channel}, Bank {bank}, PeakWave_{peakwave_idx}: {slope_pm:.2f} pm/mA")
                                            
                                            # Use different colors for different PeakWave channels
                                            color_idx = peakwave_idx % len(colors)
                                            # Use different line styles for different temperatures
                                            linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                            
                                            ax.plot(grouped_data['Set VOA(mA)'], grouped_data['delta_wavelength'], 
                                                   marker='o', linewidth=1.5, markersize=2, 
                                                   color=colors[color_idx], linestyle=linestyle,
                                                   label=f'Channel {peakwave_idx}' if temp_idx == 0 else "")
                                            plotted = True
                                        else:
                                            print(f"      Ch{channel}, Bank {bank}, Temp {temp}, PeakWave_{peakwave_idx}: No 0mA baseline found")
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set VOA Current (mA)')
                    ax.set_ylabel('ΔWavelength (nm)')
                    ax.set_xlim(0, 20)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Add zero reference line
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper left', fontsize=8)
                        
                        # Add individual channel slopes annotation
                        if channel_slopes:
                            slope_text = "Slopes (pm/mA):\n"
                            for ch_name, slope in channel_slopes.items():
                                slope_text += f"{ch_name}: {slope:.2f}\n"
                            
                            ax.text(0.98, 0.98, slope_text.strip(), 
                                   transform=ax.transAxes, fontsize=8, 
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'VOA DeltaWave (Change from 0mA baseline) - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"VOA_DeltaWave_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA DeltaWave plot saved: {plot_filename}")

    def plot_voa_mpd_pic_per_tile(self):
        """
        Create VOA MPD_PIC plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows MPD_PIC values vs Set VOA current.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA MPD_PIC plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all MPD_PIC channels (0-7) for this specific channel setting
                            for mpd_idx in range(8):
                                mpd_col = f'MPD_PIC_{mpd_idx}(uA)'
                                
                                # Check if column exists in the DataFrame
                                if mpd_col in channel_data.columns.tolist():
                                    # Group by Set VOA current and take mean MPD_PIC
                                    grouped_data = channel_data.groupby('Set VOA(mA)')[mpd_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        print(f"      Ch{channel}, Bank {bank}, Temp {temp}, MPD_PIC_{mpd_idx}: {len(grouped_data)} points, MPD range: {grouped_data[mpd_col].min():.1f} to {grouped_data[mpd_col].max():.1f} uA")
                                        
                                        # Use different colors for different MPD_PIC channels
                                        color_idx = mpd_idx % len(colors)
                                        # Use different line styles for different temperatures
                                        linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                        
                                        ax.plot(grouped_data['Set VOA(mA)'], grouped_data[mpd_col], 
                                               marker='o', linewidth=1.5, markersize=2, 
                                               color=colors[color_idx], linestyle=linestyle,
                                               label=f'Channel {mpd_idx}' if temp_idx == 0 else "")
                                        plotted = True
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set VOA Current (mA)')
                    ax.set_ylabel('MPD_PIC (uA)')
                    ax.set_xlim(0, 20)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper right', fontsize=8)
            
            plt.suptitle(f'VOA MPD_PIC vs Set VOA Current - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"VOA_MPD_PIC_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA MPD_PIC plot saved: {plot_filename}")

    def plot_voa_mpd_mux_per_tile(self):
        """
        Create VOA MPD_MUX plots for each tile with 16 subplots (8 rows × 2 columns).
        Each row represents a channel (Ch0-Ch7), columns represent Bank 0 and Bank 1.
        Shows MPD_MUX values vs Set VOA current.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        # Get unique tiles
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA MPD_MUX plot for tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            print(f"  Available PIC temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Create 16 subplots: 8 rows (channels) × 2 columns (banks)
            fig, axs = plt.subplots(8, 2, figsize=(16, 24))
            
            # Process each channel and bank combination
            for channel in range(8):  # Channels 0-7 (rows)
                for bank in [0, 1]:  # Banks 0-1 (columns)
                    ax = axs[channel, bank]
                    tile_bank_data = tile_data[tile_data['Bank'] == bank].copy()
                    plotted = False
                    
                    print(f"    Channel {channel}, Bank {bank}: {len(tile_bank_data)} bank rows")
                    
                    # Plot data for each temperature
                    for temp_idx, temp in enumerate(temps):
                        temp_data = tile_bank_data[tile_bank_data['Set PIC Temp(C)'] == temp].copy()
                        channel_data = temp_data[temp_data['Channel'] == channel].copy()
                        
                        if len(channel_data) > 0:
                            # Plot all MPD_MUX channels (0-7) for this specific channel setting
                            for mpd_idx in range(8):
                                mpd_col = f'MPD_MUX_{mpd_idx}(uA)'
                                
                                # Check if column exists in the DataFrame
                                if mpd_col in channel_data.columns.tolist():
                                    # Group by Set VOA current and take mean MPD_MUX
                                    grouped_data = channel_data.groupby('Set VOA(mA)')[mpd_col].mean().reset_index()
                                    
                                    if len(grouped_data) > 0:
                                        print(f"      Ch{channel}, Bank {bank}, Temp {temp}, MPD_MUX_{mpd_idx}: {len(grouped_data)} points, MPD range: {grouped_data[mpd_col].min():.1f} to {grouped_data[mpd_col].max():.1f} uA")
                                        
                                        # Use different colors for different MPD_MUX channels
                                        color_idx = mpd_idx % len(colors)
                                        # Use different line styles for different temperatures
                                        linestyle = '-' if temp_idx == 0 else '--' if temp_idx == 1 else ':'
                                        
                                        ax.plot(grouped_data['Set VOA(mA)'], grouped_data[mpd_col], 
                                               marker='o', linewidth=1.5, markersize=2, 
                                               color=colors[color_idx], linestyle=linestyle,
                                               label=f'Channel {mpd_idx}' if temp_idx == 0 else "")
                                        plotted = True
                    
                    # Configure subplot
                    if not plotted:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_title(f'Ch{channel} - Bank {bank}')
                    ax.set_xlabel('Set VOA Current (mA)')
                    ax.set_ylabel('MPD_MUX (uA)')
                    ax.set_xlim(0, 20)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add individual legend for each subplot
                    if plotted:
                        ax.legend(loc='upper right', fontsize=8)
            
            plt.suptitle(f'VOA MPD_MUX vs Set VOA Current - Tile {tile_sn}', fontsize=20)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for suptitle
            
            # Save the plot
            plot_filename = f"VOA_MPD_MUX_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA MPD_MUX plot saved: {plot_filename}")

    def run_voa_analysis(self):
        """
        Run complete VOA analysis: load VOA files, combine data, and create all VOA plots.
        """
        print("Starting TP2-1 VOA Data Analysis...")
        print("=" * 50)
        
        # Load all VOA files
        self.load_voa_files()
        
        # Combine VOA data
        self.combine_voa_data()
        
        # Create all VOA plots
        self.plot_voa_peakpower_per_tile()
        self.plot_voa_peakwave_per_tile()
        self.plot_voa_deltawave_per_tile()
        self.plot_voa_mpd_pic_per_tile()
        self.plot_voa_mpd_mux_per_tile()
        
        # Create combined VOA thermal cross-talk summary plot
        self.plot_combined_voa_thermal_crosstalk_summary()
        
        print("\n" + "=" * 50)
        print("VOA Analysis completed successfully!")
        print(f"VOA plots saved in: {self.output_dir}")
        print(f"Combined VOA thermal cross-talk plot saved in: {Path(__file__).parent / 'plots'}")
        
        # List generated files
        unique_tiles = self.combined_voa_data['Tile_SN'].dropna().unique() if self.combined_voa_data is not None else []
        print("\n📋 Generated VOA files:")
        for tile_sn in unique_tiles:
            print(f"   • VOA_PeakPower_{tile_sn}.png")
            print(f"   • VOA_PeakWave_{tile_sn}.png")
            print(f"   • VOA_DeltaWave_{tile_sn}.png")
            print(f"   • VOA_MPD_PIC_{tile_sn}.png")
            print(f"   • VOA_MPD_MUX_{tile_sn}.png")
        print(f"   • tp2p1_voa_thermal_crosstalkvs_tile_combined.png")
        print("=" * 50)

    def plot_combined_voa_thermal_crosstalk_summary(self):
        """
        Create a combined VOA thermal cross-talk summary plot for all tiles.
        Layout: 5 rows × 2 columns (Principal, 1st, 2nd, 3rd, 4th order vs Bank 0/1)
        Shows box plots and scatter plots of wavelength slopes vs tile SN from VOA current variation.
        """
        if self.combined_voa_data is None:
            print("No combined VOA data available! Run combine_voa_data() first.")
            return
        
        print("Creating combined VOA thermal cross-talk summary plot...")
        
        # Get unique tiles and sort them
        unique_tiles = sorted(self.combined_voa_data['Tile_SN'].dropna().unique())
        
        # Collect all cross-talk data
        crosstalk_data = {
            'Principal': {'Bank0': [], 'Bank1': []},
            '1st Order': {'Bank0': [], 'Bank1': []},
            '2nd Order': {'Bank0': [], 'Bank1': []},
            '3rd Order': {'Bank0': [], 'Bank1': []},
            '4th Order': {'Bank0': [], 'Bank1': []}
        }
        
        # Process each tile
        for tile_sn in unique_tiles:
            print(f"  Processing tile {tile_sn}...")
            
            # Get data for this tile
            tile_data = self.combined_voa_data[self.combined_voa_data['Tile_SN'] == tile_sn].copy()
            
            # Get available temperatures for this tile (use first temperature for analysis)
            pic_temp_series = tile_data['Set PIC Temp(C)'].dropna()
            available_temps = sorted(pic_temp_series.unique())
            temp = available_temps[0]  # Use first temperature
            
            # Calculate slopes for all channel/bank/peakwave combinations
            slopes_data = {}
            
            for channel in range(8):  # Channels 0-7
                for bank in [0, 1]:  # Banks 0-1
                    key = f"Ch{channel}_Bank{bank}"
                    slopes_data[key] = {}
                    
                    # Get data for this channel/bank/temperature combination
                    channel_data = tile_data[
                        (tile_data['Channel'] == channel) & 
                        (tile_data['Bank'] == bank) & 
                        (tile_data['Set PIC Temp(C)'] == temp)
                    ].copy()
                    
                    if len(channel_data) > 0:
                        # Calculate slopes for each PeakWave sensor
                        for peakwave_idx in range(8):
                            peakwave_col = f'PeakWave_{peakwave_idx}(nm)'
                            
                            if peakwave_col in channel_data.columns.tolist():
                                # Group by Set VOA current and take mean PeakWave
                                grouped_data = channel_data.groupby('Set VOA(mA)')[peakwave_col].mean().reset_index()
                                
                                if len(grouped_data) > 1:
                                    # Find the baseline (first 0mA datapoint)
                                    baseline_data = grouped_data[grouped_data['Set VOA(mA)'] == 0].copy()
                                    
                                    if len(baseline_data) > 0:
                                        baseline_wavelength = baseline_data[peakwave_col].values[0]
                                        
                                        # Calculate delta from baseline
                                        grouped_data = grouped_data.copy()
                                        grouped_data['delta_wavelength'] = grouped_data[peakwave_col] - baseline_wavelength
                                        
                                        # Calculate slope
                                        voa_currents = grouped_data['Set VOA(mA)'].values
                                        delta_wavelengths = grouped_data['delta_wavelength'].values
                                        
                                        if len(voa_currents) >= 2:
                                            slope = np.polyfit(voa_currents, delta_wavelengths, 1)[0]
                                            slope_pm = slope * 1000  # Convert nm/mA to pm/mA
                                            slopes_data[key][f'PeakWave_{peakwave_idx}'] = slope_pm
            
            # Categorize slopes by cross-talk order for each channel/bank
            for channel in range(8):
                for bank in [0, 1]:
                    key = f"Ch{channel}_Bank{bank}"
                    bank_key = f"Bank{bank}"
                    
                    if key in slopes_data and slopes_data[key]:
                        # Principal slope
                        principal_slope = slopes_data[key].get(f'PeakWave_{channel}', np.nan)
                        if not np.isnan(principal_slope):
                            crosstalk_data['Principal'][bank_key].append({
                                'tile': tile_sn,
                                'channel': channel,
                                'slope': principal_slope
                            })
                        
                        # 1st order cross-talk: adjacent channels (±1)
                        for offset in [-1, 1]:
                            adjacent_ch = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{adjacent_ch}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['1st Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 2nd order cross-talk: channels ±2 positions away
                        for offset in [-2, 2]:
                            ch_2nd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_2nd}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['2nd Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 3rd order cross-talk: channels ±3 positions away
                        for offset in [-3, 3]:
                            ch_3rd = (channel + offset) % 8
                            slope = slopes_data[key].get(f'PeakWave_{ch_3rd}', np.nan)
                            if not np.isnan(slope):
                                crosstalk_data['3rd Order'][bank_key].append({
                                    'tile': tile_sn,
                                    'channel': channel,
                                    'slope': slope
                                })
                        
                        # 4th order cross-talk: channel ±4 positions away (opposite)
                        fourth_ch = (channel + 4) % 8
                        slope = slopes_data[key].get(f'PeakWave_{fourth_ch}', np.nan)
                        if not np.isnan(slope):
                            crosstalk_data['4th Order'][bank_key].append({
                                'tile': tile_sn,
                                'channel': channel,
                                'slope': slope
                            })
        
        # Create the combined plot: 5 rows × 2 columns
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        
        orders = ['Principal', '1st Order', '2nd Order', '3rd Order', '4th Order']
        colors = ['#1f77b4', '#ff9999', '#ff6666', '#ff3333', '#ff0000']
        
        for row_idx, (order, color) in enumerate(zip(orders, colors)):
            for col_idx, bank in enumerate(['Bank0', 'Bank1']):
                ax = axes[row_idx, col_idx]
                
                # Get data for this order/bank combination
                data = crosstalk_data[order][bank]
                
                if data:
                    # Convert to DataFrame for easier manipulation
                    df = pd.DataFrame(data)
                    
                    # Create box plot data
                    box_data = []
                    tile_positions = []
                    tile_labels = []
                    
                    for i, tile in enumerate(unique_tiles):
                        tile_data = df[df['tile'] == tile]['slope'].values
                        if len(tile_data) > 0:
                            box_data.append(tile_data)
                            tile_positions.append(i)
                            tile_labels.append(tile)
                    
                    if box_data:
                        # Create box plot
                        bp = ax.boxplot(box_data, positions=tile_positions, widths=0.6, 
                                       patch_artist=True, showfliers=False)
                        
                        # Style box plots
                        for patch in bp['boxes']:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        # Add scatter plot overlay
                        for i, (tile, pos) in enumerate(zip(tile_labels, tile_positions)):
                            tile_slopes = df[df['tile'] == tile]['slope'].values
                            if len(tile_slopes) > 0:
                                # Add jitter for better visibility
                                jitter = np.random.normal(0, 0.1, len(tile_slopes))
                                ax.scatter(pos + jitter, tile_slopes, 
                                          alpha=0.6, s=20, color=color, edgecolors='black', linewidth=0.5)
                                
                                # Add average annotation
                                avg_slope = np.mean(tile_slopes)
                                ax.annotate(f'{avg_slope:.3f}', 
                                           xy=(pos, avg_slope), 
                                           xytext=(pos, avg_slope + 0.1),
                                           ha='center', va='bottom', fontsize=8,
                                           rotation=90, color='red')
                    
                    # Set labels and title
                    ax.set_title(f'{order} - Bank {col_idx}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Tile SN', fontsize=10)
                    ax.set_ylabel('VOA Thermal Slope (pm/mA)', fontsize=10)
                    
                    # Set x-axis
                    ax.set_xticks(range(len(unique_tiles)))
                    ax.set_xticklabels(unique_tiles, rotation=45, ha='right', fontsize=8)
                    
                    # Set y-axis range (use smaller ranges for VOA thermal effects)
                    if row_idx == 0:  # Principal - typically smaller than drive current
                        ax.set_ylim(-2, 2)
                    else:  # All cross-talk orders
                        ax.set_ylim(-0.5, 0.5)
                    
                    # Grid
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{order} - Bank {col_idx}', fontsize=12, fontweight='bold')
        
        plt.suptitle('VOA Thermal Cross-talk vs Tile - Combined Summary', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        # Save the plot to plots folder (not plots/TP2-1)
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = "tp2p1_voa_thermal_crosstalkvs_tile_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Combined VOA thermal cross-talk summary plot saved: {plot_filename}")

if __name__ == "__main__":
    # Create analyzer instance and run analysis
    analyzer = TP2P1CombinerAnalyzers()
    analyzer.run_analysis()