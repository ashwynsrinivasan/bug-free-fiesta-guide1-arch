#!/usr/bin/env python3
"""
TP1-1 Combined Test and Top Data Analysis Script
================================================

This script analyzes both Test and Top data from TP1-1 test point.
It produces all plots from the Test workflow, and annotates the wavelength_vs_tile_combined plot
with T_op values from the Top data for each tile and temperature.
All measurements were performed at 150mA laser current.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import glob
import re
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import io
import base64
import json
from pandas.api.types import is_datetime64_any_dtype

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TP1P1CombinedAnalyzer:
    def __init__(self, test_data_path=None, top_data_path=None):
        script_dir = Path(__file__).parent
        self.test_data_path = Path(test_data_path) if test_data_path else script_dir / "../TP1-1"
        self.top_data_path = Path(top_data_path) if top_data_path else script_dir / "../TP1-1"
        self.output_dir = script_dir / "plots"
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.test_files = []
        self.top_files = []
        self.test_data = None
        self.top_data = None
        self.tile_metadata = {}  # Store metadata for each tile

    def extract_serial_number(self, filename):
        match = re.search(r'-Y(\d+)-TP1-1 (Test|Top)\.csv$', filename)
        if match:
            return f"Y{match.group(1)}"
        return None

    def load_test_files(self):
        self.test_files = sorted(glob.glob(str(self.test_data_path / "* Test.csv")))
        print(f"Found {len(self.test_files)} Test CSV files")
        return self.test_files

    def load_top_files(self):
        self.top_files = sorted(glob.glob(str(self.top_data_path / "* Top.csv")))
        print(f"Found {len(self.top_files)} Top CSV files")
        return self.top_files

    def load_test_data(self):
        dfs = []
        for file_path in self.test_files:
            try:
                df = pd.read_csv(file_path)
                filename = Path(file_path).name
                sn_from_filename = self.extract_serial_number(filename)
                
                # Store metadata for this tile
                if sn_from_filename:
                    self.tile_metadata[sn_from_filename] = {
                        'filename': filename,
                        'batch': df['Batch'].iloc[0] if 'Batch' in df.columns else None,
                        'gelpak_number': df['Gelpaknumer'].iloc[0] if 'Gelpaknumer' in df.columns else None,
                        'gelpak_x': df['Gelpakx'].iloc[0] if 'Gelpakx' in df.columns else None,
                        'gelpak_y': df['Gelpaky'].iloc[0] if 'Gelpaky' in df.columns else None,
                        'bin': df['Bin'].iloc[0] if 'Bin' in df.columns else None,
                        'mmid': df['Mmid'].iloc[0] if 'Mmid' in df.columns else None,
                        'awg_id': df['AWG ID'].iloc[0] if 'AWG ID' in df.columns else None
                    }
                
                if 'Tile_SN' in df.columns:
                    df['filename'] = filename
                    dfs.append(df)
                elif 'AWG ID' in df.columns:
                    unified_df = pd.DataFrame()
                    unified_df['Tile_SN'] = [sn_from_filename] * len(df)
                    unified_df['Bank'] = df['Bank']
                    unified_df['Channel'] = df['Channel']
                    unified_df['Set Temp(C)'] = df['Set Temp(C)']
                    unified_df['Set Laser(mA)'] = df['Set Laser(mA)']
                    unified_df['T_PIC(C)'] = df['T_PIC(C)']
                    unified_df['Power(mW)'] = df['Power(mW)']
                    unified_df['MPD_PIC(uA)'] = df['MPD_PIC(uA)']
                    unified_df['PeakWave(nm)'] = df['PeakWave(nm)']
                    unified_df['PeakPower(dBm)'] = df['PeakPower(dBm)']
                    unified_df['Time'] = df['Time']
                    unified_df['filename'] = filename
                    dfs.append(unified_df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        if dfs:
            self.test_data = pd.concat(dfs, ignore_index=True)
            self.test_data['Time'] = pd.to_datetime(self.test_data['Time'], format='mixed', errors='coerce')
            self.test_data = self.test_data.dropna(subset=['Time'])
            self.test_data = self.test_data.sort_values('Time', axis=0)
            print(f"Combined test data shape: {self.test_data.shape}")
            print(f"Captured metadata for {len(self.tile_metadata)} tiles")
        else:
            print("No test data loaded successfully")

    def load_top_data(self):
        dfs = []
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for file_path in self.top_files:
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, header=None)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                print(f"Could not read {file_path} with any encoding")
                continue
            filename = Path(file_path).name
            ld_header = df.iloc[0, 1:].tolist()
            long_rows = []
            for i in range(1, len(df)):
                label = str(df.iloc[i, 0]).strip()
                if not label or label == 'nan':
                    continue
                for ld_idx, ld in enumerate(ld_header):
                    value = df.iloc[i, ld_idx + 1]
                    if pd.isna(value):
                        continue
                    try:
                        numeric_value = float(str(value).replace(',', ''))
                        long_rows.append({
                            'filename': filename,
                            'label': label,
                            'LD': str(ld),
                            'value': numeric_value
                        })
                    except (ValueError, TypeError):
                        continue
            if long_rows:
                dfs.append(pd.DataFrame(long_rows))
        if dfs:
            self.top_data = pd.concat(dfs, ignore_index=True)
            print(f"Combined top data shape: {self.top_data.shape}")
        else:
            print("No top data loaded successfully")

    def get_top_annotations(self, tile_sn, temp_label, annotation_type='T_op'):
        # Find the Top file for this tile
        if self.top_data is None:
            return None
        # Find the filename for this tile
        match = self.top_data[self.top_data['filename'].str.contains(tile_sn)]
        if match.empty:
            return None
        # Get the filename
        filename = match['filename'].iloc[0]  # type: ignore
        # For this file, get the requested annotation type
        top_data = self.top_data[(self.top_data['filename'] == filename) & (self.top_data['label'] == annotation_type)]
        # This is per file, not per temperature, so we annotate the same value for all temps
        top_val = top_data['value'].iloc[0] if not top_data.empty else None  # type: ignore
        return top_val

    def plot_wavelength_vs_tile_by_temperature(self):
        if self.test_data is None or self.top_data is None:
            print("Test or Top data not loaded!")
            return
        temps = [36, 43, 50]
        temp_labels = ['36C', '43C', '50C']
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        fig, axs = plt.subplots(3, 2, figsize=(24, 18), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
        for temp_idx, (temp, temp_label) in enumerate(zip(temps, temp_labels)):
            for bank in [0, 1]:
                scatter_data = []
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    if len(tile_bank_data) > 0:
                        for channel in range(8):
                            channel_data = tile_bank_data[tile_bank_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                scatter_data.append({
                                    'x': sorted_tiles.index(tile),
                                    'y': channel_data['PeakWave(nm)'].tolist(),
                                    'color': colors[channel],
                                    'channel': channel
                                })
                for scatter in scatter_data:
                    axs[temp_idx, bank].scatter(
                        [scatter['x']] * len(scatter['y']),
                        scatter['y'],
                        c=[scatter['color']],
                        s=20,
                        alpha=0.6,
                        label=f'Channel {scatter["channel"]}' if scatter['x'] == 0 else ""
                    )
                # Add average wavelength annotations
                for i, tile in enumerate(sorted_tiles):
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    if len(tile_bank_data) > 0:
                        avg_wavelength = tile_bank_data['PeakWave(nm)'].mean()
                        
                        # Create annotation text with only average wavelength
                        annotation_text = f'λ_ave={avg_wavelength:.2f}'
                        
                        axs[temp_idx, bank].text(
                            i, 1320,  # Position at 1320 nm on y-axis
                            annotation_text,
                            ha='center',
                            va='bottom',
                            fontsize=6,
                            fontweight='bold',
                            color='red',
                            rotation=90
                        )
                axs[temp_idx, bank].set_title(f'{temp}°C - Bank {bank}', fontsize=14)
                axs[temp_idx, bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
                axs[temp_idx, bank].set_xticks(range(len(sorted_tiles)))
                axs[temp_idx, bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
                axs[temp_idx, bank].set_ylabel('Wavelength (nm)', fontsize=12)
                axs[temp_idx, bank].set_ylim(1295, 1325)
                axs[temp_idx, bank].grid(True, linestyle='--', alpha=0.3)
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                markersize=8, label=f'Channel {i}') for i in range(8)]
                axs[temp_idx, bank].legend(handles=handles, loc=4, fontsize=8)
        fig.suptitle('Wavelength Distribution vs Tile SN by Temperature (with T_op Annotations) at 150mA', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tp1p1_wavelength_vs_tile_combined.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ Wavelength vs Tile plot with T_op annotations saved as tp1p1_wavelength_vs_tile_combined.png")

    def plot_power_vs_tile_by_temperature(self):
        """Create one combined figure with power distribution for each temperature, with horizontal subplots."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by their earliest measurement date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        
        # Sort tiles by date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create one large figure with 3 rows (temperatures) and 2 columns (banks)
        fig, axs = plt.subplots(3, 2, figsize=(24, 18), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for temp_idx, temp in enumerate(temps):
            for bank in [0, 1]:
                # Prepare data for box plot
                box_data = []
                scatter_data = []
                avg_powers = []  # Store average powers for annotation
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        box_data.append(tile_bank_data['Power(mW)'].tolist())
                        avg_power = tile_bank_data['Power(mW)'].mean()
                        avg_powers.append(avg_power)
                        
                        # Add scatter points color-coded by channel
                        for channel in range(8):
                            channel_data = tile_bank_data[tile_bank_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                scatter_data.append({
                                    'x': len(box_data) - 1,  # Box plot position
                                    'y': channel_data['Power(mW)'].tolist(),
                                    'color': colors[channel],
                                    'channel': channel
                                })
                    else:
                        avg_powers.append(np.nan)
                
                # Create box plot
                bp = axs[temp_idx, bank].boxplot(box_data, positions=range(len(box_data)), patch_artist=True)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                # Add scatter points color-coded by channel
                for scatter in scatter_data:
                    axs[temp_idx, bank].scatter(
                        [scatter['x']] * len(scatter['y']),
                        scatter['y'],
                        c=[scatter['color']],
                        s=20,
                        alpha=0.6,
                        label=f'Channel {scatter["channel"]}' if scatter['x'] == 0 else ""
                    )
                
                # Add average power annotations on top of tile SNs
                for i, avg_power in enumerate(avg_powers):
                    if not np.isnan(avg_power):
                        # Calculate total power for this tile/bank combination
                        tile_bank_mask = (
                            (self.test_data['Tile_SN'] == sorted_tiles[i]) &
                            (self.test_data['Set Temp(C)'] == temp) &
                            (self.test_data['Bank'] == bank)
                        )
                        tile_bank_data = self.test_data[tile_bank_mask]
                        total_power = tile_bank_data['Power(mW)'].sum()
                        
                        # Create annotation text
                        annotation_text = f'P_ave={avg_power:.1f} P_total={total_power:.1f}'
                        
                        axs[temp_idx, bank].text(
                            i, 40,  # Position at 40 mW on y-axis
                            annotation_text,
                            ha='center',
                            va='bottom',
                            fontsize=6,
                            fontweight='bold',
                            color='red',
                            rotation=90
                        )
                
                axs[temp_idx, bank].set_title(f'{temp}°C - Bank {bank}', fontsize=14)
                axs[temp_idx, bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
                axs[temp_idx, bank].set_xticks(range(len(sorted_tiles)))
                axs[temp_idx, bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
                axs[temp_idx, bank].set_ylabel('Power (mW)', fontsize=12)
                axs[temp_idx, bank].set_ylim(20, 50)  # Set y-axis range to 20-50 mW
                axs[temp_idx, bank].grid(True, linestyle='--', alpha=0.3)
                
                # Add legend for channels
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                markersize=8, label=f'Channel {i}') for i in range(8)]
                axs[temp_idx, bank].legend(handles=handles, loc=4, fontsize=8)
        
        fig.suptitle('Power Distribution vs Tile SN by Temperature (at 150mA)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tp1p1_power_vs_tile_combined.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ Power vs Tile plot saved as tp1p1_power_vs_tile_combined.png")

    def plot_pave_vs_temperature(self):
        """Create scatter plot of average power vs temperature for each tile."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by their earliest measurement date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        
        # Sort tiles by date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create figure with 2 subplots (banks)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for temperatures
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        avg_power = tile_bank_data['Power(mW)'].mean()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(avg_power)
                
                axs[bank].scatter(x_positions, y_values, 
                                c=colors[temp_idx], 
                                s=60, 
                                alpha=0.7, 
                                label=f'{temp}°C')
            
            axs[bank].set_title(f'Average Power vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            axs[bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
            axs[bank].set_ylabel('Average Power (mW)', fontsize=12)
            axs[bank].set_xticks(range(len(sorted_tiles)))
            axs[bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
            axs[bank].grid(True, linestyle='--', alpha=0.3)
            axs[bank].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tp1p1_pave_vs_temperature.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ Average Power vs Temperature plot saved as tp1p1_pave_vs_temperature.png")

    def plot_ptotal_vs_temperature(self):
        """Create scatter plot of total power vs temperature for each tile."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by their earliest measurement date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        
        # Sort tiles by date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create figure with 2 subplots (banks)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for temperatures
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        total_power = tile_bank_data['Power(mW)'].sum()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(total_power)
                
                axs[bank].scatter(x_positions, y_values, 
                                c=colors[temp_idx], 
                                s=60, 
                                alpha=0.7, 
                                label=f'{temp}°C')
            
            axs[bank].set_title(f'Total Power vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            axs[bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
            axs[bank].set_ylabel('Total Power (mW)', fontsize=12)
            axs[bank].set_xticks(range(len(sorted_tiles)))
            axs[bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
            axs[bank].grid(True, linestyle='--', alpha=0.3)
            axs[bank].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tp1p1_ptotal_vs_temperature.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ Total Power vs Temperature plot saved as tp1p1_ptotal_vs_temperature.png")

    def plot_wavelength_vs_temperature(self):
        """Create scatter plot of average wavelength vs temperature for each tile."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        temps = [36, 43, 50]
        temp_labels = ['36C', '43C', '50C']
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by their earliest measurement date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        
        # Sort tiles by date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create figure with 2 subplots (banks)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for temperatures
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        avg_wavelength = tile_bank_data['PeakWave(nm)'].mean()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(avg_wavelength)
                
                axs[bank].scatter(x_positions, y_values, 
                                c=colors[temp_idx], 
                                s=60, 
                                alpha=0.7, 
                                label=f'{temp}°C')
            
            # Add annotations for each tile
            for i, tile in enumerate(sorted_tiles):
                # Get average wavelength for this tile and bank
                mask = (
                    (self.test_data['Tile_SN'] == tile) &
                    (self.test_data['Bank'] == bank)
                )
                tile_bank_data = self.test_data[mask]
                if len(tile_bank_data) > 0:
                    avg_wavelength = tile_bank_data['PeakWave(nm)'].mean()
                    
                    # Calculate wavelength change across temperatures (36°C to 50°C)
                    wavelength_36 = None
                    wavelength_50 = None
                    
                    # Get wavelength at 36°C
                    mask_36 = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == 36) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_36_data = self.test_data[mask_36]
                    if len(tile_36_data) > 0:
                        wavelength_36 = tile_36_data['PeakWave(nm)'].mean()
                    
                    # Get wavelength at 50°C
                    mask_50 = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == 50) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_50_data = self.test_data[mask_50]
                    if len(tile_50_data) > 0:
                        wavelength_50 = tile_50_data['PeakWave(nm)'].mean()
                    
                    # Calculate wavelength change
                    wavelength_change = None
                    if wavelength_36 is not None and wavelength_50 is not None:
                        wavelength_change = wavelength_50 - wavelength_36
                    
                    # Calculate wavelength change per degree C
                    wavelength_slope = None
                    if wavelength_36 is not None and wavelength_50 is not None:
                        wavelength_slope = (wavelength_50 - wavelength_36) / (50 - 36)
                    
                    # Get T_op annotations from Top data
                    top_op = self.get_top_annotations(tile, '36C', 'T_op')  # Use any temp label
                    if bank == 0:
                        top_op_a = self.get_top_annotations(tile, '36C', 'T_op_A(1~8)')
                    else:
                        top_op_b = self.get_top_annotations(tile, '36C', 'T_op_B(9~16)')
                    
                    # Create annotation text
                    annotation_text = f'λ_ave={avg_wavelength:.2f}'
                    if top_op is not None:
                        annotation_text += f' T_op={top_op:.1f}°C'
                    if bank == 0 and top_op_a is not None:
                        annotation_text += f' T_op_A={top_op_a:.1f}°C'
                    elif bank == 1 and top_op_b is not None:
                        annotation_text += f' T_op_B={top_op_b:.1f}°C'
                    
                    axs[bank].text(
                        i, 1315,  # Position at 1315 nm on y-axis
                        annotation_text,
                        ha='center',
                        va='bottom',  # Start at 1315nm and extend upward
                        fontsize=6,
                        fontweight='bold',
                        color='red',
                        rotation=90
                    )
                    
                    # Add annotation for wavelength slope starting at 1300nm
                    if wavelength_slope is not None:
                        slope_annotation_text = f'Δλ/ΔT={wavelength_slope:.3f} nm/°C'
                        axs[bank].text(
                            i, 1300,  # Position at 1300 nm on y-axis
                            slope_annotation_text,
                            ha='center',
                            va='top',  # Start at 1300nm and extend downward
                            fontsize=6,
                            fontweight='bold',
                            color='blue',
                            rotation=90
                        )
            
            axs[bank].set_title(f'Average Wavelength vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            axs[bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
            axs[bank].set_ylabel('Average Wavelength (nm)', fontsize=12)
            axs[bank].set_xticks(range(len(sorted_tiles)))
            axs[bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
            axs[bank].set_ylim(1295, 1325)  # Set y-axis range for wavelength
            axs[bank].grid(True, linestyle='--', alpha=0.3)
            axs[bank].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tp1p1_wavelength_vs_temperature.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("✅ Average Wavelength vs Temperature plot saved as tp1p1_wavelength_vs_temperature.png")

    def create_dashboard(self):
        """Create separate Plotly figures for each plot and save as individual HTML files."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        # Create separate Plotly figures for each plot
        self.create_power_vs_tile_plotly()
        self.create_wavelength_vs_tile_plotly()
        self.create_pave_vs_temp_plotly()
        self.create_ptotal_vs_temp_plotly()
        self.create_wavelength_vs_temp_plotly()
        
        print("✅ All Plotly figures saved as individual HTML files")

    def create_power_vs_tile_plotly(self):
        """Create Plotly version of power vs tile plot."""
        if self.test_data is None:
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[f'{temp}°C - Bank {bank}' for temp in temps for bank in [0, 1]],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for temp_idx, temp in enumerate(temps):
            for bank in [0, 1]:
                row = temp_idx + 1
                col = bank + 1
                
                # Add box plots and scatter points
                for tile_idx, tile in enumerate(sorted_tiles):
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        # Add box plot
                        fig.add_trace(
                            go.Box(
                                y=tile_bank_data['Power(mW)'],
                                name=tile,
                                boxpoints=False,
                                marker_color='lightblue',
                                opacity=0.7
                            ),
                            row=row, col=col
                        )
                        
                        # Add scatter points by channel
                        for channel in range(8):
                            channel_data = tile_bank_data[tile_bank_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=[tile_idx] * len(channel_data),
                                        y=channel_data['Power(mW)'],
                                        mode='markers',
                                        marker=dict(color=colors[channel], size=6, opacity=0.6),
                                        name=f'Channel {channel}',
                                        showlegend=(temp_idx == 0 and bank == 0)
                                    ),
                                    row=row, col=col
                                )
        
        fig.update_layout(
            title='Power Distribution vs Tile SN by Temperature (at 150mA)',
            height=1200,
            width=1600,
            showlegend=True
        )
        
        fig.write_html(self.output_dir / "tp1p1_power_vs_tile_plotly.html")
        print("✅ Power vs Tile Plotly figure saved")

    def create_wavelength_vs_tile_plotly(self):
        """Create Plotly version of wavelength vs tile plot."""
        if self.test_data is None:
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[f'{temp}°C - Bank {bank}' for temp in temps for bank in [0, 1]],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for temp_idx, temp in enumerate(temps):
            for bank in [0, 1]:
                row = temp_idx + 1
                col = bank + 1
                
                for tile_idx, tile in enumerate(sorted_tiles):
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        for channel in range(8):
                            channel_data = tile_bank_data[tile_bank_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=[tile_idx] * len(channel_data),
                                        y=channel_data['PeakWave(nm)'],
                                        mode='markers',
                                        marker=dict(color=colors[channel], size=6, opacity=0.6),
                                        name=f'Channel {channel}',
                                        showlegend=(temp_idx == 0 and bank == 0)
                                    ),
                                    row=row, col=col
                                )
        
        fig.update_layout(
            title='Wavelength Distribution vs Tile SN by Temperature (at 150mA)',
            height=1200,
            width=1600,
            showlegend=True
        )
        
        fig.update_yaxes(range=[1295, 1325])
        fig.write_html(self.output_dir / "tp1p1_wavelength_vs_tile_plotly.html")
        print("✅ Wavelength vs Tile Plotly figure saved")

    def create_pave_vs_temp_plotly(self):
        """Create Plotly version of average power vs temperature plot."""
        if self.test_data is None:
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Bank 0', 'Bank 1'])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        avg_power = tile_bank_data['Power(mW)'].mean()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(avg_power)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_positions,
                        y=y_values,
                        mode='markers',
                        marker=dict(color=colors[temp_idx], size=8, opacity=0.7),
                        name=f'{temp}°C',
                        showlegend=(bank == 0)
                    ),
                    row=1, col=bank + 1
                )
        
        fig.update_layout(
            title='Average Power vs Tile SN by Temperature (at 150mA)',
            height=600,
            width=1200
        )
        
        fig.write_html(self.output_dir / "tp1p1_pave_vs_temp_plotly.html")
        print("✅ Average Power vs Temperature Plotly figure saved")

    def create_ptotal_vs_temp_plotly(self):
        """Create Plotly version of total power vs temperature plot."""
        if self.test_data is None:
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Bank 0', 'Bank 1'])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        total_power = tile_bank_data['Power(mW)'].sum()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(total_power)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_positions,
                        y=y_values,
                        mode='markers',
                        marker=dict(color=colors[temp_idx], size=8, opacity=0.7),
                        name=f'{temp}°C',
                        showlegend=(bank == 0)
                    ),
                    row=1, col=bank + 1
                )
        
        fig.update_layout(
            title='Total Power vs Tile SN by Temperature (at 150mA)',
            height=600,
            width=1200
        )
        
        fig.write_html(self.output_dir / "tp1p1_ptotal_vs_temp_plotly.html")
        print("✅ Total Power vs Temperature Plotly figure saved")

    def create_wavelength_vs_temp_plotly(self):
        """Create Plotly version of wavelength vs temperature plot."""
        if self.test_data is None:
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Bank 0', 'Bank 1'])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for bank in [0, 1]:
            for temp_idx, temp in enumerate(temps):
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        avg_wavelength = tile_bank_data['PeakWave(nm)'].mean()
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(avg_wavelength)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_positions,
                        y=y_values,
                        mode='markers',
                        marker=dict(color=colors[temp_idx], size=8, opacity=0.7),
                        name=f'{temp}°C',
                        showlegend=(bank == 0)
                    ),
                    row=1, col=bank + 1
                )
        
        fig.update_layout(
            title='Average Wavelength vs Tile SN by Temperature (at 150mA)',
            height=600,
            width=1200
        )
        
        fig.update_yaxes(range=[1295, 1325])
        fig.write_html(self.output_dir / "tp1p1_wavelength_vs_temp_plotly.html")
        print("✅ Wavelength vs Temperature Plotly figure saved")

    def plot_mpd_vs_tile_combined(self):
        """Create a combined MPD plot showing MPD_PIC vs Tile SN with separate subplots for each temperature, similar to power plot."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        temps = [36, 43, 50]
        tile_sn_series = pd.Series(self.test_data['Tile_SN'])
        unique_tiles = tile_sn_series.dropna().unique()
        
        # Order tiles by their earliest measurement date
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        
        # Sort tiles by date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        
        # Create one large figure with 3 rows (temperatures) and 2 columns (banks)
        fig, axs = plt.subplots(3, 2, figsize=(24, 18), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for temp_idx, temp in enumerate(temps):
            for bank in [0, 1]:
                # Prepare data for box plot
                box_data = []
                scatter_data = []
                avg_mpds = []  # Store average MPDs for annotation
                
                for tile in sorted_tiles:
                    mask = (
                        (self.test_data['Tile_SN'] == tile) &
                        (self.test_data['Set Temp(C)'] == temp) &
                        (self.test_data['Bank'] == bank)
                    )
                    tile_bank_data = self.test_data[mask]
                    
                    if len(tile_bank_data) > 0:
                        box_data.append(tile_bank_data['MPD_PIC(uA)'].tolist())
                        avg_mpd = tile_bank_data['MPD_PIC(uA)'].mean()
                        avg_mpds.append(avg_mpd)
                        
                        # Add scatter points color-coded by channel
                        for channel in range(8):
                            channel_data = tile_bank_data[tile_bank_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                scatter_data.append({
                                    'x': len(box_data) - 1,  # Box plot position
                                    'y': channel_data['MPD_PIC(uA)'].tolist(),
                                    'color': colors[channel],
                                    'channel': channel
                                })
                    else:
                        avg_mpds.append(np.nan)
                
                # Create box plot
                bp = axs[temp_idx, bank].boxplot(box_data, positions=range(len(box_data)), patch_artist=True)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                # Add scatter points color-coded by channel
                for scatter in scatter_data:
                    axs[temp_idx, bank].scatter(
                        [scatter['x']] * len(scatter['y']),
                        scatter['y'],
                        c=[scatter['color']],
                        s=20,
                        alpha=0.6,
                        label=f'Channel {scatter["channel"]}' if scatter['x'] == 0 else ""
                    )
                
                # Add average MPD annotations on top of tile SNs
                for i, avg_mpd in enumerate(avg_mpds):
                    if not np.isnan(avg_mpd):
                        # Create annotation text
                        annotation_text = f'M_ave={avg_mpd:.0f}uA'
                        
                        axs[temp_idx, bank].text(
                            i, 900,  # Position at 900 uA on y-axis
                            annotation_text,
                            ha='center',
                            va='bottom',
                            fontsize=6,
                            fontweight='bold',
                            color='red',
                            rotation=90
                        )
                
                axs[temp_idx, bank].set_title(f'{temp}°C - Bank {bank}', fontsize=14)
                axs[temp_idx, bank].set_xlabel('Tile SN (ordered by date)', fontsize=12)
                axs[temp_idx, bank].set_xticks(range(len(sorted_tiles)))
                axs[temp_idx, bank].set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
                axs[temp_idx, bank].set_ylabel('MPD_PIC (uA)', fontsize=12)
                axs[temp_idx, bank].set_ylim(400, 1000)  # Updated y-axis range to 400-1000 uA
                axs[temp_idx, bank].grid(True, linestyle='--', alpha=0.3)
                
                # Add legend for channels
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                markersize=8, label=f'Channel {i}') for i in range(8)]
                axs[temp_idx, bank].legend(handles=handles, loc=4, fontsize=8)
        
        fig.suptitle('MPD_PIC Distribution vs Tile SN by Temperature (at 150mA)', fontsize=16)
        plt.tight_layout()
        plot_filename = "tp1p1_mpd_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_mpd_vs_tile_html()

    def create_mpd_vs_tile_html(self):
        """Create interactive HTML plot for MPD vs tile."""
        if self.test_data is None:
            print("Test data not loaded!")
            return
        
        print("Creating interactive HTML MPD vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - MPD_PIC vs Tile SN at 150mA', 'Bank 1 - MPD_PIC vs Tile SN at 150mA'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.test_data['Tile_SN'].dropna().unique()
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.test_data[self.test_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.test_data[self.test_data['Bank'] == bank]
            
            for ch in range(8):
                ch_mpd = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                    if len(tile_150ma) > 0:
                        ch_mpd.append(tile_150ma['MPD_PIC(uA)'].mean())
                        ch_x.append(i)
                
                if ch_x:
                    fig.add_trace(
                        go.Scatter(
                            x=[sorted_tiles[i] for i in ch_x],
                            y=ch_mpd,
                            mode='markers',
                            name=f'Bank {bank} - Channel {ch}',
                            marker=dict(color=channel_colors[ch], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>MPD_PIC:</b> %{y:.2f} uA<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<br>' +
                                        '<b>Laser Current:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='MPD_PIC vs Tile SN at 150mA Laser Current',
                x=0.5,
                font=dict(size=20, color='black')
            ),
            width=1600,
            height=600,
            showlegend=False,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Tile Serial Number", row=1, col=1)
        fig.update_xaxes(title_text="Tile Serial Number", row=1, col=2)
        fig.update_yaxes(title_text="MPD_PIC (uA)", row=1, col=1, range=[400, 1000])
        fig.update_yaxes(title_text="MPD_PIC (uA)", row=1, col=2, range=[400, 1000])
        
        # Save HTML file
        html_filename = "tp1p1_mpd_vs_tile_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML MPD vs tile plot saved: {html_filename}")
        
        return fig

    def export_to_xarray(self):
        """Export all combined data (test and top) as an xarray DataArray and save as a .nc file with TP1-1 attributes."""
        import xarray as xr
        from datetime import datetime
        
        # Prepare test data
        test_df = self.test_data.copy() if self.test_data is not None else pd.DataFrame()
        top_df = self.top_data.copy() if self.top_data is not None else pd.DataFrame()
        
        # Add a column to distinguish source
        if not test_df.empty:
            test_df['source'] = 'test'
        if not top_df.empty:
            top_df['source'] = 'top'
        
        # Unify columns for concatenation
        all_columns = set(test_df.columns).union(set(top_df.columns))
        test_df = test_df.reindex(columns=all_columns)
        top_df = top_df.reindex(columns=all_columns)
        combined_df = pd.concat([test_df, top_df], ignore_index=True)
        
        # Convert datetime columns to ISO strings
        for col in combined_df.columns:
            if is_datetime64_any_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].astype(str)

        # Convert to xarray Dataset (not DataArray) for mixed types
        data_xr = xr.Dataset.from_dataframe(combined_df)
        
        # Add attributes (only serializable types)
        data_xr.attrs["test_point"] = "TP1-1"
        data_xr.attrs["laser_current_mA"] = 150
        data_xr.attrs["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_xr.attrs["tile_count"] = len(self.tile_metadata)
        data_xr.attrs["test_data_points"] = len(self.test_data) if self.test_data is not None else 0
        data_xr.attrs["top_data_points"] = len(self.top_data) if self.top_data is not None else 0
        
        # Add metadata keys as a list of strings
        if self.tile_metadata:
            sample_meta = next(iter(self.tile_metadata.values()))
            data_xr.attrs["tile_metadata_keys"] = list(sample_meta.keys())
            # Add sample metadata as a JSON string to avoid serialization issues
            sample_meta_str = {}
            for key, value in sample_meta.items():
                if pd.isna(value):
                    sample_meta_str[key] = "nan"
                else:
                    sample_meta_str[key] = str(value)
            data_xr.attrs["tile_metadata_sample"] = json.dumps(sample_meta_str)
        else:
            data_xr.attrs["tile_metadata_keys"] = []
            data_xr.attrs["tile_metadata_sample"] = {}
        
        # Save as NetCDF
        nc_path = self.data_dir / "tp1p1_combined_data.nc"
        data_xr.to_netcdf(nc_path, engine='netcdf4')
        print(f"✅ Exported combined data to NetCDF: {nc_path}")

    def run_all(self):
        print("🔄 Loading TP-1 Test data files...")
        self.load_test_files()
        print("🔄 Loading TP-1 Top data files...")
        self.load_top_files()
        print("🔄 Combining TP-1 Test data...")
        self.load_test_data()
        print("🔄 Combining TP-1 Top data...")
        self.load_top_data()
        
        print("\n" + "="*60)
        print("TP1-1 POWER VS TILE BY TEMPERATURE")
        print("="*60)
        self.plot_power_vs_tile_by_temperature()
        
        print("\n" + "="*60)
        print("TP1-1 WAVELENGTH VS TILE BY TEMPERATURE (WITH T_op ANNOTATIONS)")
        print("="*60)
        self.plot_wavelength_vs_tile_by_temperature()
        
        print("\n" + "="*60)
        print("TP1-1 MPD VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-1 AVERAGE POWER VS TEMPERATURE")
        print("="*60)
        self.plot_pave_vs_temperature()
        
        print("\n" + "="*60)
        print("TP1-1 TOTAL POWER VS TEMPERATURE")
        print("="*60)
        self.plot_ptotal_vs_temperature()
        
        print("\n" + "="*60)
        print("TP1-1 AVERAGE WAVELENGTH VS TEMPERATURE")
        print("="*60)
        self.plot_wavelength_vs_temperature()
        
        print("\n" + "="*60)
        print("TP1-1 DASHBOARD")
        print("="*60)
        self.create_dashboard()
        
        # Export to xarray/netcdf
        print("\n" + "="*60)
        print("TP1-1 EXPORT TO NETCDF")
        print("="*60)
        self.export_to_xarray()
        
        print("\n" + "="*80)
        print("TP1-1 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"✅ PNG plots saved to: {self.output_dir.absolute()}")
        print("📋 Generated plots:")
        print(f"   • tp1p1_power_vs_tile_combined.png")
        print(f"   • tp1p1_wavelength_vs_tile_combined.png")
        print(f"   • tp1p1_pave_vs_temperature.png")
        print(f"   • tp1p1_ptotal_vs_temperature.png")
        print(f"   • tp1p1_wavelength_vs_temperature.png")
        print(f"   • tp1p1_power_vs_tile_plotly.html")
        print(f"   • tp1p1_wavelength_vs_tile_plotly.html")
        print(f"   • tp1p1_pave_vs_temp_plotly.html")
        print(f"   • tp1p1_ptotal_vs_temp_plotly.html")
        print(f"   • tp1p1_wavelength_vs_temp_plotly.html")
        print(f"   • tp1p1_mpd_vs_tile_combined.png")
        print(f"   • tp1p1_mpd_vs_tile_combined.html")
        print(f"   • tp1p1_combined_data.nc (in data folder)")
        print("\n📊 Metadata Summary:")
        print(f"   • Analyzed {len(self.tile_metadata)} tiles")
        print(f"   • All measurements performed at 150mA laser current")
        print(f"   • Test data points: {len(self.test_data) if self.test_data is not None else 0}")
        print(f"   • Top data points: {len(self.top_data) if self.top_data is not None else 0}")
        print("\n🔍 Available Tile Metadata:")
        if self.tile_metadata:
            sample_tile = list(self.tile_metadata.keys())[0]
            sample_meta = self.tile_metadata[sample_tile]
            for key, value in sample_meta.items():
                if key != 'filename':
                    print(f"   • {key}: {value}")
        print("="*80)

def main():
    print("=" * 80)
    print("TP-1 LASER MODULE DATA ANALYSIS (COMBINED TEST + TOP)")
    print("=" * 80)
    print("Test Point: TP1-1")
    print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    analyzer = TP1P1CombinedAnalyzer()
    analyzer.run_all()

if __name__ == "__main__":
    main() 