#!/usr/bin/env python3
"""
TP1-4 Combined Scan and LIV Data Analysis Script
================================================

This script analyzes both Scan and LIV data from TP1-4 test point.
It creates per-tile plots showing LIV data (Power vs Set Laser) and Scan data (PeakWave vs Set Laser)
for each temperature, with separate subplots for Bank 0 and Bank 1.
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
import openpyxl

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TP1P4CombinedAnalyzer:
    def __init__(self, scan_data_path=None, liv_data_path=None):
        script_dir = Path(__file__).parent
        self.scan_data_path = Path(scan_data_path) if scan_data_path else script_dir / "../TP1-4"
        self.liv_data_path = Path(liv_data_path) if liv_data_path else script_dir / "../TP1-4"
        self.output_dir = script_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create TP1-4 subdirectory for VOA plots
        self.tp14_output_dir = self.output_dir / "TP1-4"
        self.tp14_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.scan_files = []
        self.liv_files = []
        self.scan_data = None
        self.liv_data = None
        self.tile_metadata = {}  # Store metadata for each tile

    def extract_serial_number(self, filename):
        # Handle both TP1-4 LIV and VOA files, plus original Scan/LIV patterns
        match = re.search(r'-Y(\d+)-TP1-4 (Scan|LIV|Laser|VOA)\.csv$', filename)
        if match:
            return f"Y{match.group(1)}"
        return None

    def load_scan_files(self):
        # TP1-4 uses "LIV.csv" for scan data
        self.scan_files = sorted(glob.glob(str(self.scan_data_path / "* LIV.csv")))
        print(f"Looking for LIV files in: {self.scan_data_path.absolute()}")
        print(f"Search pattern: {self.scan_data_path / '* LIV.csv'}")
        print(f"Found {len(self.scan_files)} LIV CSV files")
        if len(self.scan_files) == 0:
            print(f"⚠️  No LIV files found. Please check if the TP1-4 directory exists and contains files matching pattern '*-Y*-TP1-4 LIV.csv'")
            # List any CSV files in the directory
            all_csv_files = sorted(glob.glob(str(self.scan_data_path / "*.csv")))
            if all_csv_files:
                print(f"   However, found {len(all_csv_files)} other CSV files in directory:")
                for csv_file in all_csv_files[:5]:  # Show first 5 files
                    print(f"     • {Path(csv_file).name}")
                if len(all_csv_files) > 5:
                    print(f"     • ... and {len(all_csv_files) - 5} more")
        return self.scan_files

    def load_liv_files(self):
        # TP1-4 uses "VOA.csv" for LIV data
        self.liv_files = sorted(glob.glob(str(self.liv_data_path / "* VOA.csv")))
        
        print(f"Looking for VOA files in: {self.liv_data_path.absolute()}")
        print(f"Search pattern: {self.liv_data_path / '* VOA.csv'}")
        print(f"Found {len(self.liv_files)} VOA CSV files")
        if len(self.liv_files) == 0:
            print(f"⚠️  No VOA files found. Please check if the TP1-4 directory exists and contains files matching pattern '*-Y*-TP1-4 VOA.csv'")
            # List any CSV files in the directory
            all_csv_files = sorted(glob.glob(str(self.liv_data_path / "*.csv")))
            if all_csv_files:
                print(f"   However, found {len(all_csv_files)} other CSV files in directory:")
                for csv_file in all_csv_files[:5]:  # Show first 5 files
                    print(f"     • {Path(csv_file).name}")
                if len(all_csv_files) > 5:
                    print(f"     • ... and {len(all_csv_files) - 5} more")
        return self.liv_files

    def load_scan_data(self):
        dfs = []
        for file_path in self.scan_files:
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
            self.scan_data = pd.concat(dfs, ignore_index=True)
            # Convert '-' to NaN in numeric columns
            numeric_cols = ['Set Temp(C)', 'Set Laser(mA)', 'T_PIC(C)', 'Power(mW)', 'MPD_PIC(uA)', 'PeakWave(nm)', 'PeakPower(dBm)']
            for col in numeric_cols:
                if col in self.scan_data.columns:
                    # First replace '-' with NaN, then convert to float
                    self.scan_data[col] = self.scan_data[col].replace('-', pd.NA)
                    self.scan_data[col] = pd.to_numeric(self.scan_data[col], errors='coerce')
            self.scan_data['Time'] = pd.to_datetime(self.scan_data['Time'], format='mixed', errors='coerce')
            self.scan_data = self.scan_data.dropna(subset=['Time'])
            self.scan_data = self.scan_data.sort_values('Time', axis=0)
            print(f"Combined scan data shape: {self.scan_data.shape}")
            print(f"Scan data columns: {list(self.scan_data.columns)}")
            print(f"Captured metadata for {len(self.tile_metadata)} tiles")
        else:
            print("No scan data loaded successfully")

    def load_liv_data(self):
        dfs = []
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for file_path in self.liv_files:
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                print(f"Could not read {file_path} with any encoding")
                continue
            filename = Path(file_path).name
            sn_from_filename = self.extract_serial_number(filename)
            if 'Tile_SN' not in df.columns and sn_from_filename:
                df['Tile_SN'] = sn_from_filename
            df['filename'] = filename
            dfs.append(df)
        if dfs:
            self.liv_data = pd.concat(dfs, ignore_index=True)
            # Convert '-' to NaN in numeric columns
            numeric_cols = ['Set Temp(C)', 'Set Laser(mA)', 'T_PIC(C)', 'Power(mW)', 'MPD_PIC(uA)', 'PeakWave(nm)', 'PeakPower(dBm)']
            for col in numeric_cols:
                if col in self.liv_data.columns:
                    # First replace '-' with NaN, then convert to float
                    self.liv_data[col] = self.liv_data[col].replace('-', pd.NA)
                    self.liv_data[col] = pd.to_numeric(self.liv_data[col], errors='coerce')
            if 'Time' in self.liv_data.columns:
                self.liv_data['Time'] = pd.to_datetime(self.liv_data['Time'], format='mixed', errors='coerce')
                self.liv_data = self.liv_data.dropna(subset=['Time'])
                self.liv_data = self.liv_data.sort_values('Time', axis=0)
            print(f"Combined LIV data shape: {self.liv_data.shape}")
            print(f"LIV data columns: {list(self.liv_data.columns)}")
        else:
            print("No LIV data loaded successfully")

    def get_liv_annotations(self, tile_sn, temp_label, annotation_type='T_op'):
        # Find the LIV file for this tile
        if self.liv_data is None:
            return None
        # Find the filename for this tile
        match = self.liv_data[self.liv_data['filename'].str.contains(tile_sn)]
        if match.empty:
            return None
        # Get the filename
        filename = match['filename'].iloc[0]  # type: ignore
        # For this file, get the requested annotation type
        liv_data = self.liv_data[(self.liv_data['filename'] == filename) & (self.liv_data['label'] == annotation_type)]
        # This is per file, not per temperature, so we annotate the same value for all temps
        liv_val = liv_data['value'].iloc[0] if not liv_data.empty else None  # type: ignore
        return liv_val

    def get_current_column_name(self, data_type='scan'):
        """Get the correct column name for laser/VOA current based on data type."""
        if data_type == 'liv' and self.liv_data is not None and 'Set VOA(mA)' in self.liv_data.columns:
            return 'Set VOA(mA)'
        else:
            return 'Set Laser(mA)'

    def plot_liv_per_tile(self):
        """Create LIV plots for each tile showing Power vs Set Current for each temperature."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for tile_sn in unique_tiles:
            print(f"Creating LIV plot for tile {tile_sn}...")
            
            # Get temperatures for THIS tile only
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = self.liv_data[(self.liv_data['Tile_SN'] == tile_sn) & (self.liv_data['Bank'] == bank)]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                grouped_data = channel_data.groupby(current_col)['Power(mW)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Power range: {grouped_data['Power(mW)'].min():.6f} to {grouped_data['Power(mW)'].max():.6f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data[current_col], grouped_data['Power(mW)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel(f'{current_col.replace("(", " (").replace("mA)", "mA)")}')
                ax.set_ylabel('Power (mW)')
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 50)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'LIV - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"LIV_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ LIV plot saved: {plot_filename}")

    def plot_scan_per_tile(self):
        """Create Scan plots for each tile showing PeakWave vs Set Laser for each temperature."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for tile_sn in unique_tiles:
            print(f"Creating Scan plot for tile {tile_sn}...")
            
            # Get temperatures for THIS tile only
            tile_data = self.scan_data[self.scan_data['Tile_SN'] == tile_sn]
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = self.scan_data[(self.scan_data['Tile_SN'] == tile_sn) & (self.scan_data['Bank'] == bank)]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                grouped_data = channel_data.groupby('Set Laser(mA)')['PeakWave(nm)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Wavelength range: {grouped_data['PeakWave(nm)'].min():.2f} to {grouped_data['PeakWave(nm)'].max():.2f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data['Set Laser(mA)'], grouped_data['PeakWave(nm)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set Laser (mA)')
                ax.set_ylabel('Peak Wavelength (nm)')
                ax.set_xlim(120, 170)
                ax.set_ylim(1300, 1320)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Scan - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"Scan_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Scan plot saved: {plot_filename}")

    def plot_mpd_per_tile(self):
        """Create MPD plots for each tile showing MPD_PIC vs Set Laser for each temperature."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for tile_sn in unique_tiles:
            print(f"Creating MPD plot for tile {tile_sn}...")
            
            # Get temperatures for THIS tile only
            tile_data = self.scan_data[self.scan_data['Tile_SN'] == tile_sn]
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = self.scan_data[(self.scan_data['Tile_SN'] == tile_sn) & (self.scan_data['Bank'] == bank)]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                grouped_data = channel_data.groupby('Set Laser(mA)')['MPD_PIC(uA)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, MPD range: {grouped_data['MPD_PIC(uA)'].min():.6f} to {grouped_data['MPD_PIC(uA)'].max():.6f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data['Set Laser(mA)'], grouped_data['MPD_PIC(uA)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set Laser (mA)')
                ax.set_ylabel('MPD_PIC (uA)')
                ax.set_xlim(120, 170)
                ax.set_ylim(0, 1000)  # Updated y-axis range from 0 to 1000
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'MPD - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"MPD_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ MPD plot saved: {plot_filename}")

    def plot_mpd_vs_tile_combined(self):
        """Create a combined MPD plot showing MPD_PIC vs Tile SN with scatter plots and box plots for each bank, matching tp1p1_power_vs_tile_combined.png format."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        n_tiles = len(sorted_tiles)
        n_channels = 8
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = self.scan_data[self.scan_data['Bank'] == bank]
            # For each channel, plot all tiles
            for ch in range(n_channels):
                ch_mpd = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                    if len(tile_150ma) > 0:
                        ch_mpd.append(tile_150ma['MPD_PIC(uA)'].mean())
                        ch_x.append(i)
                if ch_x:
                    ax.scatter(ch_x, ch_mpd, color=channel_colors[ch], s=60, alpha=0.7, label=f'Ch{ch}', marker='o', edgecolor='black', linewidth=0.5)
            # Box plot for each tile (all channels)
            box_data = []
            for tile_sn in sorted_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                if len(tile_150ma) > 0:
                    box_data.append(tile_150ma['MPD_PIC(uA)'].values)
                else:
                    box_data.append([])
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
                patch.set_linewidth(0.5)
            # Annotate average MPD for each tile
            for i, tile_sn in enumerate(sorted_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                if len(tile_150ma) > 0:
                    avg_mpd = tile_150ma['MPD_PIC(uA)'].mean()
                    ax.text(i, 200, f'avg={avg_mpd:.0f}uA', ha='center', va='bottom', fontsize=7, color='red', rotation=90, fontweight='bold')
            ax.set_title(f'MPD_PIC vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            if bank == 0:
                ax.set_ylabel('MPD_PIC (uA)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(sorted_tiles, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1000)
            ax.grid(True, linestyle='--', alpha=0.3)
            # Legend for each subplot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title='Channel')
        plt.suptitle('MPD_PIC vs Tile SN at 150mA Laser Current', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p4_mpd_vs_tile_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD vs Tile Combined plot saved: {plot_filename}")
        self.create_mpd_vs_tile_html()

    def create_mpd_vs_tile_html(self):
        """Create interactive HTML plot for MPD vs tile."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        print("Creating interactive HTML MPD vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - MPD_PIC vs Tile SN at 150mA', 'Bank 1 - MPD_PIC vs Tile SN at 150mA'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        
        for bank in [0, 1]:
            bank_data = self.scan_data[self.scan_data['Bank'] == bank]
            
            # Get MPD readings at 150mA for each tile and channel
            for tile_sn in sorted_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                
                # Get data closest to 150mA (within ±5mA range)
                tile_150ma_data = tile_data[
                    (tile_data['Set Laser(mA)'] >= 145) & 
                    (tile_data['Set Laser(mA)'] <= 155)
                ]
                
                if len(tile_150ma_data) > 0:
                    # Group by channel and get mean MPD value
                    for channel in sorted(tile_150ma_data['Channel'].unique()):
                        channel_data = tile_150ma_data[tile_150ma_data['Channel'] == channel]
                        if len(channel_data) > 0:
                            avg_mpd = channel_data['MPD_PIC(uA)'].mean()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[tile_sn],
                                    y=[avg_mpd],
                                    mode='markers',
                                    name=f'Bank {bank} - Channel {channel}',
                                    marker=dict(size=8, opacity=0.7),
                                    hovertemplate='<b>Tile:</b> ' + tile_sn + '<br>' +
                                                '<b>Bank:</b> ' + str(bank) + '<br>' +
                                                '<b>Channel:</b> ' + str(channel) + '<br>' +
                                                '<b>MPD_PIC:</b> %{y:.2f} uA<br>' +
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
        fig.update_yaxes(title_text="MPD_PIC (uA)", row=1, col=1, range=[0, 1000])
        fig.update_yaxes(title_text="MPD_PIC (uA)", row=1, col=2, range=[0, 1000])
        
        # Save HTML file
        html_filename = "tp1p4_mpd_vs_tile_combined.html"
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML MPD vs tile plot saved: {html_filename}")
        
        return fig

    def calculate_tuning_efficiency(self):
        """Calculate tuning efficiency (delta_lambda / delta_mA) for each channel."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return None
        
        efficiency_data = []
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        for tile_sn in unique_tiles:
            print(f"Calculating tuning efficiency for tile {tile_sn}...")
            
            tile_data = self.scan_data[self.scan_data['Tile_SN'] == tile_sn]
            unique_channels = sorted(tile_data['Channel'].unique())
            
            for bank in [0, 1]:
                for channel in unique_channels:
                    # Get data for this specific tile, bank, and channel
                    channel_data = tile_data[(tile_data['Bank'] == bank) & (tile_data['Channel'] == channel)]
                    
                    if len(channel_data) > 1:  # Need at least 2 points for slope calculation
                        # Sort by laser current
                        channel_data = channel_data.sort_values('Set Laser(mA)')
                        
                        # Calculate slope using linear regression
                        x = channel_data['Set Laser(mA)'].values
                        y = channel_data['PeakWave(nm)'].values
                        
                        # Remove any NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            # Calculate slope using numpy polyfit
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            
                            # Calculate R-squared for quality assessment
                            y_pred = slope * x_clean + intercept
                            ss_res = np.sum((y_clean - y_pred) ** 2)
                            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            
                            efficiency_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Slope_nm_mA': slope,
                                'Intercept_nm': intercept,
                                'R_squared': r_squared,
                                'Data_points': len(x_clean),
                                'Current_range_mA': f"{x_clean.min():.1f}-{x_clean.max():.1f}",
                                'Wavelength_range_nm': f"{y_clean.min():.2f}-{y_clean.max():.2f}"
                            })
                            
                            print(f"    Bank {bank}, Channel {channel}: slope = {slope:.4f} nm/mA, R² = {r_squared:.3f}")
        
        efficiency_df = pd.DataFrame(efficiency_data)
        print(f"\nCalculated tuning efficiency for {len(efficiency_df)} channel measurements")
        return efficiency_df

    def plot_tuning_efficiency(self, efficiency_df):
        """Plot tuning efficiency vs current for each tile and channel."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data available!")
            return
        
        print(f"Creating tuning efficiency plot with {len(efficiency_df)} data points...")
        
        # Get unique tiles and sort them by date (if available) or alphabetically
        unique_tiles = efficiency_df['Tile_SN'].unique()
        sorted_tiles = sorted(unique_tiles)
        
        # Create figure with 2 subplots (Bank 0 and Bank 1) - matching TP1-1 format exactly
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        # Colors for different channels - matching TP1-1 color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = efficiency_df[efficiency_df['Bank'] == bank]
            
            # Plot each channel separately
            for channel in sorted(bank_data['Channel'].unique()):
                channel_data = bank_data[bank_data['Channel'] == channel]
                color = colors[channel % len(colors)]
                
                # Create scatter plot with tile positions on x-axis
                x_positions = []
                y_values = []
                
                for tile in sorted_tiles:
                    tile_channel_data = channel_data[channel_data['Tile_SN'] == tile]
                    if len(tile_channel_data) > 0:
                        x_positions.append(sorted_tiles.index(tile))
                        y_values.append(tile_channel_data['Slope_nm_mA'].iloc[0])
                
                if x_positions:
                    ax.scatter(x_positions, y_values, 
                             c=color, s=60, alpha=0.7, label=f'Channel {channel}')
            
            # Add annotations for each tile showing average tuning efficiency
            for i, tile in enumerate(sorted_tiles):
                tile_bank_data = bank_data[bank_data['Tile_SN'] == tile]
                if len(tile_bank_data) > 0:
                    avg_efficiency = tile_bank_data['Slope_nm_mA'].mean()
                    
                    # Create annotation text
                    annotation_text = f'η_ave={avg_efficiency:.4f}'
                    
                    ax.text(
                        i, 0.001,  # Position at 0.001 nm/mA on y-axis
                        annotation_text,
                        ha='center',
                        va='bottom',  # Start at 0.001 nm/mA and extend upward
                        fontsize=6,
                        fontweight='bold',
                        color='red',
                        rotation=90
                    )
            
            # Match TP1-1 title format exactly
            ax.set_title(f'Average Tuning Efficiency vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Average Tuning Efficiency (nm/mA)', fontsize=12)
            ax.set_xticks(range(len(sorted_tiles)))
            ax.set_xticklabels(sorted_tiles, rotation=45, fontsize=8)
            ax.set_ylim(0, 0.01)  # Set y-axis range from 0 to 0.01 nm/mA
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save plot in main plots folder
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p4_tuning_efficiency.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Tuning efficiency plot saved: {plot_filename}")
        
        # Create interactive HTML plot using Plotly
        self.create_tuning_efficiency_html(efficiency_df, plots_dir)
        
        # Print summary statistics
        print("\n📊 Tuning Efficiency Summary:")
        for bank in [0, 1]:
            bank_data = efficiency_df[efficiency_df['Bank'] == bank]
            print(f"\nBank {bank}:")
            print(f"  Total measurements: {len(bank_data)}")
            if not bank_data.empty:
                print(f"  Mean slope: {bank_data['Slope_nm_mA'].mean():.4f} nm/mA")
                print(f"  Std slope: {bank_data['Slope_nm_mA'].std():.4f} nm/mA")
                print(f"  Min slope: {bank_data['Slope_nm_mA'].min():.4f} nm/mA")
                print(f"  Max slope: {bank_data['Slope_nm_mA'].max():.4f} nm/mA")
                print(f"  Mean R²: {bank_data['R_squared'].mean():.3f}")
        
        return efficiency_df

    def create_tuning_efficiency_html(self, efficiency_df, plots_dir):
        """Create interactive HTML plot for tuning efficiency."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data available for HTML plot!")
            return
        
        print("Creating interactive HTML tuning efficiency plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - Tuning Efficiency by Channel', 'Bank 1 - Tuning Efficiency by Channel'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Colors for different channels
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Add traces for Bank 0
        bank0_data = efficiency_df[efficiency_df['Bank'] == 0]
        for channel in sorted(bank0_data['Channel'].unique()):
            channel_data = bank0_data[bank0_data['Channel'] == channel]
            color = colors[channel % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=channel_data['Tile_SN'],
                    y=channel_data['Slope_nm_mA'],
                    mode='markers',
                    name=f'Bank 0 - Channel {channel}',
                    marker=dict(color=color, size=8, opacity=0.7),
                    hovertemplate='<b>Tile:</b> %{x}<br>' +
                                '<b>Slope:</b> %{y:.4f} nm/mA<br>' +
                                '<b>Channel:</b> ' + str(channel) + '<br>' +
                                '<b>R²:</b> ' + channel_data['R_squared'].round(3).astype(str) + '<br>' +
                                '<b>Data Points:</b> ' + channel_data['Data_points'].astype(str) + '<br>' +
                                '<b>Current Range:</b> ' + channel_data['Current_range_mA'] + ' mA<br>' +
                                '<b>Wavelength Range:</b> ' + channel_data['Wavelength_range_nm'] + ' nm<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add traces for Bank 1
        bank1_data = efficiency_df[efficiency_df['Bank'] == 1]
        for channel in sorted(bank1_data['Channel'].unique()):
            channel_data = bank1_data[bank1_data['Channel'] == channel]
            color = colors[channel % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=channel_data['Tile_SN'],
                    y=channel_data['Slope_nm_mA'],
                    mode='markers',
                    name=f'Bank 1 - Channel {channel}',
                    marker=dict(color=color, size=8, opacity=0.7),
                    hovertemplate='<b>Tile:</b> %{x}<br>' +
                                '<b>Slope:</b> %{y:.4f} nm/mA<br>' +
                                '<b>Channel:</b> ' + str(channel) + '<br>' +
                                '<b>R²:</b> ' + channel_data['R_squared'].round(3).astype(str) + '<br>' +
                                '<b>Data Points:</b> ' + channel_data['Data_points'].astype(str) + '<br>' +
                                '<b>Current Range:</b> ' + channel_data['Current_range_mA'] + ' mA<br>' +
                                '<b>Wavelength Range:</b> ' + channel_data['Wavelength_range_nm'] + ' nm<extra></extra>',
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='TP1-4 Tuning Efficiency (Δλ/ΔmA) by Tile and Channel',
                x=0.5,
                font=dict(size=20, color='black')
            ),
            width=1600,
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Tile Serial Number", row=1, col=1)
        fig.update_xaxes(title_text="Tile Serial Number", row=1, col=2)
        fig.update_yaxes(title_text="Tuning Efficiency (nm/mA)", row=1, col=1)
        fig.update_yaxes(title_text="Tuning Efficiency (nm/mA)", row=1, col=2)
        
        # Save HTML file
        html_filename = "tp1p4_tuning_efficiency.html"
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML plot saved: {html_filename}")
        
        return fig

    def export_tuning_efficiency(self, efficiency_df):
        """Export tuning efficiency data to CSV."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data to export!")
            return
        
        csv_path = self.data_dir / "tp1p4_tuning_efficiency.csv"
        efficiency_df.to_csv(csv_path, index=False)
        print(f"✅ Tuning efficiency data exported to: {csv_path}")
        
        # Also save as Excel with multiple sheets
        excel_path = self.data_dir / "tp1p4_tuning_efficiency.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            efficiency_df.to_excel(writer, sheet_name='All_Data', index=False)
            
            # Summary by bank
            for bank in [0, 1]:
                bank_data = efficiency_df[efficiency_df['Bank'] == bank]
                if not bank_data.empty:
                    bank_data.to_excel(writer, sheet_name=f'Bank_{bank}', index=False)
            
            # Summary by channel
            for channel in sorted(efficiency_df['Channel'].unique()):
                channel_data = efficiency_df[efficiency_df['Channel'] == channel]
                if not channel_data.empty:
                    channel_data.to_excel(writer, sheet_name=f'Channel_{channel}', index=False)
        
        print(f"✅ Tuning efficiency data exported to Excel: {excel_path}")

    def export_to_xarray(self):
        """Export all combined data (scan and liv) as an xarray DataArray and save as a .nc file with TP1-4 attributes."""
        import xarray as xr
        from datetime import datetime
        
        # Prepare scan data
        scan_df = self.scan_data.copy() if self.scan_data is not None else pd.DataFrame()
        liv_df = self.liv_data.copy() if self.liv_data is not None else pd.DataFrame()
        
        # Add a column to distinguish source
        if not scan_df.empty:
            scan_df['source'] = 'scan'
        if not liv_df.empty:
            liv_df['source'] = 'liv'
        
        # Unify columns for concatenation
        all_columns = set(scan_df.columns).union(set(liv_df.columns))
        scan_df = scan_df.reindex(columns=all_columns)
        liv_df = liv_df.reindex(columns=all_columns)
        combined_df = pd.concat([scan_df, liv_df], ignore_index=True)
        
        # Convert datetime columns to ISO strings
        for col in combined_df.columns:
            if is_datetime64_any_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].astype(str)

        # Convert to xarray Dataset (not DataArray) for mixed types
        data_xr = xr.Dataset.from_dataframe(combined_df)
        
        # Add attributes (only serializable types)
        data_xr.attrs["test_point"] = "TP1-4"
        data_xr.attrs["laser_current_mA"] = 150
        data_xr.attrs["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_xr.attrs["tile_count"] = len(self.tile_metadata)
        data_xr.attrs["scan_data_points"] = len(self.scan_data) if self.scan_data is not None else 0
        data_xr.attrs["liv_data_points"] = len(self.liv_data) if self.liv_data is not None else 0
        
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
            data_xr.attrs["tile_metadata_sample"] = json.dumps({})  # Convert empty dict to JSON string
        
        # Save as NetCDF
        nc_path = self.data_dir / "tp1p4_combined_data.nc"
        data_xr.to_netcdf(nc_path, engine='netcdf4')
        print(f"✅ Exported combined data to NetCDF: {nc_path}")

    def plot_threshold_current_vs_tile_combined(self):
        """Create threshold current analysis plots for each tile using LIV data."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column for threshold analysis: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        
        # Calculate threshold currents for each tile and channel
        threshold_data = []
        for tile_sn in unique_tiles:
            for bank in [0, 1]:
                for ch in range(8):
                    tile_data = self.liv_data[
                        (self.liv_data['Tile_SN'] == tile_sn) & 
                        (self.liv_data['Bank'] == bank) & 
                        (self.liv_data['Channel'] == ch)
                    ]
                    
                    if len(tile_data) > 0:
                        # Calculate threshold current (intercept of linear fit)
                        x = tile_data[current_col].values
                        y = tile_data['Power(mW)'].values
                        
                        # Remove any NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            # Fit linear regression
                            z = np.polyfit(x_clean, y_clean, 1)
                            # Threshold current is x-intercept (where y=0)
                            threshold_current = -z[1] / z[0] if z[0] != 0 else np.nan
                            
                            threshold_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': ch,
                                'Threshold_Current': threshold_current
                            })
        
        if not threshold_data:
            print("No threshold current data calculated!")
            return
        
        threshold_df = pd.DataFrame(threshold_data)
        
        # Create plots
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = threshold_df[threshold_df['Bank'] == bank]
            
            for ch in range(8):
                channel_data = bank_data[bank_data['Channel'] == ch]
                if len(channel_data) > 0:
                    tile_positions = [list(unique_tiles).index(tile) for tile in channel_data['Tile_SN']]
                    ax.scatter(tile_positions, channel_data['Threshold_Current'], 
                             c=colors[ch], s=60, alpha=0.7, 
                             label=f'Channel {ch}' if bank == 0 else "")
            
            ax.set_title(f'Threshold Current - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN', fontsize=12)
            ax.set_ylabel(f'Threshold Current ({current_col.split("(")[1]}', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if bank == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        plt.suptitle(f'Threshold Current vs Tile SN', fontsize=16)
        plt.tight_layout()
        plot_filename = "tp1p4_threshold_current_vs_tiles_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Threshold Current vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_threshold_current_vs_tile_html()

    def plot_slope_efficiency_vs_tile_combined(self):
        """Create slope efficiency analysis plots for each tile using LIV data."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column for slope efficiency analysis: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        
        # Calculate slope efficiencies for each tile and channel
        slope_data = []
        for tile_sn in unique_tiles:
            for bank in [0, 1]:
                for ch in range(8):
                    tile_data = self.liv_data[
                        (self.liv_data['Tile_SN'] == tile_sn) & 
                        (self.liv_data['Bank'] == bank) & 
                        (self.liv_data['Channel'] == ch)
                    ]
                    
                    if len(tile_data) > 0:
                        # Calculate slope efficiency at 1.1x threshold current
                        x = tile_data[current_col].values
                        y = tile_data['Power(mW)'].values
                        
                        # Remove any NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            # Calculate linear fit
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            # Slope efficiency is the slope of the linear fit (mW/mA)
                            slope_efficiency = slope
                            
                            slope_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': ch,
                                'Slope_Efficiency': slope_efficiency
                            })
        
        if not slope_data:
            print("No slope efficiency data calculated!")
            return
        
        slope_df = pd.DataFrame(slope_data)
        
        # Create plots
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = slope_df[slope_df['Bank'] == bank]
            
            for ch in range(8):
                channel_data = bank_data[bank_data['Channel'] == ch]
                if len(channel_data) > 0:
                    tile_positions = [list(unique_tiles).index(tile) for tile in channel_data['Tile_SN']]
                    ax.scatter(tile_positions, channel_data['Slope_Efficiency'], 
                             c=colors[ch], s=60, alpha=0.7, 
                             label=f'Channel {ch}' if bank == 0 else "")
            
            ax.set_title(f'Slope Efficiency - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN', fontsize=12)
            ax.set_ylabel(f'Slope Efficiency (mW/{current_col.split("(")[1]}', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if bank == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        plt.suptitle(f'Slope Efficiency vs Tile SN', fontsize=16)
        plt.tight_layout()
        plot_filename = "tp1p4_slope_efficiency_vs_tiles_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Slope Efficiency vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_slope_efficiency_vs_tile_html()

    def create_threshold_current_vs_tile_html(self):
        """Create interactive HTML plot for threshold current vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print("Creating interactive HTML threshold current vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Bank 0 - Threshold Current vs Tile SN ({current_col.split("(")[0]})', 
                           f'Bank 1 - Threshold Current vs Tile SN ({current_col.split("(")[0]})'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_threshold = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        # Calculate threshold current (intercept of linear fit)
                        x = tile_data[current_col].values
                        y = tile_data['Power(mW)'].values
                        
                        # Remove any NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            # Calculate linear fit
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            # Threshold current is where power = 0, so I_th = -intercept/slope
                            threshold_current = -intercept / slope if slope != 0 else np.nan
                            if not np.isnan(threshold_current) and threshold_current > 0:
                                ch_threshold.append(threshold_current)
                                ch_x.append(i)
                
                if ch_x:
                    fig.add_trace(
                        go.Scatter(
                            x=[sorted_tiles[i] for i in ch_x],
                            y=ch_threshold,
                            mode='markers',
                            name=f'Bank {bank} - Channel {ch}',
                            marker=dict(color=channel_colors[ch], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>Threshold Current:</b> %{y:.2f} mA<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text=f'Threshold Current vs Tile SN ({current_col.split("(")[0]} Current)',
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
        fig.update_yaxes(title_text=f"Threshold Current ({current_col.split('(')[1]}", row=1, col=1, range=[10, 40])
        fig.update_yaxes(title_text=f"Threshold Current ({current_col.split('(')[1]}", row=1, col=2, range=[10, 40])
        
        # Save HTML file
        html_filename = "tp1p4_threshold_current_vs_tiles_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML threshold current vs tile plot saved: {html_filename}")
        
        return fig

    def create_slope_efficiency_vs_tile_html(self):
        """Create interactive HTML plot for slope efficiency vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print("Creating interactive HTML slope efficiency vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Bank 0 - Slope Efficiency vs Tile SN ({current_col.split("(")[0]})', 
                           f'Bank 1 - Slope Efficiency vs Tile SN ({current_col.split("(")[0]})'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_slope_eff = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        x = tile_data[current_col].values
                        y = tile_data['Power(mW)'].values
                        
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            threshold_current = -intercept / slope if slope != 0 else np.nan
                            
                            if not np.isnan(threshold_current) and threshold_current > 0:
                                target_current = 1.1 * threshold_current
                                current_mask = (x_clean >= target_current * 0.9) & (x_clean <= target_current * 1.1)
                                if np.sum(current_mask) > 1:
                                    x_target = x_clean[current_mask]
                                    y_target = y_clean[current_mask]
                                    slope_eff = np.polyfit(x_target, y_target, 1)[0]
                                    ch_slope_eff.append(slope_eff)
                                    ch_x.append(i)
                
                if ch_x:
                    fig.add_trace(
                        go.Scatter(
                            x=[sorted_tiles[i] for i in ch_x],
                            y=ch_slope_eff,
                            mode='markers',
                            name=f'Bank {bank} - Channel {ch}',
                            marker=dict(color=channel_colors[ch], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        f'<b>Slope Efficiency:</b> %{{y:.4f}} mW/{current_col.split("(")[1]}<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<br>' +
                                        '<b>At:</b> 1.1x I_th<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text=f'Slope Efficiency vs Tile SN at 1.1x Threshold Current ({current_col.split("(")[0]})',
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
        fig.update_yaxes(title_text=f"Slope Efficiency (mW/{current_col.split('(')[1]}", row=1, col=1, range=[0, 0.2])
        fig.update_yaxes(title_text=f"Slope Efficiency (mW/{current_col.split('(')[1]}", row=1, col=2, range=[0, 0.2])
        
        # Save HTML file
        html_filename = "tp1p4_slope_efficiency_vs_tiles_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML slope efficiency vs tile plot saved: {html_filename}")
        
        return fig

    def plot_power_vs_tile_combined(self):
        """Create a combined power plot showing power distribution for each tile at 150mA."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column for LIV data: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        n_tiles = len(unique_tiles)
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            box_data = []
            for tile_sn in unique_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data[current_col] >= 145) & (tile_data[current_col] <= 155)]
                if len(tile_150ma) > 0:
                    box_data.append(tile_150ma['Power(mW)'].values)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            # Add scatter points for each channel
            for i, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data[current_col] >= 145) & (tile_data[current_col] <= 155)]
                
                for ch in range(8):
                    channel_data = tile_150ma[tile_150ma['Channel'] == ch]
                    if len(channel_data) > 0:
                        avg_power = channel_data['Power(mW)'].mean()
                        ax.scatter([i] * len(channel_data), channel_data['Power(mW)'], 
                                 c=colors[ch], s=20, alpha=0.6, 
                                 label=f'Channel {ch}' if i == 0 else "")
            
            ax.set_title(f'Power Distribution - Bank {bank} (at 150mA {current_col.split("(")[0]})', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Power (mW)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=8)
            ax.set_ylim(0, 40)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if bank == 0:
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                    markersize=8, label=f'Channel {i}') for i in range(8)]
                ax.legend(handles=handles, loc='upper right', fontsize=8)
        
        plt.suptitle(f'Power Distribution vs Tile SN (at 150mA {current_col.split("(")[0]})', fontsize=16)
        plt.tight_layout()
        plot_filename = "tp1p4_power_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Power vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_power_vs_tile_html()

    def create_power_vs_tile_html(self):
        """Create interactive HTML plot for power vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print("Creating interactive HTML power vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Bank 0 - Power vs Tile SN at 150mA {current_col.split("(")[0]}', 
                           f'Bank 1 - Power vs Tile SN at 150mA {current_col.split("(")[0]}'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        tile_dates = {}
        for tile in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile]
            earliest_date = tile_data['Time'].min()
            tile_dates[tile] = earliest_date
        sorted_tiles = sorted(unique_tiles, key=lambda x: tile_dates[x])
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_power = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    tile_150ma = tile_data[(tile_data[current_col] >= 145) & (tile_data[current_col] <= 155)]
                    if len(tile_150ma) > 0:
                        ch_power.append(tile_150ma['Power(mW)'].mean())
                        ch_x.append(i)
                
                if ch_x:
                    fig.add_trace(
                        go.Scatter(
                            x=[sorted_tiles[i] for i in ch_x],
                            y=ch_power,
                            mode='markers',
                            name=f'Bank {bank} - Channel {ch}',
                            marker=dict(color=channel_colors[ch], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>Power:</b> %{y:.2f} mW<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<br>' +
                                        f'<b>{current_col.split("(")[0]}:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text=f'Power vs Tile SN at 150mA {current_col.split("(")[0]} Current',
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
        fig.update_yaxes(title_text="Power (mW)", row=1, col=1, range=[20, 50])
        fig.update_yaxes(title_text="Power (mW)", row=1, col=2, range=[20, 50])
        
        # Save HTML file
        html_filename = "tp1p4_power_vs_tile_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML power vs tile plot saved: {html_filename}")
        
        return fig

    def run_all(self):
        print("🔄 Loading TP1-4 LIV (scan) data files...")
        self.load_scan_files()
        print("🔄 Loading TP1-4 VOA (LIV) data files...")
        self.load_liv_files()
        print("🔄 Combining TP1-4 LIV (scan) data...")
        self.load_scan_data()
        print("🔄 Combining TP1-4 VOA (LIV) data...")
        self.load_liv_data()
        
        print("\n" + "="*60)
        print("TP1-4 WAVELENGTH SLOPE VS VOA CURRENT")
        print("="*60)
        self.plot_wavelength_voa_slope_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-4 VOA POWER LOSS (0mA to 10mA)")
        print("="*60)
        self.plot_voa_loss_10mA_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-4 VOA POWER LOSS (0mA to 20mA)")
        print("="*60)
        self.plot_voa_loss_20mA_vs_tile_combined()

        print("\n" + "="*60)
        print("TP1-4 POWER VS VOA PLOTS PER TILE")
        print("="*60)
        self.plot_voa_per_tile()

        print("\n" + "="*60)
        print("TP1-4 WAVELENGTH VS VOA PLOTS PER TILE")
        print("="*60)
        self.plot_wavelength_vs_voa_per_tile()

        print("\n" + "="*60)
        print("TP1-4 SCAN PLOTS PER TILE")
        print("="*60)
        self.plot_scan_per_tile()
        
        print("\n" + "="*60)
        print("TP1-4 MPD PLOTS PER TILE")
        print("="*60)
        self.plot_mpd_per_tile()
        
        print("\n" + "="*60)
        print("TP1-4 MPD RESPONSIVITY PER TILE")
        print("="*60)
        self.plot_mpd_responsivity_per_tile()
        
        print("\n" + "="*60)
        print("TP1-4 MPD RESPONSIVITY VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_responsivity_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-4 MPD VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-4 TUNING EFFICIENCY")
        print("="*60)
        efficiency_df = self.calculate_tuning_efficiency()
        if efficiency_df is not None:
            self.plot_tuning_efficiency(efficiency_df)
            self.export_tuning_efficiency(efficiency_df)
        
        # Export to xarray/netcdf
        print("\n" + "="*60)
        print("TP1-4 EXPORT TO NETCDF")
        print("="*60)
        self.export_to_xarray()
        
        print("\n" + "="*80)
        print("TP1-4 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"✅ PNG plots saved to: {self.output_dir.absolute()}")
        
        # List generated plots
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique() if self.scan_data is not None else []
        print("📋 Generated plots:")
        
        # New VOA analysis plots
        print(f"   • tp1p4_wavelength_voa_slope_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p4_voa_loss_10mA_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p4_voa_loss_20mA_vs_tile_combined.png (in plots folder)")
        
        for tile_sn in unique_tiles:
            print(f"   • VOA_{tile_sn}.png (Power vs VOA) (in TP1-4 folder)")
            print(f"   • Wavelength_vs_VOA_{tile_sn}.png (in TP1-4 folder)")
            print(f"   • Scan_{tile_sn}.png")
            print(f"   • MPD_{tile_sn}.png")
            print(f"   • MPD_Responsivity_{tile_sn}.png (in plots folder)")
            print(f"   • MPD_Responsivity_Summary_{tile_sn}.csv (in plots folder)")
        print(f"   • MPD_Responsivity_Summary_All_Tiles.png (in plots folder)")
        print(f"   • MPD_Responsivity_Summary_All_Tiles.csv (in plots folder)")
        print(f"   • tp1p4_mpd_responsivity_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p4_mpd_responsivity_vs_tile_combined.html (in plots folder)")
        print(f"   • tp1p4_tuning_efficiency.png (in plots folder)")
        print(f"   • tp1p4_tuning_efficiency.html (in plots folder)")
        print(f"   • tp1p4_mpd_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p4_mpd_vs_tile_combined.html (in plots folder)")
        
        print(f"   • tp1p4_combined_data.nc (in data folder)")
        print(f"   • tp1p4_tuning_efficiency.csv (in data folder)")
        print(f"   • tp1p4_tuning_efficiency.xlsx (in data folder)")
        
        # New VOA analysis data exports
        print(f"   • tp1p4_wavelength_voa_slope_data.csv (in data folder)")
        print(f"   • tp1p4_voa_loss_10mA_data.csv (in data folder)")
        print(f"   • tp1p4_voa_loss_20mA_data.csv (in data folder)")
        
        print("\n📊 Metadata Summary:")
        print(f"   • Analyzed {len(self.tile_metadata)} tiles")
        print(f"   • All measurements performed at 150mA laser current")
        print(f"   • Scan data points: {len(self.scan_data) if self.scan_data is not None else 0}")
        print(f"   • VOA data points: {len(self.liv_data) if self.liv_data is not None else 0}")
        if efficiency_df is not None:
            print(f"   • Tuning efficiency measurements: {len(efficiency_df)}")
        print("\n🔍 Available Tile Metadata:")
        if self.tile_metadata:
            sample_tile = list(self.tile_metadata.keys())[0]
            sample_meta = self.tile_metadata[sample_tile]
            for key, value in sample_meta.items():
                if key != 'filename':
                    print(f"   • {key}: {value}")
        print("="*80)

    def plot_mpd_responsivity_per_tile(self):
        """Create MPD responsivity plots for each tile showing MPD responsivity vs Set Laser for each temperature.
        MPD responsivity is defined as MPD_PIC(uA) / Power(mW)."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for tile_sn in unique_tiles:
            print(f"Creating MPD responsivity plot for tile {tile_sn}...")
            
            # Get tile data and filter out invalid values
            tile_data = self.scan_data[self.scan_data['Tile_SN'] == tile_sn].copy()
            # Filter out rows where Power is 0 or NaN to avoid division by zero
            tile_data = tile_data[(tile_data['Power(mW)'] > 0) & (tile_data['MPD_PIC(uA)'].notna()) & (tile_data['Power(mW)'].notna())]
            
            if len(tile_data) == 0:
                print(f"  ❌ No valid data for tile {tile_sn}")
                continue
                
            # Calculate MPD responsivity
            tile_data['MPD_Responsivity(uA/mW)'] = tile_data['MPD_PIC(uA)'] / tile_data['Power(mW)']
            
            # Get temperatures for this tile
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            # Create figure with 2 subplots (Bank 0 and Bank 1)
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = tile_data[tile_data['Bank'] == bank]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                # Group by Set Laser current and calculate mean responsivity
                                grouped_data = channel_data.groupby('Set Laser(mA)')['MPD_Responsivity(uA/mW)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Responsivity range: {grouped_data['MPD_Responsivity(uA/mW)'].min():.3f} to {grouped_data['MPD_Responsivity(uA/mW)'].max():.3f} uA/mW")
                                
                                # Use different colors for different channels, with different line styles for temperatures
                                color_idx = channel_idx % len(colors)
                                line_style = ['-', '--', '-.'][temp_idx % 3]
                                
                                ax.plot(grouped_data['Set Laser(mA)'], grouped_data['MPD_Responsivity(uA/mW)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], linestyle=line_style,
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    # Calculate and display average responsivity for this bank
                    bank_avg_responsivity = tile_bank_data['MPD_Responsivity(uA/mW)'].mean()
                    ax.text(0.02, 0.98, f'Avg: {bank_avg_responsivity:.3f} uA/mW', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                           fontsize=10, fontweight='bold')
                
                ax.set_title(f'Bank {bank}', fontsize=14)
                ax.set_xlabel('Set Laser (mA)', fontsize=12)
                ax.set_ylabel('MPD Responsivity (uA/mW)', fontsize=12)
                ax.set_xlim(120, 170)
                
                # Set reasonable y-axis limits based on data
                if plotted:
                    y_min = tile_bank_data['MPD_Responsivity(uA/mW)'].min() * 0.9
                    y_max = tile_bank_data['MPD_Responsivity(uA/mW)'].max() * 1.1
                    ax.set_ylim(max(0, y_min), y_max)
                
                ax.grid(True, linestyle='--', alpha=0.3)
                if plotted:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.suptitle(f'MPD Responsivity - Tile {tile_sn}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot with unique filename for each tile directly under TP1-4 folder
            plot_filename = f"MPD_Responsivity_{tile_sn}.png"
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ MPD responsivity plot saved: {plot_filename}")
            
            # Also create a summary CSV file for this tile with responsivity data
            summary_data = []
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                for temp in temps:
                    temp_data = bank_data[bank_data['Set Temp(C)'] == temp]
                    for channel in unique_channels:
                        channel_data = temp_data[temp_data['Channel'] == channel]
                        if len(channel_data) > 0:
                            avg_responsivity = channel_data['MPD_Responsivity(uA/mW)'].mean()
                            avg_laser = channel_data['Set Laser(mA)'].mean()
                            avg_power = channel_data['Power(mW)'].mean()
                            avg_mpd = channel_data['MPD_PIC(uA)'].mean()
                            
                            summary_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Temp_C': temp,
                                'Avg_Laser_mA': avg_laser,
                                'Avg_Power_mW': avg_power,
                                'Avg_MPD_uA': avg_mpd,
                                'Avg_Responsivity_uA_per_mW': avg_responsivity,
                                'Data_Points': len(channel_data)
                            })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                csv_filename = f"MPD_Responsivity_Summary_{tile_sn}.csv"
                summary_df.to_csv(self.output_dir / csv_filename, index=False)
                print(f"✅ MPD responsivity summary saved: {csv_filename}")
        
        print(f"\n📂 All MPD responsivity plots and summaries saved to: {self.output_dir}")
        
        # Create an overall summary of all tiles
        self.create_mpd_responsivity_summary()

    def create_mpd_responsivity_summary(self):
        """Create a summary plot and data file showing MPD responsivity statistics for all tiles."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        print("Creating MPD responsivity summary for all tiles...")
        
        # Calculate responsivity for all data
        valid_data = self.scan_data[(self.scan_data['Power(mW)'] > 0) & 
                                   (self.scan_data['MPD_PIC(uA)'].notna()) & 
                                   (self.scan_data['Power(mW)'].notna())].copy()
        
        if len(valid_data) == 0:
            print("❌ No valid data for responsivity calculation")
            return
        
        valid_data['MPD_Responsivity(uA/mW)'] = valid_data['MPD_PIC(uA)'] / valid_data['Power(mW)']
        
        # Get unique tiles
        unique_tiles = sorted(valid_data['Tile_SN'].dropna().unique())
        
        # Create summary statistics
        summary_stats = []
        for tile_sn in unique_tiles:
            tile_data = valid_data[valid_data['Tile_SN'] == tile_sn]
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                if len(bank_data) > 0:
                    summary_stats.append({
                        'Tile_SN': tile_sn,
                        'Bank': bank,
                        'Mean_Responsivity': bank_data['MPD_Responsivity(uA/mW)'].mean(),
                        'Std_Responsivity': bank_data['MPD_Responsivity(uA/mW)'].std(),
                        'Min_Responsivity': bank_data['MPD_Responsivity(uA/mW)'].min(),
                        'Max_Responsivity': bank_data['MPD_Responsivity(uA/mW)'].max(),
                        'Data_Points': len(bank_data)
                    })
        
        if not summary_stats:
            print("❌ No summary statistics generated")
            return
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create summary plot
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_stats = summary_df[summary_df['Bank'] == bank]
            
            if len(bank_stats) > 0:
                x_positions = range(len(bank_stats))
                means = bank_stats['Mean_Responsivity'].values
                stds = bank_stats['Std_Responsivity'].values
                tiles = bank_stats['Tile_SN'].values
                
                # Plot mean responsivity with error bars
                ax.errorbar(x_positions, means, yerr=stds, 
                           marker='o', capsize=5, capthick=2, markersize=6,
                           linewidth=2, label='Mean ± Std')
                
                # Add individual points for min/max
                ax.scatter(x_positions, bank_stats['Min_Responsivity'], 
                          marker='v', color='red', alpha=0.7, s=30, label='Min')
                ax.scatter(x_positions, bank_stats['Max_Responsivity'], 
                          marker='^', color='green', alpha=0.7, s=30, label='Max')
                
                ax.set_title(f'MPD Responsivity Summary - Bank {bank}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Tile SN', fontsize=12)
                ax.set_ylabel('MPD Responsivity (uA/mW)', fontsize=12)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(tiles, rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend()
                
                # Add statistics text
                overall_mean = bank_stats['Mean_Responsivity'].mean()
                overall_std = bank_stats['Mean_Responsivity'].std()
                stats_text = f'Overall: μ={overall_mean:.3f}, σ={overall_std:.3f} uA/mW'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                       fontsize=10)
        
        plt.suptitle('MPD Responsivity Summary - All Tiles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save summary plot directly under TP1-4 folder
        summary_plot_filename = "MPD_Responsivity_Summary_All_Tiles.png"
        plt.savefig(self.output_dir / summary_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD responsivity summary plot saved: {summary_plot_filename}")
        
        # Save summary CSV directly under TP1-4 folder
        summary_csv_filename = "MPD_Responsivity_Summary_All_Tiles.csv"
        summary_df.to_csv(self.output_dir / summary_csv_filename, index=False)
        print(f"✅ MPD responsivity summary CSV saved: {summary_csv_filename}")
        
        # Print summary statistics
        print(f"\n📊 MPD Responsivity Summary:")
        print(f"   • Total tiles analyzed: {len(unique_tiles)}")
        print(f"   • Total data points: {len(valid_data)}")
        for bank in [0, 1]:
            bank_stats = summary_df[summary_df['Bank'] == bank]
            if len(bank_stats) > 0:
                overall_mean = bank_stats['Mean_Responsivity'].mean()
                overall_std = bank_stats['Mean_Responsivity'].std()
                print(f"   • Bank {bank}: μ={overall_mean:.3f} ± {overall_std:.3f} uA/mW")
        
        return summary_df

    def plot_mpd_responsivity_vs_tile_combined(self):
        """Create a combined MPD responsivity plot showing responsivity vs Tile SN with scatter plots and box plots for each bank.
        MPD responsivity is converted to A/W units (from uA/mW)."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        # Filter out invalid data
        valid_data = self.scan_data[(self.scan_data['Power(mW)'] > 0) & 
                                   (self.scan_data['MPD_PIC(uA)'].notna()) & 
                                   (self.scan_data['Power(mW)'].notna())].copy()
        
        if len(valid_data) == 0:
            print("❌ No valid data for responsivity calculation")
            return
        
        # Calculate MPD responsivity in A/W (convert from uA/mW)
        # uA/mW = 1e-6 A / 1e-3 W = 1e-3 A/W
        valid_data['MPD_Responsivity(A/W)'] = (valid_data['MPD_PIC(uA)'] / valid_data['Power(mW)']) * 1e-3
        
        # Get data at 150mA laser current
        data_150ma = valid_data[(valid_data['Set Laser(mA)'] >= 145) & (valid_data['Set Laser(mA)'] <= 155)]
        
        if len(data_150ma) == 0:
            print("❌ No data at 150mA laser current")
            return
        
        unique_tiles = sorted(data_150ma['Tile_SN'].dropna().unique())
        n_tiles = len(unique_tiles)
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            
            # For each channel, plot all tiles
            for ch in range(8):
                ch_responsivity = []
                ch_x = []
                for i, tile_sn in enumerate(unique_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        ch_responsivity.append(tile_data['MPD_Responsivity(A/W)'].mean())
                        ch_x.append(i)
                
                if ch_x:
                    ax.scatter(ch_x, ch_responsivity, color=channel_colors[ch], s=60, alpha=0.7, 
                             label=f'Ch{ch}', marker='o', edgecolor='black', linewidth=0.5)
            
            # Box plot for each tile (all channels)
            box_data = []
            for tile_sn in unique_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                if len(tile_data) > 0:
                    box_data.append(tile_data['MPD_Responsivity(A/W)'].values)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
                patch.set_linewidth(0.5)
            
            # Annotate average responsivity for each tile
            for i, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                if len(tile_data) > 0:
                    avg_responsivity = tile_data['MPD_Responsivity(A/W)'].mean()
                    ax.text(i, 0.045, f'avg={avg_responsivity:.4f}', ha='center', va='bottom', 
                           fontsize=7, color='red', rotation=90, fontweight='bold')
            
            ax.set_title(f'MPD Responsivity vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            if bank == 0:
                ax.set_ylabel('MPD Responsivity (A/W)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(unique_tiles, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 0.05)  # Set y-axis range from 0 to 0.05 A/W
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Annotate average responsivity for each tile at 0.035 A/W level
            for i, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                if len(tile_data) > 0:
                    avg_responsivity = tile_data['MPD_Responsivity(A/W)'].mean()
                    ax.text(i, 0.035, f'avg={avg_responsivity:.4f}', ha='center', va='bottom', 
                           fontsize=7, color='red', rotation=90, fontweight='bold')
            
            # Legend for each subplot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title='Channel')
        
        plt.suptitle('MPD Responsivity vs Tile SN at 150mA Laser Current', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        
        # Save plot in main plots folder
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p4_mpd_responsivity_vs_tile_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD Responsivity vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_mpd_responsivity_vs_tile_html(data_150ma, unique_tiles)

    def create_mpd_responsivity_vs_tile_html(self, data_150ma, unique_tiles):
        """Create interactive HTML plot for MPD responsivity vs tile."""
        print("Creating interactive HTML MPD responsivity vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - MPD Responsivity vs Tile SN at 150mA', 'Bank 1 - MPD Responsivity vs Tile SN at 150mA'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            
            for ch in range(8):
                ch_responsivity = []
                ch_x = []
                for i, tile_sn in enumerate(unique_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        ch_responsivity.append(tile_data['MPD_Responsivity(A/W)'].mean())
                        ch_x.append(i)
                
                if ch_x:
                    fig.add_trace(
                        go.Scatter(
                            x=[unique_tiles[i] for i in ch_x],
                            y=ch_responsivity,
                            mode='markers',
                            name=f'Bank {bank} - Channel {ch}',
                            marker=dict(color=channel_colors[ch], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>Responsivity:</b> %{y:.4f} A/W<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<br>' +
                                        '<b>Laser Current:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='MPD Responsivity vs Tile SN at 150mA Laser Current',
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
        fig.update_yaxes(title_text="MPD Responsivity (A/W)", row=1, col=1, range=[0, 0.05])
        fig.update_yaxes(title_text="MPD Responsivity (A/W)", row=1, col=2, range=[0, 0.05])
        
        # Save HTML file
        html_filename = "tp1p4_mpd_responsivity_vs_tile_combined.html"
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML MPD responsivity vs tile plot saved: {html_filename}")
        
        return fig

    def plot_voa_per_tile(self):
        """Create VOA plots for each tile showing Power vs Set VOA Current for each temperature."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for tile_sn in unique_tiles:
            print(f"Creating Power vs VOA plot for tile {tile_sn}...")
            
            # Get temperatures for THIS tile only
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = self.liv_data[(self.liv_data['Tile_SN'] == tile_sn) & (self.liv_data['Bank'] == bank)]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0:
                                grouped_data = channel_data.groupby(current_col)['Power(mW)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Power range: {grouped_data['Power(mW)'].min():.6f} to {grouped_data['Power(mW)'].max():.6f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data[current_col], grouped_data['Power(mW)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel(f'{current_col.replace("(", " (").replace("mA)", "mA)")}')
                ax.set_ylabel('Power (mW)')
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 50)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Power vs VOA - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"VOA_{tile_sn}.png"
            plt.savefig(self.tp14_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Power vs VOA plot saved: {plot_filename}")

    def plot_wavelength_vs_voa_per_tile(self):
        """Create Wavelength vs VOA plots for each tile showing PeakWave vs Set VOA Current for each temperature."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        # Check if wavelength data exists in VOA data
        if 'PeakWave(nm)' not in self.liv_data.columns:
            print("⚠️  No wavelength data found in VOA files. Skipping Wavelength vs VOA plots.")
            return
        
        current_col = self.get_current_column_name('liv')
        print(f"Using current column for Wavelength vs VOA plots: {current_col}")
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for tile_sn in unique_tiles:
            print(f"Creating Wavelength vs VOA plot for tile {tile_sn}...")
            
            # Get temperatures for THIS tile only
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            available_temps = sorted(tile_data['Set Temp(C)'].dropna().unique().tolist())
            print(f"  Available temperatures for {tile_sn}: {available_temps}")
            
            # Use up to 3 temperatures, prioritizing the middle range
            if len(available_temps) >= 3:
                temps = [available_temps[0], available_temps[len(available_temps)//2], available_temps[-1]]
            else:
                temps = available_temps
            print(f"  Selected temperatures for plotting: {temps}")
            
            # Get all unique channels for this tile
            unique_channels = sorted(tile_data['Channel'].unique())
            print(f"  Available channels: {unique_channels}")
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for bank in [0, 1]:
                ax = axs[bank]
                tile_bank_data = self.liv_data[(self.liv_data['Tile_SN'] == tile_sn) & (self.liv_data['Bank'] == bank)]
                print(f"    Bank {bank}: {len(tile_bank_data)} rows")
                plotted = False
                
                for temp_idx, temp in enumerate(temps):
                    temp_data = tile_bank_data[tile_bank_data['Set Temp(C)'] == temp]
                    print(f"      Temp {temp}: {len(temp_data)} rows")
                    
                    if len(temp_data) > 0:
                        # Plot each channel separately
                        for channel_idx, channel in enumerate(unique_channels):
                            channel_data = temp_data[temp_data['Channel'] == channel]
                            if len(channel_data) > 0 and 'PeakWave(nm)' in channel_data.columns:
                                # Filter out NaN wavelength values
                                valid_data = channel_data.dropna(subset=['PeakWave(nm)'])
                                if len(valid_data) > 0:
                                    grouped_data = valid_data.groupby(current_col)['PeakWave(nm)'].mean().reset_index()
                                    print(f"        Channel {channel}: {len(grouped_data)} points, Wavelength range: {grouped_data['PeakWave(nm)'].min():.2f} to {grouped_data['PeakWave(nm)'].max():.2f}")
                                    
                                    # Use different colors for different channels
                                    color_idx = channel_idx % len(colors)
                                    ax.plot(grouped_data[current_col], grouped_data['PeakWave(nm)'], 
                                           marker='o', linewidth=1.5, markersize=4, 
                                           color=colors[color_idx], 
                                           label=f'{temp:.1f}°C Ch{channel}')
                                    plotted = True
                
                if not plotted:
                    print(f"    ❌ No wavelength data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No wavelength data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel(f'{current_col.replace("(", " (").replace("mA)", "mA)")}')
                ax.set_ylabel('Peak Wavelength (nm)')
                ax.set_xlim(0, 20)
                ax.set_ylim(1300, 1320)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Wavelength vs VOA - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"Wavelength_vs_VOA_{tile_sn}.png"
            plt.savefig(self.tp14_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Wavelength vs VOA plot saved: {plot_filename}")

    def plot_wavelength_voa_slope_vs_tile_combined(self):
        """Create a combined plot showing wavelength slope vs VOA current for each tile.
        Slope is calculated as delta nm per mA of VOA current.
        """
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        # Calculate wavelength slope for each tile/bank/channel combination
        slope_data = []
        
        unique_tiles = sorted(self.liv_data['Tile_SN'].dropna().unique())
        for tile_sn in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    
                    if len(channel_data) > 1:
                        # Sort by VOA current
                        channel_data = channel_data.sort_values('Set VOA(mA)')
                        
                        # Calculate slope using linear regression
                        x = channel_data['Set VOA(mA)'].values
                        y = channel_data['PeakWave(nm)'].values
                        
                        # Remove any NaN values
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            # Calculate slope using numpy polyfit
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            
                            slope_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Slope_nm_per_mA': slope,
                                'Intercept_nm': intercept,
                                'Data_points': len(x_clean)
                            })
        
        if not slope_data:
            print("❌ No wavelength slope data calculated")
            return
        
        slope_df = pd.DataFrame(slope_data)
        
        # Create plot with 2 subplots (one for each bank)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = slope_df[slope_df['Bank'] == bank]
            
            if len(bank_data) == 0:
                continue
            
            # Scatter plot for each channel
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    x_positions = [unique_tiles.index(tile) for tile in channel_data['Tile_SN']]
                    y_values = channel_data['Slope_nm_per_mA'].values
                    
                    ax.scatter(x_positions, y_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50,
                              label=f'Ch{channel}')
            
            # Box plot
            box_data = []
            for tile_sn in unique_tiles:
                tile_slopes = bank_data[bank_data['Tile_SN'] == tile_sn]['Slope_nm_per_mA'].values
                if len(tile_slopes) > 0:
                    box_data.append(tile_slopes)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(len(unique_tiles)), patch_artist=True, 
                           showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
            
            # Calculate and annotate average slope for each tile
            for i, tile_sn in enumerate(unique_tiles):
                tile_slopes = bank_data[bank_data['Tile_SN'] == tile_sn]['Slope_nm_per_mA'].values
                if len(tile_slopes) > 0:
                    avg_slope = np.mean(tile_slopes)
                    ax.text(i, 0.005, f'avg={avg_slope:.2f}nm/mA', 
                           ha='center', va='bottom', fontsize=8, color='red', 
                           rotation=90, fontweight='bold')
            
            ax.set_title(f'Wavelength Slope vs VOA Current - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Wavelength Slope (nm/mA)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=9)
            ax.set_ylim(-0.01, 0.01)
            ax.grid(True, linestyle='--', alpha=0.3)
            if bank == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.suptitle('Wavelength Slope vs VOA Current by Tile', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp1p4_wavelength_voa_slope_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Wavelength slope vs tile plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp1p4_wavelength_voa_slope_data.csv"
        slope_df.to_csv(csv_path, index=False)
        print(f"✅ Wavelength slope data exported to: {csv_path}")

    def plot_voa_loss_10mA_vs_tile_combined(self):
        """Create a combined plot showing power loss from 0mA to 10mA VOA current for each tile."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        # Calculate power loss from 0mA to 10mA for each tile/bank/channel combination
        loss_data = []
        
        unique_tiles = sorted(self.liv_data['Tile_SN'].dropna().unique())
        for tile_sn in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    
                    # Get power at 0mA and 10mA
                    power_0mA = channel_data[channel_data['Set VOA(mA)'] == 0]['Power(mW)'].values
                    power_10mA = channel_data[channel_data['Set VOA(mA)'] == 10]['Power(mW)'].values
                    
                    if len(power_0mA) > 0 and len(power_10mA) > 0:
                        # Calculate power loss in dB
                        # Loss = -10 * log10(P_10mA / P_0mA)
                        if power_0mA[0] > 0 and power_10mA[0] > 0:
                            loss_dB = -10 * np.log10(power_10mA[0] / power_0mA[0])
                            
                            loss_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Power_0mA': power_0mA[0],
                                'Power_10mA': power_10mA[0],
                                'Loss_dB': loss_dB
                            })
        
        if not loss_data:
            print("❌ No power loss data calculated")
            return
        
        loss_df = pd.DataFrame(loss_data)
        
        # Create plot with 2 subplots (one for each bank)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = loss_df[loss_df['Bank'] == bank]
            
            if len(bank_data) == 0:
                continue
            
            # Scatter plot for each channel
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    x_positions = [unique_tiles.index(tile) for tile in channel_data['Tile_SN']]
                    y_values = channel_data['Loss_dB'].values
                    
                    ax.scatter(x_positions, y_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50,
                              label=f'Ch{channel}')
            
            # Box plot
            box_data = []
            for tile_sn in unique_tiles:
                tile_losses = bank_data[bank_data['Tile_SN'] == tile_sn]['Loss_dB'].values
                if len(tile_losses) > 0:
                    box_data.append(tile_losses)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(len(unique_tiles)), patch_artist=True, 
                           showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
            
            # Calculate and annotate average loss for each tile
            for i, tile_sn in enumerate(unique_tiles):
                tile_losses = bank_data[bank_data['Tile_SN'] == tile_sn]['Loss_dB'].values
                if len(tile_losses) > 0:
                    avg_loss = np.mean(tile_losses)
                    ax.text(i, 4, f'avg={avg_loss:.1f}dB', 
                           ha='center', va='bottom', fontsize=8, color='red', 
                           rotation=90, fontweight='bold')
            
            ax.set_title(f'VOA Power Loss (0mA to 10mA) - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Power Loss (dB)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=9)
            ax.set_ylim(1, 5)
            ax.grid(True, linestyle='--', alpha=0.3)
            if bank == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.suptitle('VOA Power Loss (0mA to 10mA) by Tile', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp1p4_voa_loss_10mA_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ VOA power loss (10mA) vs tile plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp1p4_voa_loss_10mA_data.csv"
        loss_df.to_csv(csv_path, index=False)
        print(f"✅ VOA power loss (10mA) data exported to: {csv_path}")

    def plot_voa_loss_20mA_vs_tile_combined(self):
        """Create a combined plot showing power loss from 0mA to 20mA VOA current for each tile."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        # Calculate power loss from 0mA to 20mA for each tile/bank/channel combination
        loss_data = []
        
        unique_tiles = sorted(self.liv_data['Tile_SN'].dropna().unique())
        for tile_sn in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    
                    # Get power at 0mA and 20mA
                    power_0mA = channel_data[channel_data['Set VOA(mA)'] == 0]['Power(mW)'].values
                    power_20mA = channel_data[channel_data['Set VOA(mA)'] == 20]['Power(mW)'].values
                    
                    if len(power_0mA) > 0 and len(power_20mA) > 0:
                        # Calculate power loss in dB
                        # Loss = -10 * log10(P_20mA / P_0mA)
                        if power_0mA[0] > 0 and power_20mA[0] > 0:
                            loss_dB = -10 * np.log10(power_20mA[0] / power_0mA[0])
                            
                            loss_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Power_0mA': power_0mA[0],
                                'Power_20mA': power_20mA[0],
                                'Loss_dB': loss_dB
                            })
        
        if not loss_data:
            print("❌ No power loss data calculated")
            return
        
        loss_df = pd.DataFrame(loss_data)
        
        # Create plot with 2 subplots (one for each bank)
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = loss_df[loss_df['Bank'] == bank]
            
            if len(bank_data) == 0:
                continue
            
            # Scatter plot for each channel
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    x_positions = [unique_tiles.index(tile) for tile in channel_data['Tile_SN']]
                    y_values = channel_data['Loss_dB'].values
                    
                    ax.scatter(x_positions, y_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50,
                              label=f'Ch{channel}')
            
            # Box plot
            box_data = []
            for tile_sn in unique_tiles:
                tile_losses = bank_data[bank_data['Tile_SN'] == tile_sn]['Loss_dB'].values
                if len(tile_losses) > 0:
                    box_data.append(tile_losses)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(len(unique_tiles)), patch_artist=True, 
                           showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
            
            # Calculate and annotate average loss for each tile
            for i, tile_sn in enumerate(unique_tiles):
                tile_losses = bank_data[bank_data['Tile_SN'] == tile_sn]['Loss_dB'].values
                if len(tile_losses) > 0:
                    avg_loss = np.mean(tile_losses)
                    ax.text(i, 4, f'avg={avg_loss:.1f}dB', 
                           ha='center', va='bottom', fontsize=8, color='red', 
                           rotation=90, fontweight='bold')
            
            ax.set_title(f'VOA Power Loss (0mA to 20mA) - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Power Loss (dB)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=9)
            ax.set_ylim(1, 5)
            ax.grid(True, linestyle='--', alpha=0.3)
            if bank == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.suptitle('VOA Power Loss (0mA to 20mA) by Tile', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp1p4_voa_loss_20mA_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ VOA power loss (20mA) vs tile plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp1p4_voa_loss_20mA_data.csv"
        loss_df.to_csv(csv_path, index=False)
        print(f"✅ VOA power loss (20mA) data exported to: {csv_path}")

def main():
    print("=" * 80)
    print("TP1-4 LASER MODULE DATA ANALYSIS (PER-TILE PLOTS)")
    print("=" * 80)
    print("Test Point: TP1-4")
    print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    analyzer = TP1P4CombinedAnalyzer()
    analyzer.run_all()

if __name__ == "__main__":
    main() 