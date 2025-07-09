#!/usr/bin/env python3
"""
TP1-2 Combined Scan and LIV Data Analysis Script
================================================

This script analyzes both Scan and LIV data from TP1-2 test point.
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

class TP1P2CombinedAnalyzer:
    def __init__(self, scan_data_path=None, liv_data_path=None):
        script_dir = Path(__file__).parent
        self.scan_data_path = Path(scan_data_path) if scan_data_path else script_dir / "../TP1-2"
        self.liv_data_path = Path(liv_data_path) if liv_data_path else script_dir / "../TP1-2"
        self.output_dir = script_dir / "plots" / "TP1-2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.scan_files = []
        self.liv_files = []
        self.scan_data = None
        self.liv_data = None
        self.tile_metadata = {}  # Store metadata for each tile

    def extract_serial_number(self, filename):
        match = re.search(r'-Y(\d+)-TP1-2 (Scan|LIV)\.csv$', filename)
        if match:
            return f"Y{match.group(1)}"
        return None

    def load_scan_files(self):
        self.scan_files = sorted(glob.glob(str(self.scan_data_path / "* Scan.csv")))
        print(f"Found {len(self.scan_files)} Scan CSV files")
        return self.scan_files

    def load_liv_files(self):
        self.liv_files = sorted(glob.glob(str(self.liv_data_path / "* LIV.csv")))
        print(f"Found {len(self.liv_files)} LIV CSV files")
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

    def plot_liv_per_tile(self):
        """Create LIV plots for each tile showing Power vs Set Laser for each temperature."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
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
                                grouped_data = channel_data.groupby('Set Laser(mA)')['Power(mW)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Power range: {grouped_data['Power(mW)'].min():.6f} to {grouped_data['Power(mW)'].max():.6f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data['Set Laser(mA)'], grouped_data['Power(mW)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set Laser (mA)')
                ax.set_ylabel('Power (mW)')
                ax.set_xlim(0, 200)
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
        plot_filename = "tp1p2_mpd_vs_tile_combined.png"
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
        html_filename = "tp1p2_mpd_vs_tile_combined.html"
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
        plot_filename = "tp1p2_tuning_efficiency.png"
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
                text='TP1-2 Tuning Efficiency (Δλ/ΔmA) by Tile and Channel',
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
        html_filename = "tp1p2_tuning_efficiency.html"
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML plot saved: {html_filename}")
        
        return fig

    def export_tuning_efficiency(self, efficiency_df):
        """Export tuning efficiency data to CSV."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data to export!")
            return
        
        csv_path = self.data_dir / "tp1p2_tuning_efficiency.csv"
        efficiency_df.to_csv(csv_path, index=False)
        print(f"✅ Tuning efficiency data exported to: {csv_path}")
        
        # Also save as Excel with multiple sheets
        excel_path = self.data_dir / "tp1p2_tuning_efficiency.xlsx"
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
        """Export all combined data (scan and liv) as an xarray DataArray and save as a .nc file with TP1-2 attributes."""
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
        data_xr.attrs["test_point"] = "TP1-2"
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
            data_xr.attrs["tile_metadata_sample"] = {}
        
        # Save as NetCDF
        nc_path = self.data_dir / "tp1p2_combined_data.nc"
        data_xr.to_netcdf(nc_path, engine='netcdf4')
        print(f"✅ Exported combined data to NetCDF: {nc_path}")

    def plot_threshold_current_vs_tile_combined(self):
        """Create a combined threshold current plot showing threshold current vs Tile SN with scatter plots and box plots for each bank."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        n_tiles = len(sorted_tiles)
        n_channels = 8
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            # For each channel, plot all tiles
            for ch in range(n_channels):
                ch_threshold = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        # Calculate threshold current (intercept of linear fit)
                        x = tile_data['Set Laser(mA)'].values
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
                    ax.scatter(ch_x, ch_threshold, color=channel_colors[ch], s=60, alpha=0.7, label=f'Ch{ch}', marker='o', edgecolor='black', linewidth=0.5)
            
            # Box plot for each tile (all channels)
            box_data = []
            for tile_sn in sorted_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_thresholds = []
                
                for ch in range(n_channels):
                    channel_data = tile_data[tile_data['Channel'] == ch]
                    if len(channel_data) > 0:
                        x = channel_data['Set Laser(mA)'].values
                        y = channel_data['Power(mW)'].values
                        
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            threshold_current = -intercept / slope if slope != 0 else np.nan
                            if not np.isnan(threshold_current) and threshold_current > 0:
                                tile_thresholds.append(threshold_current)
                
                if tile_thresholds:
                    box_data.append(tile_thresholds)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
                patch.set_linewidth(0.5)
            
            # Annotate average threshold current for each tile
            for i, tile_sn in enumerate(sorted_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_thresholds = []
                
                for ch in range(n_channels):
                    channel_data = tile_data[tile_data['Channel'] == ch]
                    if len(channel_data) > 0:
                        x = channel_data['Set Laser(mA)'].values
                        y = channel_data['Power(mW)'].values
                        
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            threshold_current = -intercept / slope if slope != 0 else np.nan
                            if not np.isnan(threshold_current) and threshold_current > 0:
                                tile_thresholds.append(threshold_current)
                
                if tile_thresholds:
                    avg_threshold = np.mean(tile_thresholds)
                    ax.text(i, 30, f'avg={avg_threshold:.1f}mA', ha='center', va='bottom', fontsize=7, color='red', rotation=90, fontweight='bold')
            
            ax.set_title(f'Threshold Current vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            if bank == 0:
                ax.set_ylabel('Threshold Current (mA)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(sorted_tiles, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(10, 40)  # Updated y-axis range from 10 to 40 mA
            ax.grid(True, linestyle='--', alpha=0.3)
            # Legend for each subplot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title='Channel')
        
        plt.suptitle('Threshold Current vs Tile SN at 150mA Laser Current', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p2_threshold_current_vs_tiles_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Threshold Current vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_threshold_current_vs_tile_html()

    def plot_slope_efficiency_vs_tile_combined(self):
        """Create a combined slope efficiency plot showing slope efficiency vs Tile SN with scatter plots and box plots for each bank."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        n_tiles = len(sorted_tiles)
        n_channels = 8
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            # For each channel, plot all tiles
            for ch in range(n_channels):
                ch_slope_eff = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        # Calculate slope efficiency at 1.1x threshold current
                        x = tile_data['Set Laser(mA)'].values
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
                                # Calculate slope efficiency at 1.1x threshold current
                                target_current = 1.1 * threshold_current
                                
                                # Find data points around 1.1x threshold current
                                current_mask = (x_clean >= target_current * 0.9) & (x_clean <= target_current * 1.1)
                                if np.sum(current_mask) > 1:
                                    x_target = x_clean[current_mask]
                                    y_target = y_clean[current_mask]
                                    
                                    # Calculate slope efficiency (dP/dI) in mW/mA
                                    slope_eff = np.polyfit(x_target, y_target, 1)[0]
                                    ch_slope_eff.append(slope_eff)
                                    ch_x.append(i)
                
                if ch_x:
                    ax.scatter(ch_x, ch_slope_eff, color=channel_colors[ch], s=60, alpha=0.7, label=f'Ch{ch}', marker='o', edgecolor='black', linewidth=0.5)
            
            # Box plot for each tile (all channels)
            box_data = []
            for tile_sn in sorted_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_slope_effs = []
                
                for ch in range(n_channels):
                    channel_data = tile_data[tile_data['Channel'] == ch]
                    if len(channel_data) > 0:
                        x = channel_data['Set Laser(mA)'].values
                        y = channel_data['Power(mW)'].values
                        
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
                                    tile_slope_effs.append(slope_eff)
                
                if tile_slope_effs:
                    box_data.append(tile_slope_effs)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
                patch.set_linewidth(0.5)
            
            # Annotate average slope efficiency for each tile
            for i, tile_sn in enumerate(sorted_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_slope_effs = []
                
                for ch in range(n_channels):
                    channel_data = tile_data[tile_data['Channel'] == ch]
                    if len(channel_data) > 0:
                        x = channel_data['Set Laser(mA)'].values
                        y = channel_data['Power(mW)'].values
                        
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
                                    tile_slope_effs.append(slope_eff)
                
                if tile_slope_effs:
                    avg_slope_eff = np.mean(tile_slope_effs)
                    ax.text(i, 0.1, f'avg={avg_slope_eff:.3f}mW/mA', ha='center', va='bottom', fontsize=7, color='red', rotation=90, fontweight='bold')
            
            ax.set_title(f'Slope Efficiency vs Tile SN - Bank {bank} (at 1.1x I_th)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            if bank == 0:
                ax.set_ylabel('Slope Efficiency (mW/mA)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(sorted_tiles, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 0.2)  # Updated y-axis range from 0 to 0.2 mW/mA
            ax.grid(True, linestyle='--', alpha=0.3)
            # Legend for each subplot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title='Channel')
        
        plt.suptitle('Slope Efficiency vs Tile SN at 1.1x Threshold Current', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p2_slope_efficiency_vs_tiles_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Slope Efficiency vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_slope_efficiency_vs_tile_html()

    def create_threshold_current_vs_tile_html(self):
        """Create interactive HTML plot for threshold current vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        print("Creating interactive HTML threshold current vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - Threshold Current vs Tile SN', 'Bank 1 - Threshold Current vs Tile SN'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_threshold = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        x = tile_data['Set Laser(mA)'].values
                        y = tile_data['Power(mW)'].values
                        
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) > 1:
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
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
                text='Threshold Current vs Tile SN at 150mA Laser Current',
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
        fig.update_yaxes(title_text="Threshold Current (mA)", row=1, col=1, range=[10, 40])
        fig.update_yaxes(title_text="Threshold Current (mA)", row=1, col=2, range=[10, 40])
        
        # Save HTML file
        html_filename = "tp1p2_threshold_current_vs_tiles_combined.html"
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML threshold current vs tile plot saved: {html_filename}")
        
        return fig

    def create_slope_efficiency_vs_tile_html(self):
        """Create interactive HTML plot for slope efficiency vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        print("Creating interactive HTML slope efficiency vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - Slope Efficiency vs Tile SN', 'Bank 1 - Slope Efficiency vs Tile SN'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_slope_eff = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    if len(tile_data) > 0:
                        x = tile_data['Set Laser(mA)'].values
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
                                        '<b>Slope Efficiency:</b> %{y:.4f} mW/mA<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(ch) + '<br>' +
                                        '<b>At:</b> 1.1x I_th<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='Slope Efficiency vs Tile SN at 1.1x Threshold Current',
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
        fig.update_yaxes(title_text="Slope Efficiency (mW/mA)", row=1, col=1, range=[0, 0.2])
        fig.update_yaxes(title_text="Slope Efficiency (mW/mA)", row=1, col=2, range=[0, 0.2])
        
        # Save HTML file
        html_filename = "tp1p2_slope_efficiency_vs_tiles_combined.html"
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML slope efficiency vs tile plot saved: {html_filename}")
        
        return fig

    def plot_power_vs_tile_combined(self):
        """Create a combined power plot showing Power vs Tile SN with scatter plots and box plots for each bank, similar to MPD plot."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        n_tiles = len(sorted_tiles)
        n_channels = 8
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            # For each channel, plot all tiles
            for ch in range(n_channels):
                ch_power = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                    if len(tile_150ma) > 0:
                        ch_power.append(tile_150ma['Power(mW)'].mean())
                        ch_x.append(i)
                
                if ch_x:
                    ax.scatter(ch_x, ch_power, color=channel_colors[ch], s=60, alpha=0.7, label=f'Ch{ch}', marker='o', edgecolor='black', linewidth=0.5)
            
            # Box plot for each tile (all channels)
            box_data = []
            for tile_sn in sorted_tiles:
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                if len(tile_150ma) > 0:
                    box_data.append(tile_150ma['Power(mW)'].values)
                else:
                    box_data.append([])
            
            bp = ax.boxplot(box_data, positions=range(n_tiles), patch_artist=True, showfliers=False, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
                patch.set_linewidth(0.5)
            
            # Annotate average power for each tile
            for i, tile_sn in enumerate(sorted_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
                if len(tile_150ma) > 0:
                    avg_power = tile_150ma['Power(mW)'].mean()
                    ax.text(i, 40, f'avg={avg_power:.1f}mW', ha='center', va='bottom', fontsize=7, color='red', rotation=90, fontweight='bold')
            
            ax.set_title(f'Power vs Tile SN - Bank {bank} (at 150mA)', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            if bank == 0:
                ax.set_ylabel('Power (mW)', fontsize=12)
            ax.set_xticks(range(n_tiles))
            ax.set_xticklabels(sorted_tiles, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(20, 50)  # Updated y-axis range from 20 to 50 mW
            ax.grid(True, linestyle='--', alpha=0.3)
            # Legend for each subplot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, title='Channel')
        
        plt.suptitle('Power vs Tile SN at 150mA Laser Current', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = "tp1p2_power_vs_tile_combined.png"
        plt.savefig(plots_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Power vs Tile Combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_power_vs_tile_html()

    def create_power_vs_tile_html(self):
        """Create interactive HTML plot for power vs tile."""
        if self.liv_data is None:
            print("LIV data not loaded!")
            return
        
        print("Creating interactive HTML power vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - Power vs Tile SN at 150mA', 'Bank 1 - Power vs Tile SN at 150mA'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        sorted_tiles = sorted(unique_tiles)
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = self.liv_data[self.liv_data['Bank'] == bank]
            
            for ch in range(8):
                ch_power = []
                ch_x = []
                for i, tile_sn in enumerate(sorted_tiles):
                    tile_data = bank_data[(bank_data['Tile_SN'] == tile_sn) & (bank_data['Channel'] == ch)]
                    tile_150ma = tile_data[(tile_data['Set Laser(mA)'] >= 145) & (tile_data['Set Laser(mA)'] <= 155)]
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
                                        '<b>Laser Current:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='Power vs Tile SN at 150mA Laser Current',
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
        html_filename = "tp1p2_power_vs_tile_combined.html"
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.write_html(plots_dir / html_filename)
        print(f"✅ Interactive HTML power vs tile plot saved: {html_filename}")
        
        return fig

    def run_all(self):
        print("🔄 Loading TP1-2 Scan data files...")
        self.load_scan_files()
        print("🔄 Loading TP1-2 LIV data files...")
        self.load_liv_files()
        print("🔄 Combining TP1-2 Scan data...")
        self.load_scan_data()
        print("🔄 Combining TP1-2 LIV data...")
        self.load_liv_data()
        
        print("\n" + "="*60)
        print("TP1-2 LIV PLOTS PER TILE")
        print("="*60)
        self.plot_liv_per_tile()

        print("\n" + "="*60)
        print("TP1-2 SCAN PLOTS PER TILE")
        print("="*60)
        self.plot_scan_per_tile()
        
        print("\n" + "="*60)
        print("TP1-2 MPD PLOTS PER TILE")
        print("="*60)
        self.plot_mpd_per_tile()
        
        print("\n" + "="*60)
        print("TP1-2 MPD VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-2 POWER VS TILE COMBINED")
        print("="*60)
        self.plot_power_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-2 THRESHOLD CURRENT VS TILE COMBINED")
        print("="*60)
        self.plot_threshold_current_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-2 SLOPE EFFICIENCY VS TILE COMBINED")
        print("="*60)
        self.plot_slope_efficiency_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP1-2 TUNING EFFICIENCY ANALYSIS")
        print("="*60)
        efficiency_df = self.calculate_tuning_efficiency()
        if efficiency_df is not None:
            self.plot_tuning_efficiency(efficiency_df)
            self.export_tuning_efficiency(efficiency_df)
        
        # Export to xarray/netcdf
        print("\n" + "="*60)
        print("TP1-2 EXPORT TO NETCDF")
        print("="*60)
        self.export_to_xarray()
        
        print("\n" + "="*80)
        print("TP1-2 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"✅ PNG plots saved to: {self.output_dir.absolute()}")
        
        # List generated plots
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique() if self.scan_data is not None else []
        print("📋 Generated plots:")
        for tile_sn in unique_tiles:
            print(f"   • LIV_{tile_sn}.png")
            print(f"   • Scan_{tile_sn}.png")
            print(f"   • MPD_{tile_sn}.png")
        print(f"   • tp1p2_tuning_efficiency.png (in plots folder)")
        print(f"   • tp1p2_tuning_efficiency.html (in plots folder)")
        print(f"   • tp1p2_mpd_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p2_mpd_vs_tile_combined.html (in plots folder)")
        print(f"   • tp1p2_power_vs_tile_combined.png (in plots folder)")
        print(f"   • tp1p2_power_vs_tile_combined.html (in plots folder)")
        print(f"   • tp1p2_threshold_current_vs_tiles_combined.png (in plots folder)")
        print(f"   • tp1p2_slope_efficiency_vs_tiles_combined.png (in plots folder)")
        print(f"   • tp1p2_threshold_current_vs_tiles_combined.html (in plots folder)")
        print(f"   • tp1p2_slope_efficiency_vs_tiles_combined.html (in plots folder)")
        
        print(f"   • tp1p2_combined_data.nc (in data folder)")
        print(f"   • tp1p2_tuning_efficiency.csv (in data folder)")
        print(f"   • tp1p2_tuning_efficiency.xlsx (in data folder)")
        print("\n📊 Metadata Summary:")
        print(f"   • Analyzed {len(self.tile_metadata)} tiles")
        print(f"   • All measurements performed at 150mA laser current")
        print(f"   • Scan data points: {len(self.scan_data) if self.scan_data is not None else 0}")
        print(f"   • LIV data points: {len(self.liv_data) if self.liv_data is not None else 0}")
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

def main():
    print("=" * 80)
    print("TP1-2 LASER MODULE DATA ANALYSIS (PER-TILE PLOTS)")
    print("=" * 80)
    print("Test Point: TP1-2")
    print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    analyzer = TP1P2CombinedAnalyzer()
    analyzer.run_all()

if __name__ == "__main__":
    main() 