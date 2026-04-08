#!/usr/bin/env python3
"""
TP3-1 Combined Scan and LIV Data Analysis Script
================================================

This script analyzes both Laser and VOA data from TP3-1 test point.
It creates per-tile plots showing Laser data (PeakWave vs Set Laser) and VOA data (Power vs Set VOA)
for each temperature, with separate subplots for Bank 0 and Bank 1.
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

class TP3P1CombinedAnalyzer:
    def __init__(self, scan_data_path=None, liv_data_path=None):
        script_dir = Path(__file__).parent
        self.scan_data_path = Path(scan_data_path) if scan_data_path else script_dir / "../TP3-1"
        self.liv_data_path = Path(liv_data_path) if liv_data_path else script_dir / "../TP3-1"
        self.output_dir = script_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create TP3-1 subdirectory for tile plots
        self.tp31_output_dir = self.output_dir / "TP3-1"
        self.tp31_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.scan_files = []
        self.liv_files = []
        self.scan_data = None
        self.liv_data = None
        self.tile_metadata = {}  # Store metadata for each tile

    def extract_serial_number(self, filename):
        # Handle both TP3-1 Laser and VOA files
        match = re.search(r'-Y(\d+)-TP3-1 (Laser|VOA)\.csv$', filename)
        if match:
            return f"Y{match.group(1)}"
        return None

    def load_scan_files(self):
        # TP3-1 uses "Laser.csv" for scan data
        self.scan_files = sorted(glob.glob(str(self.scan_data_path / "* Laser.csv")))
        print(f"Looking for Laser files in: {self.scan_data_path.absolute()}")
        print(f"Search pattern: {self.scan_data_path / '* Laser.csv'}")
        print(f"Found {len(self.scan_files)} Laser CSV files")
        if len(self.scan_files) == 0:
            print(f"⚠️  No Laser files found. Please check if the TP3-1 directory exists and contains files matching pattern '*-Y*-TP3-1 Laser.csv'")
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
        # TP3-1 uses "VOA.csv" for LIV data
        self.liv_files = sorted(glob.glob(str(self.liv_data_path / "* VOA.csv")))
        
        print(f"Looking for VOA files in: {self.liv_data_path.absolute()}")
        print(f"Search pattern: {self.liv_data_path / '* VOA.csv'}")
        print(f"Found {len(self.liv_files)} VOA CSV files")
        if len(self.liv_files) == 0:
            print(f"⚠️  No VOA files found. Please check if the TP3-1 directory exists and contains files matching pattern '*-Y*-TP3-1 VOA.csv'")
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
                
                # TP3-1 Laser data format: each row has data for all 8 channels
                # Convert from wide to long format
                if sn_from_filename:
                    processed_rows = []
                    
                    for _, row in df.iterrows():
                        bank = row['Bank']
                        
                        # Process each channel (0-7)
                        for channel in range(8):
                            processed_row = {
                                'Tile_SN': sn_from_filename,
                                'Bank': bank,
                                'Channel': channel,
                                'Set Temp(C)': row['Set PIC Temp(C)'] if 'Set PIC Temp(C)' in df.columns else 42,
                                'Set Laser(mA)': row['Set Laser(mA)'],
                                'T_PIC(C)': row['T_PIC(C)'],
                                'Power(mW)': row['Power(mW)'],
                                'MPD_PIC(uA)': row[f'MPD_PIC_{channel}(uA)'] if f'MPD_PIC_{channel}(uA)' in df.columns else None,
                                'MPD_MUX(uA)': row[f'MPD_MUX_{channel}(uA)'] if f'MPD_MUX_{channel}(uA)' in df.columns else None,
                                'PeakWave(nm)': row[f'PeakWave_{channel}(nm)'] if f'PeakWave_{channel}(nm)' in df.columns else None,
                                'PeakPower(dBm)': row[f'PeakPower_{channel}(dBm)'] if f'PeakPower_{channel}(dBm)' in df.columns else None,
                                'Time': row['Time'],
                                'filename': filename
                            }
                            processed_rows.append(processed_row)
                    
                    if processed_rows:
                        unified_df = pd.DataFrame(processed_rows)
                        dfs.append(unified_df)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        if dfs:
            self.scan_data = pd.concat(dfs, ignore_index=True)
            # Convert '-' to NaN in numeric columns
            numeric_cols = ['Set Temp(C)', 'Set Laser(mA)', 'T_PIC(C)', 'Power(mW)', 'MPD_PIC(uA)', 'MPD_MUX(uA)', 'PeakWave(nm)', 'PeakPower(dBm)']
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
            
            # TP3-1 VOA data format: each row has data for all 8 channels
            # Convert from wide to long format
            if sn_from_filename:
                processed_rows = []
                
                for _, row in df.iterrows():
                    bank = row['Bank']
                    
                    # Process each channel (0-7)
                    for channel in range(8):
                        processed_row = {
                            'Tile_SN': sn_from_filename,
                            'Bank': bank,
                            'Channel': channel,
                            'Set Temp(C)': row['Set PIC Temp(C)'] if 'Set PIC Temp(C)' in df.columns else 42,
                            'Set VOA(mA)': row['Set VOA(mA)'],
                            'T_PIC(C)': row['T_PIC(C)'],
                            'Power(mW)': row['Power(mW)'],
                            'MPD_PIC(uA)': row[f'MPD_PIC_{channel}(uA)'] if f'MPD_PIC_{channel}(uA)' in df.columns else None,
                            'MPD_MUX(uA)': row[f'MPD_MUX_{channel}(uA)'] if f'MPD_MUX_{channel}(uA)' in df.columns else None,
                            'PeakWave(nm)': row[f'PeakWave_{channel}(nm)'] if f'PeakWave_{channel}(nm)' in df.columns else None,
                            'PeakPower(dBm)': row[f'PeakPower_{channel}(dBm)'] if f'PeakPower_{channel}(dBm)' in df.columns else None,
                            'Time': row['Time'],
                            'filename': filename
                        }
                        processed_rows.append(processed_row)
                
                if processed_rows:
                    unified_df = pd.DataFrame(processed_rows)
                    dfs.append(unified_df)
                    
        if dfs:
            self.liv_data = pd.concat(dfs, ignore_index=True)
            # Convert '-' to NaN in numeric columns
            numeric_cols = ['Set Temp(C)', 'Set VOA(mA)', 'T_PIC(C)', 'Power(mW)', 'MPD_PIC(uA)', 'MPD_MUX(uA)', 'PeakWave(nm)', 'PeakPower(dBm)']
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

    def get_current_column_name(self, data_type='scan'):
        """Get the correct column name for laser/VOA current based on data type."""
        if data_type == 'liv' and self.liv_data is not None and 'Set VOA(mA)' in self.liv_data.columns:
            return 'Set VOA(mA)'
        else:
            return 'Set Laser(mA)'

    def run_all(self):
        print("🔄 Loading TP3-1 Laser (scan) data files...")
        self.load_scan_files()
        print("🔄 Loading TP3-1 VOA (LIV) data files...")
        self.load_liv_files()
        print("🔄 Combining TP3-1 Laser (scan) data...")
        self.load_scan_data()
        print("🔄 Combining TP3-1 VOA (LIV) data...")
        self.load_liv_data()
        
        print("\n" + "="*60)
        print("TP3-1 WAVELENGTH SLOPE VS VOA CURRENT")
        print("="*60)
        self.plot_wavelength_voa_slope_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP3-1 VOA POWER LOSS (0mA to 10mA)")
        print("="*60)
        self.plot_voa_loss_10mA_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP3-1 VOA POWER LOSS (0mA to 20mA)")
        print("="*60)
        self.plot_voa_loss_20mA_vs_tile_combined()

        print("\n" + "="*60)
        print("TP3-1 POWER VS VOA PLOTS PER TILE")
        print("="*60)
        self.plot_voa_per_tile()

        print("\n" + "="*60)
        print("TP3-1 WAVELENGTH VS VOA PLOTS PER TILE")
        print("="*60)
        self.plot_wavelength_vs_voa_per_tile()

        print("\n" + "="*60)
        print("TP3-1 SCAN PLOTS PER TILE")
        print("="*60)
        self.plot_scan_per_tile()
        
        print("\n" + "="*60)
        print("TP3-1 MPD PLOTS PER TILE")
        print("="*60)
        self.plot_mpd_per_tile()
        
        print("\n" + "="*60)
        print("TP3-1 MPD RESPONSIVITY PER TILE")
        print("="*60)
        self.plot_mpd_responsivity_per_tile()
        
        print("\n" + "="*60)
        print("TP3-1 MPD_MUX RESPONSIVITY PER TILE")
        print("="*60)
        self.plot_mpd_mux_responsivity_per_tile()
        
        print("\n" + "="*60)
        print("TP3-1 MPD RESPONSIVITY VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_responsivity_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP3-1 MPD_MUX RESPONSIVITY VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_mux_responsivity_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP3-1 MPD VS TILE COMBINED")
        print("="*60)
        self.plot_mpd_vs_tile_combined()
        
        print("\n" + "="*60)
        print("TP3-1 TUNING EFFICIENCY")
        print("="*60)
        efficiency_df = self.calculate_tuning_efficiency()
        if efficiency_df is not None:
            self.plot_tuning_efficiency(efficiency_df)
            self.export_tuning_efficiency(efficiency_df)
        
        # Export to xarray/netcdf
        print("\n" + "="*60)
        print("TP3-1 EXPORT TO NETCDF")
        print("="*60)
        self.export_to_xarray()
        
        print("\n" + "="*80)
        print("TP3-1 ANALYSIS COMPLETE!")
        print("="*80)
        print(f"✅ PNG plots saved to: {self.output_dir.absolute()}")
        
        # List generated plots
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique() if self.scan_data is not None else []
        print("📋 Generated plots:")
        
        # VOA analysis plots
        print(f"   • tp3p1_wavelength_voa_slope_vs_tile_combined.png (in plots folder)")
        print(f"   • tp3p1_voa_loss_10mA_vs_tile_combined.png (in plots folder)")
        print(f"   • tp3p1_voa_loss_20mA_vs_tile_combined.png (in plots folder)")
        
        for tile_sn in unique_tiles:
            print(f"   • VOA_{tile_sn}.png (Power vs VOA) (in TP3-1 folder)")
            print(f"   • Wavelength_vs_VOA_{tile_sn}.png (in TP3-1 folder)")
            print(f"   • Scan_{tile_sn}.png (in TP3-1 folder)")
            print(f"   • MPD_{tile_sn}.png (in TP3-1 folder)")
            print(f"   • MPD_Responsivity_{tile_sn}.png (in TP3-1 folder)")
            print(f"   • MPD_MUX_Responsivity_{tile_sn}.png (in TP3-1 folder)")
            print(f"   • MPD_Responsivity_Summary_{tile_sn}.csv (in TP3-1 folder)")
        print(f"   • MPD_Responsivity_Summary_All_Tiles.png (in TP3-1 folder)")
        print(f"   • MPD_Responsivity_Summary_All_Tiles.csv (in TP3-1 folder)")
        print(f"   • tp3p1_mpd_responsivity_vs_tile_combined.png (in plots folder)")
        print(f"   • tp3p1_mpd_responsivity_vs_tile_combined.html (in plots folder)")
        print(f"   • tp3p1_mpd_mux_responsivity_vs_tile_combined.png (in plots folder)")
        print(f"   • tp3p1_mpd_mux_responsivity_vs_tile_combined.html (in plots folder)")
        print(f"   • tp3p1_tuning_efficiency.png (in plots folder)")
        print(f"   • tp3p1_tuning_efficiency.html (in plots folder)")
        print(f"   • tp3p1_mpd_vs_tile_combined.png (in plots folder)")
        print(f"   • tp3p1_mpd_vs_tile_combined.html (in plots folder)")
        
        print(f"   • tp3p1_combined_data.nc (in data folder)")
        print(f"   • tp3p1_tuning_efficiency.csv (in data folder)")
        print(f"   • tp3p1_tuning_efficiency.xlsx (in data folder)")
        
        # VOA analysis data exports
        print(f"   • tp3p1_wavelength_voa_slope_data.csv (in data folder)")
        print(f"   • tp3p1_voa_loss_10mA_data.csv (in data folder)")
        print(f"   • tp3p1_voa_loss_20mA_data.csv (in data folder)")
        
        print("\n📊 Metadata Summary:")
        print(f"   • Analyzed {len(self.tile_metadata)} tiles")
        print(f"   • Scan data points: {len(self.scan_data) if self.scan_data is not None else 0}")
        print(f"   • VOA data points: {len(self.liv_data) if self.liv_data is not None else 0}")
        if hasattr(self, 'efficiency_df') and self.efficiency_df is not None:
            print(f"   • Tuning efficiency measurements: {len(self.efficiency_df)}")
        print("\n🔍 Available Tile Metadata:")
        if self.tile_metadata:
            sample_tile = list(self.tile_metadata.keys())[0]
            sample_meta = self.tile_metadata[sample_tile]
            for key, value in sample_meta.items():
                if key != 'filename':
                    print(f"   • {key}: {value}")
        print("="*80)

    def plot_scan_per_tile(self):
        """Create scan plots for each tile showing PeakWave vs Set Laser for each temperature."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for tile_sn in unique_tiles:
            print(f"Creating scan plot for tile {tile_sn}...")
            
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
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
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
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ MPD plot saved: {plot_filename}")

    def plot_voa_per_tile(self):
        """Create VOA plots for each tile showing Power vs Set VOA."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for tile_sn in unique_tiles:
            print(f"Creating VOA plot for tile {tile_sn}...")
            
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
                                grouped_data = channel_data.groupby('Set VOA(mA)')['PeakPower(dBm)'].mean().reset_index()
                                if len(grouped_data) > 0:
                                    print(f"        Channel {channel}: {len(grouped_data)} points, Power range: {grouped_data['PeakPower(dBm)'].min():.2f} to {grouped_data['PeakPower(dBm)'].max():.2f}")
                                    
                                    # Use different colors for different channels
                                    color_idx = channel_idx % len(colors)
                                    ax.plot(grouped_data['Set VOA(mA)'], grouped_data['PeakPower(dBm)'], 
                                           marker='o', linewidth=1.5, markersize=4, 
                                           color=colors[color_idx], 
                                           label=f'{temp:.1f}°C Ch{channel}')
                                    plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set VOA (mA)')
                ax.set_ylabel('Peak Power (dBm)')
                ax.set_xlim(0, 25)
                ax.set_ylim(-20, 10)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'VOA - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"VOA_{tile_sn}.png"
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ VOA plot saved: {plot_filename}")

    def plot_wavelength_vs_voa_per_tile(self):
        """Create wavelength vs VOA plots for each tile showing PeakWave vs Set VOA."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        unique_tiles = self.liv_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for tile_sn in unique_tiles:
            print(f"Creating wavelength vs VOA plot for tile {tile_sn}...")
            
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
                                grouped_data = channel_data.groupby('Set VOA(mA)')['PeakWave(nm)'].mean().reset_index()
                                if len(grouped_data) > 0:
                                    print(f"        Channel {channel}: {len(grouped_data)} points, Wavelength range: {grouped_data['PeakWave(nm)'].min():.2f} to {grouped_data['PeakWave(nm)'].max():.2f}")
                                    
                                    # Use different colors for different channels
                                    color_idx = channel_idx % len(colors)
                                    ax.plot(grouped_data['Set VOA(mA)'], grouped_data['PeakWave(nm)'], 
                                           marker='o', linewidth=1.5, markersize=4, 
                                           color=colors[color_idx], 
                                           label=f'{temp:.1f}°C Ch{channel}')
                                    plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set VOA (mA)')
                ax.set_ylabel('Peak Wavelength (nm)')
                ax.set_xlim(0, 25)
                ax.set_ylim(1300, 1320)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'Wavelength vs VOA - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"Wavelength_vs_VOA_{tile_sn}.png"
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Wavelength vs VOA plot saved: {plot_filename}")

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
                    
                    # Get power at 0mA and 10mA using PeakPower(dBm)
                    power_0mA = channel_data[channel_data['Set VOA(mA)'] == 0]['PeakPower(dBm)'].values
                    power_10mA = channel_data[channel_data['Set VOA(mA)'] == 10]['PeakPower(dBm)'].values
                    
                    if len(power_0mA) > 0 and len(power_10mA) > 0:
                        # Calculate power loss in dB (already in dBm, so direct subtraction)
                        loss_dB = power_0mA[0] - power_10mA[0]  # Loss = P_0mA - P_10mA
                        
                        loss_data.append({
                            'Tile_SN': tile_sn,
                            'Bank': bank,
                            'Channel': channel,
                            'Power_0mA_dBm': power_0mA[0],
                            'Power_10mA_dBm': power_10mA[0],
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
        plot_filename = "tp3p1_voa_loss_10mA_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ VOA power loss (10mA) vs tile plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp3p1_voa_loss_10mA_data.csv"
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
                    
                    # Get power at 0mA and 20mA using PeakPower(dBm)
                    power_0mA = channel_data[channel_data['Set VOA(mA)'] == 0]['PeakPower(dBm)'].values
                    power_20mA = channel_data[channel_data['Set VOA(mA)'] == 20]['PeakPower(dBm)'].values
                    
                    if len(power_0mA) > 0 and len(power_20mA) > 0:
                        # Calculate power loss in dB (already in dBm, so direct subtraction)
                        loss_dB = power_0mA[0] - power_20mA[0]  # Loss = P_0mA - P_20mA
                        
                        loss_data.append({
                            'Tile_SN': tile_sn,
                            'Bank': bank,
                            'Channel': channel,
                            'Power_0mA_dBm': power_0mA[0],
                            'Power_20mA_dBm': power_20mA[0],
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
                    ax.text(i, 8, f'avg={avg_loss:.1f}dB', 
                           ha='center', va='bottom', fontsize=8, color='red', 
                           rotation=90, fontweight='bold')
            
            ax.set_title(f'VOA Power Loss (0mA to 20mA) - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Power Loss (dB)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=9)
            ax.set_ylim(3, 10)
            ax.grid(True, linestyle='--', alpha=0.3)
            if bank == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.suptitle('VOA Power Loss (0mA to 20mA) by Tile', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp3p1_voa_loss_20mA_vs_tile_combined.png" 
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ VOA power loss (20mA) vs tile plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp3p1_voa_loss_20mA_data.csv"
        loss_df.to_csv(csv_path, index=False)
        print(f"✅ VOA power loss (20mA) data exported to: {csv_path}")

    def plot_wavelength_voa_slope_vs_tile_combined(self):
        """Create a combined plot showing wavelength slope vs VOA current for each tile."""
        if self.liv_data is None:
            print("VOA data not loaded!")
            return
        
        # Calculate wavelength vs VOA slope for each tile/bank/channel combination
        slope_data = []
        
        unique_tiles = sorted(self.liv_data['Tile_SN'].dropna().unique())
        for tile_sn in unique_tiles:
            tile_data = self.liv_data[self.liv_data['Tile_SN'] == tile_sn]
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    
                    if len(channel_data) > 1:  # Need at least 2 points for slope
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
                            
                            # Calculate R-squared for quality assessment
                            y_pred = slope * x_clean + intercept
                            ss_res = np.sum((y_clean - y_pred) ** 2)
                            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            
                            slope_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Slope_nm_mA': slope,
                                'Intercept_nm': intercept,
                                'R_squared': r_squared,
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
                    y_values = channel_data['Slope_nm_mA'].values
                    
                    ax.scatter(x_positions, y_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50,
                              label=f'Ch{channel}')
            
            # Box plot
            box_data = []
            for tile_sn in unique_tiles:
                tile_slopes = bank_data[bank_data['Tile_SN'] == tile_sn]['Slope_nm_mA'].values
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
                tile_slopes = bank_data[bank_data['Tile_SN'] == tile_sn]['Slope_nm_mA'].values
                if len(tile_slopes) > 0:
                    avg_slope = np.mean(tile_slopes)
                    ax.text(i, -0.002, f'avg={avg_slope:.5f}', 
                           ha='center', va='bottom', fontsize=8, color='red', 
                           rotation=90, fontweight='bold')
            
            ax.set_title(f'Wavelength Slope vs VOA Current - Bank {bank}', fontsize=14)
            ax.set_xlabel('Tile SN (ordered by date)', fontsize=12)
            ax.set_ylabel('Wavelength Slope (nm/mA)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, fontsize=9)
            ax.set_ylim(-0.003, 0.001)
            ax.grid(True, linestyle='--', alpha=0.3)
            if bank == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.suptitle('Wavelength Slope vs VOA Current by Tile', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp3p1_wavelength_voa_slope_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Wavelength vs VOA slope plot saved: {plot_filename}")
        
        # Export data
        csv_path = self.data_dir / "tp3p1_wavelength_voa_slope_data.csv"
        slope_df.to_csv(csv_path, index=False)
        print(f"✅ Wavelength vs VOA slope data exported to: {csv_path}")

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
                tile_bank_data = tile_data[(tile_data['Bank'] == bank)]
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
                                grouped_data = channel_data.groupby('Set Laser(mA)')['MPD_Responsivity(uA/mW)'].mean().reset_index()
                                print(f"        Channel {channel}: {len(grouped_data)} points, Responsivity range: {grouped_data['MPD_Responsivity(uA/mW)'].min():.3f} to {grouped_data['MPD_Responsivity(uA/mW)'].max():.3f}")
                                
                                # Use different colors for different channels
                                color_idx = channel_idx % len(colors)
                                ax.plot(grouped_data['Set Laser(mA)'], grouped_data['MPD_Responsivity(uA/mW)'], 
                                       marker='o', linewidth=1.5, markersize=4, 
                                       color=colors[color_idx], 
                                       label=f'{temp:.1f}°C Ch{channel}')
                                plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set Laser (mA)')
                ax.set_ylabel('MPD Responsivity (uA/mW)')
                ax.set_xlim(120, 170)
                ax.set_ylim(0, 50)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'MPD Responsivity - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"MPD_Responsivity_{tile_sn}.png"
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ MPD responsivity plot saved: {plot_filename}")

    def plot_mpd_responsivity_vs_tile_combined(self):
        """Create a combined MPD responsivity plot showing responsivity vs Tile SN with channel-based scatter plots and box plots for each bank.
        Similar to tp2p4_channel_power_vs_tile_combined pattern. MPD responsivity calculated from PeakPower(dBm) and Power(mW)."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        # Filter out invalid data and calculate responsivity
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
        
        # Create figure with 2 subplots (Bank 0 and Bank 1) - similar to tp2p4_channel_power pattern
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle('TP3-1 MPD Responsivity vs Tile Summary (Channel-based)', fontsize=16, fontweight='bold')
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            
            if len(bank_data) == 0:
                continue
            
            # Channel-based scatter plot for each channel (like tp2p4_channel_power pattern)
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [unique_tiles.index(tile) for tile in channel_data['Tile_SN']]
                    responsivity_values = channel_data['MPD_Responsivity(A/W)'].values
                    
                    ax.scatter(positions, responsivity_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50, 
                              label=f'Ch{channel}')
            
            # Box plot for overall distribution per tile (all channels combined)
            box_data = []
            box_positions = []
            for pos, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]['MPD_Responsivity(A/W)'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average responsivity for each tile
            for pos, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]['MPD_Responsivity(A/W)']
                if not tile_data.empty:
                    avg_responsivity = tile_data.mean()
                    ax.text(pos, 0.025, f'{avg_responsivity:.3f}', 
                           fontsize=8, color='red', fontweight='bold',
                           ha='center', va='bottom', rotation=90)
            
            ax.set_title(f'Bank {bank}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tile Serial Number', fontsize=12)
            ax.set_ylabel('MPD Responsivity (A/W)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 0.03)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp3p1_mpd_responsivity_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD responsivity vs tile combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_mpd_responsivity_vs_tile_html(data_150ma, unique_tiles)
        
        # Print summary statistics
        print("\n📊 MPD Responsivity Summary:")
        for bank in [0, 1]:
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            if not bank_data.empty:
                print(f"\nBank {bank}:")
                print(f"  Total measurements: {len(bank_data)}")
                print(f"  Mean responsivity: {bank_data['MPD_Responsivity(A/W)'].mean():.4f} A/W")
                print(f"  Std responsivity: {bank_data['MPD_Responsivity(A/W)'].std():.4f} A/W")
                print(f"  Min responsivity: {bank_data['MPD_Responsivity(A/W)'].min():.4f} A/W")
                print(f"  Max responsivity: {bank_data['MPD_Responsivity(A/W)'].max():.4f} A/W")

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
            
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_data['Tile_SN'],
                            y=channel_data['MPD_Responsivity(A/W)'],
                            mode='markers',
                            name=f'Bank {bank} - Channel {channel}',
                            marker=dict(color=channel_colors[channel], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>Responsivity:</b> %{y:.4f} A/W<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(channel) + '<br>' +
                                        '<b>Laser Current:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='MPD Responsivity vs Tile SN at 150mA Laser Current (Channel-based)',
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
        fig.update_yaxes(title_text="MPD Responsivity (A/W)", row=1, col=1, range=[0, 0.03])
        fig.update_yaxes(title_text="MPD Responsivity (A/W)", row=1, col=2, range=[0, 0.03])
        
        # Save HTML file
        html_filename = "tp3p1_mpd_responsivity_vs_tile_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML MPD responsivity vs tile plot saved: {html_filename}")
        
        return fig

    def plot_mpd_mux_responsivity_per_tile(self):
        """Create MPD_MUX responsivity plots for each tile showing MPD_MUX responsivity vs Set Laser for each temperature.
        MPD_MUX responsivity is defined as MPD_MUX(uA) / Power(mW)."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        unique_tiles = self.scan_data['Tile_SN'].dropna().unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for tile_sn in unique_tiles:
            print(f"Creating MPD_MUX responsivity plot for tile {tile_sn}...")
            
            # Get tile data and filter out invalid values
            tile_data = self.scan_data[self.scan_data['Tile_SN'] == tile_sn].copy()
            # Filter out rows where Power is 0 or NaN, or MPD_MUX is NaN, to avoid division by zero
            tile_data = tile_data[(tile_data['Power(mW)'] > 0) & (tile_data['MPD_MUX(uA)'].notna()) & (tile_data['Power(mW)'].notna())]
            
            if len(tile_data) == 0:
                print(f"  ❌ No valid MPD_MUX data for tile {tile_sn}")
                continue
                
            # Calculate MPD_MUX responsivity
            tile_data['MPD_MUX_Responsivity(uA/mW)'] = tile_data['MPD_MUX(uA)'] / tile_data['Power(mW)']
            
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
                tile_bank_data = tile_data[(tile_data['Bank'] == bank)]
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
                                grouped_data = channel_data.groupby('Set Laser(mA)')['MPD_MUX_Responsivity(uA/mW)'].mean().reset_index()
                                if len(grouped_data) > 0:
                                    print(f"        Channel {channel}: {len(grouped_data)} points, MUX Responsivity range: {grouped_data['MPD_MUX_Responsivity(uA/mW)'].min():.3f} to {grouped_data['MPD_MUX_Responsivity(uA/mW)'].max():.3f}")
                                    
                                    # Use different colors for different channels
                                    color_idx = channel_idx % len(colors)
                                    ax.plot(grouped_data['Set Laser(mA)'], grouped_data['MPD_MUX_Responsivity(uA/mW)'], 
                                           marker='o', linewidth=1.5, markersize=4, 
                                           color=colors[color_idx], 
                                           label=f'{temp:.1f}°C Ch{channel}')
                                    plotted = True
                
                if not plotted:
                    print(f"    ❌ No data plotted for bank {bank}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'Bank {bank}')
                ax.set_xlabel('Set Laser (mA)')
                ax.set_ylabel('MPD_MUX Responsivity (uA/mW)')
                ax.set_xlim(120, 170)
                ax.set_ylim(0, 50)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.suptitle(f'MPD_MUX Responsivity - Tile {tile_sn}', fontsize=16)
            plt.tight_layout()
            plot_filename = f"MPD_MUX_Responsivity_{tile_sn}.png"
            plt.savefig(self.tp31_output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ MPD_MUX responsivity plot saved: {plot_filename}")

    def plot_mpd_mux_responsivity_vs_tile_combined(self):
        """Create a combined MPD_MUX responsivity plot showing responsivity vs Tile SN with channel-based scatter plots and box plots for each bank.
        Similar to tp2p4_channel_power_vs_tile_combined pattern. MPD_MUX responsivity calculated from MPD_MUX(uA) and Power(mW)."""
        if self.scan_data is None:
            print("Scan data not loaded!")
            return
        
        # Filter out invalid data and calculate responsivity
        valid_data = self.scan_data[(self.scan_data['Power(mW)'] > 0) & 
                                   (self.scan_data['MPD_MUX(uA)'].notna()) & 
                                   (self.scan_data['Power(mW)'].notna())].copy()
        
        if len(valid_data) == 0:
            print("❌ No valid data for MPD_MUX responsivity calculation")
            return
        
        # Calculate MPD_MUX responsivity in A/W (convert from uA/mW)
        # uA/mW = 1e-6 A / 1e-3 W = 1e-3 A/W
        valid_data['MPD_MUX_Responsivity(A/W)'] = (valid_data['MPD_MUX(uA)'] / valid_data['Power(mW)']) * 1e-3
        
        # Get data at 150mA laser current
        data_150ma = valid_data[(valid_data['Set Laser(mA)'] >= 145) & (valid_data['Set Laser(mA)'] <= 155)]
        
        if len(data_150ma) == 0:
            print("❌ No MPD_MUX data at 150mA laser current")
            return
        
        unique_tiles = sorted(data_150ma['Tile_SN'].dropna().unique())
        n_tiles = len(unique_tiles)
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Create figure with 2 subplots (Bank 0 and Bank 1) - similar to tp2p4_channel_power pattern
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle('TP3-1 MPD_MUX Responsivity vs Tile Summary (Channel-based)', fontsize=16, fontweight='bold')
        
        for bank in [0, 1]:
            ax = axs[bank]
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            
            if len(bank_data) == 0:
                continue
            
            # Channel-based scatter plot for each channel (like tp2p4_channel_power pattern)
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [unique_tiles.index(tile) for tile in channel_data['Tile_SN']]
                    responsivity_values = channel_data['MPD_MUX_Responsivity(A/W)'].values
                    
                    ax.scatter(positions, responsivity_values, 
                              color=channel_colors[channel], 
                              alpha=0.7, s=50, 
                              label=f'Ch{channel}')
            
            # Box plot for overall distribution per tile (all channels combined)
            box_data = []
            box_positions = []
            for pos, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]['MPD_MUX_Responsivity(A/W)'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average responsivity for each tile
            for pos, tile_sn in enumerate(unique_tiles):
                tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]['MPD_MUX_Responsivity(A/W)']
                if not tile_data.empty:
                    avg_responsivity = tile_data.mean()
                    ax.text(pos, 0.025, f'{avg_responsivity:.3f}', 
                           fontsize=8, color='red', fontweight='bold',
                           ha='center', va='bottom', rotation=90)
            
            ax.set_title(f'Bank {bank}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tile Serial Number', fontsize=12)
            ax.set_ylabel('MPD_MUX Responsivity (A/W)', fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 0.03)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "tp3p1_mpd_mux_responsivity_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD_MUX responsivity vs tile combined plot saved: {plot_filename}")
        
        # Create HTML version
        self.create_mpd_mux_responsivity_vs_tile_html(data_150ma, unique_tiles)
        
        # Print summary statistics
        print("\n📊 MPD_MUX Responsivity Summary:")
        for bank in [0, 1]:
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            if not bank_data.empty:
                print(f"\nBank {bank}:")
                print(f"  Total measurements: {len(bank_data)}")
                print(f"  Mean responsivity: {bank_data['MPD_MUX_Responsivity(A/W)'].mean():.4f} A/W")
                print(f"  Std responsivity: {bank_data['MPD_MUX_Responsivity(A/W)'].std():.4f} A/W")
                print(f"  Min responsivity: {bank_data['MPD_MUX_Responsivity(A/W)'].min():.4f} A/W")
                print(f"  Max responsivity: {bank_data['MPD_MUX_Responsivity(A/W)'].max():.4f} A/W")

    def create_mpd_mux_responsivity_vs_tile_html(self, data_150ma, unique_tiles):
        """Create interactive HTML plot for MPD_MUX responsivity vs tile."""
        print("Creating interactive HTML MPD_MUX responsivity vs tile plot...")
        
        # Create subplots for Bank 0 and Bank 1
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bank 0 - MPD_MUX Responsivity vs Tile SN at 150mA', 'Bank 1 - MPD_MUX Responsivity vs Tile SN at 150mA'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for bank in [0, 1]:
            bank_data = data_150ma[data_150ma['Bank'] == bank]
            
            for channel in range(8):
                channel_data = bank_data[bank_data['Channel'] == channel]
                if len(channel_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=channel_data['Tile_SN'],
                            y=channel_data['MPD_MUX_Responsivity(A/W)'],
                            mode='markers',
                            name=f'Bank {bank} - Channel {channel}',
                            marker=dict(color=channel_colors[channel], size=8, opacity=0.7),
                            hovertemplate='<b>Tile:</b> %{x}<br>' +
                                        '<b>MUX Responsivity:</b> %{y:.4f} A/W<br>' +
                                        '<b>Bank:</b> ' + str(bank) + '<br>' +
                                        '<b>Channel:</b> ' + str(channel) + '<br>' +
                                        '<b>Laser Current:</b> 150 mA<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=bank+1
                    )
        
        fig.update_layout(
            title=dict(
                text='MPD_MUX Responsivity vs Tile SN at 150mA Laser Current (Channel-based)',
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
        fig.update_yaxes(title_text="MPD_MUX Responsivity (A/W)", row=1, col=1, range=[0, 0.03])
        fig.update_yaxes(title_text="MPD_MUX Responsivity (A/W)", row=1, col=2, range=[0, 0.03])
        
        # Save HTML file
        html_filename = "tp3p1_mpd_mux_responsivity_vs_tile_combined.html"
        fig.write_html(self.output_dir / html_filename)
        print(f"✅ Interactive HTML MPD_MUX responsivity vs tile plot saved: {html_filename}")
        
        return fig

    def plot_mpd_vs_tile_combined(self):
        """Create a combined MPD plot showing MPD_PIC vs Tile SN with scatter plots and box plots for each bank, matching tp1p4_mpd_vs_tile_combined.png format."""
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
        plot_filename = "tp3p1_mpd_vs_tile_combined.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ MPD vs Tile Combined plot saved: {plot_filename}")

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
        
        if efficiency_data:
            efficiency_df = pd.DataFrame(efficiency_data)
            print(f"\nCalculated tuning efficiency for {len(efficiency_df)} channel measurements")
            return efficiency_df
        else:
            print("No tuning efficiency data calculated")
            return None

    def plot_tuning_efficiency(self, efficiency_df):
        """Plot tuning efficiency vs current for each tile and channel."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data available!")
            return
        
        print(f"Creating tuning efficiency plot with {len(efficiency_df)} data points...")
        
        # Get unique tiles and sort them by date (if available) or alphabetically
        unique_tiles = efficiency_df['Tile_SN'].unique()
        sorted_tiles = sorted(unique_tiles)
        
        # Create figure with 2 subplots (Bank 0 and Bank 1) - matching TP1-4 format exactly
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        
        # Colors for different channels - matching TP1-4 color scheme
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
            
            # Match TP1-4 title format exactly
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
        plot_filename = "tp3p1_tuning_efficiency.png"
        plt.savefig(self.output_dir / plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✅ Tuning efficiency plot saved: {plot_filename}")
        
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

    def export_tuning_efficiency(self, efficiency_df):
        """Export tuning efficiency data to CSV and Excel files."""
        if efficiency_df is None or efficiency_df.empty:
            print("No tuning efficiency data to export!")
            return
        
        print("📊 Exporting tuning efficiency data...")
        
        # Save to CSV
        csv_filename = "tp3p1_tuning_efficiency.csv"
        efficiency_df.to_csv(self.data_dir / csv_filename, index=False)
        print(f"✅ Tuning efficiency CSV saved: {csv_filename}")
        
        # Save to Excel with formatting
        excel_filename = "tp3p1_tuning_efficiency.xlsx"
        with pd.ExcelWriter(self.data_dir / excel_filename, engine='openpyxl') as writer:
            efficiency_df.to_excel(writer, sheet_name='Tuning_Efficiency', index=False)
        print(f"✅ Tuning efficiency Excel saved: {excel_filename}")

    # Placeholder methods - still need to implement    
    def export_to_xarray(self):
        print("⚠️  Method export_to_xarray not fully implemented yet")


def main():
    print("=" * 80)
    print("TP3-1 LASER MODULE DATA ANALYSIS (PER-TILE PLOTS)")
    print("=" * 80)
    print("Test Point: TP3-1")
    print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    analyzer = TP3P1CombinedAnalyzer()
    analyzer.run_all()

if __name__ == "__main__":
    main()
