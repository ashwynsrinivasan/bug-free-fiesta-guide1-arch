#!/usr/bin/env python3
"""
TP3P3 - Test Point 3-3 Data Analysis Script
==========================================

This script analyzes TP3P3 data and creates wavelength setpoint plots with frequency error.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import glob
import re
from typing import Dict, List, Optional, Tuple, Any
from scipy.signal import find_peaks
from wavelength_grid_utils import load_wavelength_grid, get_channel_value

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12


class TP3p3CombinedAnalyzers:
    """
    TP3P3 Combined Analysis Class
    
    This class provides analysis capabilities for TP3P3 data with dual y-axis plotting.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize TP3P3 combined analysis class
        
        Parameters
        ----------
        data_path : Optional[str]
            Path to TP3P3 data directory
        """
        script_dir = Path(__file__).parent
        self.data_path = Path(data_path) if data_path else script_dir / "../TP3-3"
        self.output_dir = script_dir / "plots" / "TP3-3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Summary plots go to main plots folder
        self.summary_plots_dir = script_dir / "plots"
        self.summary_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load wavelength grid for frequency error calculation
        self.wavelength_grid = load_wavelength_grid()
        
        # Data storage
        self.raw_data = []
        self.processed_data = None
        
        print(f"TP3P3 Combined Analyzer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Summary plots directory: {self.summary_plots_dir}")
    
    def calculate_frequency_error(self, wavelength_nm: float, bank: int, channel: int) -> float:
        """
        Calculate frequency error in GHz compared to reference wavelength grid
        
        Parameters
        ----------
        wavelength_nm : float
            Measured wavelength in nm
        bank : int
            Bank number (0 or 1)
        channel : int
            Channel number (0-7, will be converted to 1-8 for grid lookup)
            
        Returns
        -------
        float
            Frequency error in GHz
        """
        try:
            # Convert channel from 0-7 to 1-8 for grid lookup
            grid_channel = channel + 1
            
            # Get reference wavelength from grid
            ref_wavelength = get_channel_value(bank, grid_channel, 'wavelength', self.wavelength_grid)
            
            # Calculate frequency error using the formula:
            # Δf = -c * Δλ / λ²
            # where c is speed of light, Δλ is wavelength error, λ is reference wavelength
            
            c = 299792458  # Speed of light in m/s
            wavelength_error_m = (wavelength_nm - ref_wavelength) * 1e-9  # Convert nm to m
            ref_wavelength_m = ref_wavelength * 1e-9  # Convert nm to m
            
            frequency_error_hz = -c * wavelength_error_m / (ref_wavelength_m ** 2)
            frequency_error_ghz = frequency_error_hz / 1e9  # Convert Hz to GHz
            
            return frequency_error_ghz
            
        except Exception as e:
            print(f"Error calculating frequency error for bank {bank}, channel {channel}: {e}")
            return 0.0
    
    def calculate_wavelength_error(self, wavelength_nm: float, bank: int, channel: int) -> float:
        """
        Calculate wavelength error in nm compared to reference wavelength grid
        
        Parameters
        ----------
        wavelength_nm : float
            Measured wavelength in nm
        bank : int
            Bank number (0 or 1)
        channel : int
            Channel number (0-7, will be converted to 1-8 for grid lookup)
            
        Returns
        -------
        float
            Wavelength error in nm
        """
        try:
            # Convert channel from 0-7 to 1-8 for grid lookup
            grid_channel = channel + 1
            
            # Get reference wavelength from grid
            ref_wavelength = get_channel_value(bank, grid_channel, 'wavelength', self.wavelength_grid)
            
            # Calculate wavelength error
            wavelength_error = wavelength_nm - ref_wavelength
            
            return wavelength_error
            
        except Exception as e:
            print(f"Error calculating wavelength error for bank {bank}, channel {channel}: {e}")
            return 0.0

    def load_data(self) -> bool:
        """
        Load data from TP3P3 CSV files and extract Y-numbers from filenames
        
        Returns
        -------
        bool
            True if data loaded successfully, False otherwise
        """
        print("🔍 Loading TP3P3 data...")
        
        if not self.data_path.exists():
            print(f"❌ Data directory does not exist: {self.data_path}")
            return False
        
        # Find all CSV files
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in: {self.data_path}")
            return False
        
        print(f"✅ Found {len(csv_files)} CSV files")
        
        all_data = []
        
        for file_path in csv_files:
            try:
                data = pd.read_csv(file_path)
                
                # Extract Y-number from filename as the tile serial number
                filename = file_path.name
                # Look for pattern like "Y25170084" in the filename
                y_number_match = re.search(r'Y\d+', filename)
                if y_number_match:
                    y_number = y_number_match.group(0)
                    data['TileSerialNumber'] = y_number
                else:
                    print(f"⚠️  Could not extract Y-number from {filename}")
                    continue
                
                all_data.append(data)
                print(f"✅ Loaded {len(data)} records from {file_path.name} (Tile: {y_number})")
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
                continue
        
        if all_data:
            # Combine all data
            self.processed_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Successfully loaded {len(self.processed_data)} total records")
            return True
        else:
            print("❌ No data loaded")
            return False

    def create_wavelength_setpoint_plot(self):
        """
        Create individual Wavelength_Setpoint_TileSN.png plots for each tile using Y-numbers
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating individual Wavelength_Setpoint_TileSN.png plots...")
        
        # Get unique tile serial numbers (Y-numbers)
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        print(f"Found {len(tiles)} unique tile serial numbers: {tiles}")
        
        for tile in tiles:
            tile_data = self.processed_data[self.processed_data['TileSerialNumber'] == tile]
            
            # Create figure with 2 subplots (one for each bank)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f'Wavelength Setpoint vs Channel - {tile}', fontsize=16, fontweight='bold')
            
            # Plot for Bank 0
            bank0_data = tile_data[tile_data['Bank'] == 0]
            if not bank0_data.empty:
                ax1_right = ax1.twinx()
                
                # Sort by channel for proper line connection
                bank0_data = bank0_data.sort_values('Channel')
                
                # Calculate frequency error for each data point
                frequency_errors = []
                for _, row in bank0_data.iterrows():
                    freq_error = self.calculate_frequency_error(row['OSA_Wave(nm)'], 0, row['Channel'])
                    frequency_errors.append(freq_error)
                
                # Plot Set Laser (mA) on left y-axis
                ax1.plot(bank0_data['Channel'], bank0_data['Set Laser(mA)'], 
                        'o-', color='blue', linewidth=2, markersize=6,
                        label='Set Laser (mA)')
                
                # Plot frequency error on right y-axis
                ax1_right.plot(bank0_data['Channel'], frequency_errors, 
                             's--', color='red', linewidth=2, markersize=6,
                             alpha=0.7, label='Frequency Error (GHz)')
                
                # Configure Bank 0 subplot
                ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Channel')
                ax1.set_ylabel('Set Laser (mA)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(range(8))
                
                ax1_right.set_ylabel('Frequency Error (GHz)', color='red')
                ax1_right.tick_params(axis='y', labelcolor='red')
                
                # Add legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_right.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Plot for Bank 1
            bank1_data = tile_data[tile_data['Bank'] == 1]
            if not bank1_data.empty:
                ax2_right = ax2.twinx()
                
                # Sort by channel for proper line connection
                bank1_data = bank1_data.sort_values('Channel')
                
                # Calculate frequency error for each data point
                frequency_errors = []
                for _, row in bank1_data.iterrows():
                    freq_error = self.calculate_frequency_error(row['OSA_Wave(nm)'], 1, row['Channel'])
                    frequency_errors.append(freq_error)
                
                # Plot Set Laser (mA) on left y-axis
                ax2.plot(bank1_data['Channel'], bank1_data['Set Laser(mA)'], 
                        'o-', color='blue', linewidth=2, markersize=6,
                        label='Set Laser (mA)')
                
                # Plot frequency error on right y-axis
                ax2_right.plot(bank1_data['Channel'], frequency_errors, 
                             's--', color='red', linewidth=2, markersize=6,
                             alpha=0.7, label='Frequency Error (GHz)')
                
                # Configure Bank 1 subplot
                ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Channel')
                ax2.set_ylabel('Set Laser (mA)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(range(8))
                
                ax2_right.set_ylabel('Frequency Error (GHz)', color='red')
                ax2_right.tick_params(axis='y', labelcolor='red')
                
                # Add legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_right.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            # Save plot to TP3-3 directory
            plot_path = self.output_dir / f"Wavelength_Setpoint_{tile}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Individual plot saved: {plot_path}")
            plt.close()

    def create_wavelength_error_summary_plot(self):
        """
        Create summary plot for wavelength error vs tile combined
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating wavelength error summary plot...")
        
        # Calculate wavelength errors for all data
        wavelength_errors = []
        for _, row in self.processed_data.iterrows():
            wl_error = self.calculate_wavelength_error(row['OSA_Wave(nm)'], row['Bank'], row['Channel'])
            wavelength_errors.append(wl_error)
        
        self.processed_data['WavelengthError'] = wavelength_errors
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('TP3-3 Analysis with Wavelength Locking Algorithm - Wavelength Error vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        channels = sorted(self.processed_data['Channel'].unique())
        colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
        # Plot for Bank 0
        bank0_data = self.processed_data[self.processed_data['Bank'] == 0]
        if not bank0_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank0_data[bank0_data['Channel'] == channel]
                if not channel_data.empty:
                    ax1.scatter(channel_data['TileSerialNumber'], channel_data['WavelengthError'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['WavelengthError'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax1.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average error for each individual tile
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['WavelengthError'].values
                if len(tile_data) > 0:
                    avg_error = tile_data.mean()
                    ax1.text(pos, 0.15, f'{avg_error:.3f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Wavelength Error (nm)')
            ax1.set_ylim(-0.25, 0.25)
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis labels
            ax1.set_xticks(range(len(tiles)))
            ax1.set_xticklabels(tiles, rotation=45, ha='right')
        
        # Plot for Bank 1
        bank1_data = self.processed_data[self.processed_data['Bank'] == 1]
        if not bank1_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank1_data[bank1_data['Channel'] == channel]
                if not channel_data.empty:
                    ax2.scatter(channel_data['TileSerialNumber'], channel_data['WavelengthError'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['WavelengthError'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax2.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average error for each individual tile
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['WavelengthError'].values
                if len(tile_data) > 0:
                    avg_error = tile_data.mean()
                    ax2.text(pos, 0.15, f'{avg_error:.3f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Tile Serial Number')
            ax2.set_ylabel('Wavelength Error (nm)')
            ax2.set_ylim(-0.25, 0.25)
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis labels
            ax2.set_xticks(range(len(tiles)))
            ax2.set_xticklabels(tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot to main plots folder
        plot_path = self.summary_plots_dir / "tp3p3_wavelength_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Wavelength error summary plot saved: {plot_path}")
        plt.close()

    def create_frequency_error_summary_plot(self):
        """
        Create summary plot for frequency error vs tile combined
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating frequency error summary plot...")
        
        # Calculate frequency errors for all data (reuse existing method)
        frequency_errors = []
        for _, row in self.processed_data.iterrows():
            freq_error = self.calculate_frequency_error(row['OSA_Wave(nm)'], row['Bank'], row['Channel'])
            frequency_errors.append(freq_error)
        
        self.processed_data['FrequencyError'] = frequency_errors
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('TP3-3 Analysis with Wavelength Locking Algorithm - Frequency Error vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        channels = sorted(self.processed_data['Channel'].unique())
        colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
        # Plot for Bank 0
        bank0_data = self.processed_data[self.processed_data['Bank'] == 0]
        if not bank0_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank0_data[bank0_data['Channel'] == channel]
                if not channel_data.empty:
                    ax1.scatter(channel_data['TileSerialNumber'], channel_data['FrequencyError'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['FrequencyError'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax1.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average error for each individual tile
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['FrequencyError'].values
                if len(tile_data) > 0:
                    avg_error = tile_data.mean()
                    ax1.text(pos, 15, f'{avg_error:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Frequency Error (GHz)')
            ax1.set_ylim(-25, 25)
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis labels
            ax1.set_xticks(range(len(tiles)))
            ax1.set_xticklabels(tiles, rotation=45, ha='right')
        
        # Plot for Bank 1
        bank1_data = self.processed_data[self.processed_data['Bank'] == 1]
        if not bank1_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank1_data[bank1_data['Channel'] == channel]
                if not channel_data.empty:
                    ax2.scatter(channel_data['TileSerialNumber'], channel_data['FrequencyError'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['FrequencyError'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax2.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average error for each individual tile
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['FrequencyError'].values
                if len(tile_data) > 0:
                    avg_error = tile_data.mean()
                    ax2.text(pos, 15, f'{avg_error:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Tile Serial Number')
            ax2.set_ylabel('Frequency Error (GHz)')
            ax2.set_ylim(-25, 25)
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis labels
            ax2.set_xticks(range(len(tiles)))
            ax2.set_xticklabels(tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot to main plots folder
        plot_path = self.summary_plots_dir / "tp3p3_frequency_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Frequency error summary plot saved: {plot_path}")
        plt.close()
    
    def create_tp3p3_power_vs_tile_plot(self):
        """
        Create total power vs tile plot from TP3-3 data
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating TP3-3 total power vs tile plot...")
        
        # Extract total power data and multiply by 2 (following tp2p4 pattern)
        power_data = self.processed_data.copy()
        power_data['TotalPower_mW'] = power_data['Power(mW)'] * 2
        
        # Create figure with single plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        fig.suptitle('TP3-3 Analysis - Total Power vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(power_data['TileSerialNumber'].unique())
        channels = sorted(power_data['Channel'].unique())
        
        # Create separate colors for Bank 0 and Bank 1
        bank_colors = ['blue', 'red']
        bank_markers = ['o', 's']  # circle for Bank 0, square for Bank 1
        
        # Plot data for both banks
        for bank in [0, 1]:
            bank_data = power_data[power_data['Bank'] == bank]
            if not bank_data.empty:
                # Scatter plot for each channel
                for i, channel in enumerate(channels):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    if not channel_data.empty:
                        ax.scatter(channel_data['TileSerialNumber'], channel_data['TotalPower_mW'], 
                                  color=bank_colors[bank], marker=bank_markers[bank],
                                  alpha=0.7, s=50, label=f'Bank {bank} Ch {channel}')
        
        # Configure plot
        ax.set_title('Bank 0 and Bank 1 Combined', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tile Serial Number')
        ax.set_ylabel('Total Power (mW)')
        ax.set_ylim(0, 250)
        ax.grid(True, alpha=0.3)
        
        # Create custom legend to avoid too many entries
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, alpha=0.7, label='Bank 0'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=8, alpha=0.7, label='Bank 1')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set x-axis labels
        ax.set_xticks(range(len(tiles)))
        ax.set_xticklabels(tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot to main plots folder
        plot_path = self.summary_plots_dir / "tp3p3_total_power_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ TP3-3 total power plot saved: {plot_path}")
        plt.close()
        
        # Print summary statistics
        print(f"\n📊 TOTAL POWER SUMMARY:")
        
        # Bank 0 statistics
        bank0_data = power_data[power_data['Bank'] == 0]
        if not bank0_data.empty:
            bank0_powers = bank0_data['TotalPower_mW'].values
            print(f"   Bank 0:")
            print(f"     Mean: {bank0_powers.mean():.2f} mW")
            print(f"     Std:  {bank0_powers.std():.2f} mW")
            print(f"     Min:  {bank0_powers.min():.2f} mW")
            print(f"     Max:  {bank0_powers.max():.2f} mW")
        
        # Bank 1 statistics
        bank1_data = power_data[power_data['Bank'] == 1]
        if not bank1_data.empty:
            bank1_powers = bank1_data['TotalPower_mW'].values
            print(f"   Bank 1:")
            print(f"     Mean: {bank1_powers.mean():.2f} mW")
            print(f"     Std:  {bank1_powers.std():.2f} mW")
            print(f"     Min:  {bank1_powers.min():.2f} mW")
            print(f"     Max:  {bank1_powers.max():.2f} mW")
        
        # Combined statistics
        all_powers = power_data['TotalPower_mW'].values
        print(f"   Combined:")
        print(f"     Mean: {all_powers.mean():.2f} mW")
        print(f"     Std:  {all_powers.std():.2f} mW")
        print(f"     Min:  {all_powers.min():.2f} mW")
        print(f"     Max:  {all_powers.max():.2f} mW")
        print(f"     Total records: {len(all_powers)}")

    def create_tp3p3_channel_power_vs_tile_summary_plot(self):
        """
        Create TP3-3 channel power vs tile summary plot with scatter plots and boxplots
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating TP3-3 channel power vs tile summary plot...")
        
        # Convert OSAl_Power from dBm to mW for direct comparison
        power_data = self.processed_data.copy()
        power_data['Power_mW'] = 10 ** (power_data['OSAl_Power(dBm)'] / 10)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('TP3-3 Channel Power vs Tile Summary', fontsize=16, fontweight='bold')
        
        # Get unique tiles for x-axis
        tiles = sorted(power_data['TileSerialNumber'].unique())
        channels = sorted(power_data['Channel'].unique())
        colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
        # Plot for Bank 0
        bank0_data = power_data[power_data['Bank'] == 0]
        if not bank0_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank0_data[bank0_data['Channel'] == channel]
                if not channel_data.empty:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [tiles.index(tile) for tile in channel_data['TileSerialNumber']]
                    ax1.scatter(positions, channel_data['Power_mW'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['Power_mW'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax1.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average power for each tile
            for pos, tile in enumerate(tiles):
                tile_data = bank0_data[bank0_data['TileSerialNumber'] == tile]['Power_mW']
                if not tile_data.empty:
                    avg_power = tile_data.mean()
                    ax1.text(pos, 30, f'{avg_power:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='bottom', rotation=90)
        
        ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Power (mW)')
        ax1.set_ylim(0, 40)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis labels
        ax1.set_xticks(range(len(tiles)))
        ax1.set_xticklabels(tiles, rotation=45, ha='right')
        
        # Plot for Bank 1
        bank1_data = power_data[power_data['Bank'] == 1]
        if not bank1_data.empty:
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank1_data[bank1_data['Channel'] == channel]
                if not channel_data.empty:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [tiles.index(tile) for tile in channel_data['TileSerialNumber']]
                    ax2.scatter(positions, channel_data['Power_mW'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['Power_mW'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax2.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
            
            # Calculate and annotate average power for each tile
            for pos, tile in enumerate(tiles):
                tile_data = bank1_data[bank1_data['TileSerialNumber'] == tile]['Power_mW']
                if not tile_data.empty:
                    avg_power = tile_data.mean()
                    ax2.text(pos, 30, f'{avg_power:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='bottom', rotation=90)
        
        ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tile Serial Number')
        ax2.set_ylabel('Power (mW)')
        ax2.set_ylim(0, 40)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis labels
        ax2.set_xticks(range(len(tiles)))
        ax2.set_xticklabels(tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot to main plots folder
        plot_path = self.summary_plots_dir / "tp3p3_channel_power_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ TP3-3 channel power vs tile plot saved: {plot_path}")
        plt.close()

    def calculate_centre_frequency_error(self, tile_data):
        """
        Calculate centre frequency error - average of all wavelengths in Bank0 and Bank1
        compared to average of all wavelengths in wavelength grid for bank0 and bank1
        
        Returns: dict with 'frequency_error_ghz' and 'wavelength_error_nm'
        """
        try:
            # Get all wavelengths for this tile
            bank0_data = tile_data[tile_data['Bank'] == 0]
            bank1_data = tile_data[tile_data['Bank'] == 1]
            
            measured_wavelengths = []
            reference_wavelengths = []
            
            # Process both banks
            for bank_idx, bank_data in enumerate([bank0_data, bank1_data]):
                if not bank_data.empty:
                    for _, row in bank_data.iterrows():
                        measured_wavelengths.append(row['OSA_Wave(nm)'])
                        # Get reference wavelength
                        grid_channel = row['Channel'] + 1  # Convert 0-7 to 1-8
                        ref_wl = get_channel_value(bank_idx, grid_channel, 'wavelength', self.wavelength_grid)
                        reference_wavelengths.append(ref_wl)
            
            if measured_wavelengths and reference_wavelengths:
                # Calculate average wavelengths
                avg_measured = np.mean(measured_wavelengths)
                avg_reference = np.mean(reference_wavelengths)
                
                # Calculate wavelength error
                wavelength_error = avg_measured - avg_reference
                
                # Calculate frequency error
                c = 299792458  # Speed of light in m/s
                wavelength_error_m = wavelength_error * 1e-9
                ref_wavelength_m = avg_reference * 1e-9
                frequency_error_hz = -c * wavelength_error_m / (ref_wavelength_m ** 2)
                frequency_error_ghz = frequency_error_hz / 1e9
                
                return {
                    'frequency_error_ghz': frequency_error_ghz,
                    'wavelength_error_nm': wavelength_error
                }
            else:
                return {'frequency_error_ghz': 0.0, 'wavelength_error_nm': 0.0}
                
        except Exception as e:
            print(f"Error calculating centre frequency error: {e}")
            return {'frequency_error_ghz': 0.0, 'wavelength_error_nm': 0.0}

    def calculate_channel_spacing_errors(self, tile_data, spacing_type):
        """
        Calculate channel spacing errors for adjacent, 4th, or 5th channel pairs
        
        Returns: dict with 'frequency_errors_ghz', 'wavelength_errors_nm', 'channel_pairs', 'banks'
        """
        frequency_errors = []
        wavelength_errors = []
        channel_pairs = []
        banks = []
        
        try:
            # Define channel pairs based on spacing type
            if spacing_type == 'adjacent':
                pairs = [(i, i+1) for i in range(8) if i+1 < 8]  # 0-1, 1-2, ..., 6-7
            elif spacing_type == '4th':
                pairs = [(i, i+4) for i in range(8) if i+4 < 8]  # 0-4, 1-5, 2-6, 3-7
            elif spacing_type == '5th':
                pairs = [(i, i+5) for i in range(8) if i+5 < 8]  # 0-5, 1-6, 2-7
            else:
                return {'frequency_errors_ghz': [], 'wavelength_errors_nm': [], 'channel_pairs': [], 'banks': []}
            
            # Process both banks
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                if not bank_data.empty:
                    # Get wavelengths for each channel
                    channel_wavelengths = {}
                    for _, row in bank_data.iterrows():
                        channel_wavelengths[row['Channel']] = row['OSA_Wave(nm)']
                    
                    # Calculate spacing errors for each pair
                    for ch1, ch2 in pairs:
                        if ch1 in channel_wavelengths and ch2 in channel_wavelengths:
                            # Measured spacing
                            measured_spacing = channel_wavelengths[ch2] - channel_wavelengths[ch1]
                            
                            # Reference spacing
                            ref_ch1 = get_channel_value(bank, ch1+1, 'wavelength', self.wavelength_grid)
                            ref_ch2 = get_channel_value(bank, ch2+1, 'wavelength', self.wavelength_grid)
                            reference_spacing = ref_ch2 - ref_ch1
                            
                            # Calculate errors
                            wavelength_error = measured_spacing - reference_spacing
                            
                            # Convert to frequency error
                            c = 299792458
                            avg_wavelength = (channel_wavelengths[ch1] + channel_wavelengths[ch2]) / 2
                            wavelength_error_m = wavelength_error * 1e-9
                            avg_wavelength_m = avg_wavelength * 1e-9
                            frequency_error_hz = -c * wavelength_error_m / (avg_wavelength_m ** 2)
                            frequency_error_ghz = frequency_error_hz / 1e9
                            
                            frequency_errors.append(frequency_error_ghz)
                            wavelength_errors.append(wavelength_error)
                            channel_pairs.append(f'{ch1+1}vs{ch2+1}')  # Convert to 1-8 numbering
                            banks.append(bank)
            
            return {
                'frequency_errors_ghz': frequency_errors,
                'wavelength_errors_nm': wavelength_errors,
                'channel_pairs': channel_pairs,
                'banks': banks
            }
            
        except Exception as e:
            print(f"Error calculating channel spacing errors: {e}")
            return {'frequency_errors_ghz': [], 'wavelength_errors_nm': [], 'channel_pairs': [], 'banks': []}

    def create_new_summary_plots(self):
        """Create summary plots for the new metrics with individual channel pair data points"""
        print("\n📊 Creating new summary analysis plots...")
        
        if self.processed_data is None or len(self.processed_data) == 0:
            print("❌ No processed data available for new summary plots")
            return
        
        # Group data by tile
        grouped = self.processed_data.groupby('TileSerialNumber')
        tile_names = sorted(grouped.groups.keys())
        
        # Create separate plots for each metric type
        self._create_centre_error_plots(grouped, tile_names)
        self._create_channel_spacing_plots(grouped, tile_names, 'adjacent', 7)
        self._create_channel_spacing_plots(grouped, tile_names, '4th', 4) 
        self._create_channel_spacing_plots(grouped, tile_names, '5th', 3)
        
        print(f"   ✓ Created 8 new summary plots")
    
    def _create_centre_error_plots(self, grouped, tile_names):
        """Create centre frequency/wavelength error plots"""
        # Collect centre error data
        centre_freq_errors = []
        centre_wl_errors = []
        
        for tile_sn in tile_names:
            tile_data = grouped.get_group(tile_sn)
            centre_error = self.calculate_centre_frequency_error(tile_data)
            centre_freq_errors.append(centre_error['frequency_error_ghz'])
            centre_wl_errors.append(centre_error['wavelength_error_nm'])
        
        # Create plots (these have one value per tile, so use the simple method)
        self._create_simple_metric_plot(tile_names, centre_freq_errors, 'Centre Frequency Error', 'GHz', 'tp3p3_centre_frequency_error_vs_tile_combined.png')
        self._create_simple_metric_plot(tile_names, centre_wl_errors, 'Centre Wavelength Error', 'nm', 'tp3p3_centre_wavelength_error_vs_tile_combined.png')
    
    def _create_channel_spacing_plots(self, grouped, tile_names, spacing_type, expected_pairs):
        """Create channel spacing error plots with individual channel pair data points"""
        
        # Collect all channel spacing data organized by channel pair combination
        all_freq_data = {'Bank0': {}, 'Bank1': {}}
        all_wl_data = {'Bank0': {}, 'Bank1': {}}
        
        # Get the expected channel pair combinations for this spacing type
        if spacing_type == 'adjacent':
            expected_combinations = [f'{i}vs{i+1}' for i in range(1, 8)]
        elif spacing_type == '4th':
            expected_combinations = [f'{i}vs{i+4}' for i in range(1, 5)]
        elif spacing_type == '5th':
            expected_combinations = [f'{i}vs{i+5}' for i in range(1, 4)]
        
        # Initialize data structure for each combination
        for bank in ['Bank0', 'Bank1']:
            for combo in expected_combinations:
                all_freq_data[bank][combo] = {}
                all_wl_data[bank][combo] = {}
                for tile in tile_names:
                    all_freq_data[bank][combo][tile] = []
                    all_wl_data[bank][combo][tile] = []
        
        for tile_sn in tile_names:
            tile_data = grouped.get_group(tile_sn)
            spacing_errors = self.calculate_channel_spacing_errors(tile_data, spacing_type)
            
            # Organize errors by channel pair combination and bank
            freq_errors = spacing_errors['frequency_errors_ghz']
            wl_errors = spacing_errors['wavelength_errors_nm']
            channel_pairs = spacing_errors['channel_pairs']
            banks = spacing_errors['banks']
            
            for freq_err, wl_err, pair, bank in zip(freq_errors, wl_errors, channel_pairs, banks):
                bank_key = f'Bank{bank}'
                all_freq_data[bank_key][pair][tile_sn].append(freq_err)
                all_wl_data[bank_key][pair][tile_sn].append(wl_err)
        
        # Create the plots
        spacing_name = {'adjacent': 'Adjacent', '4th': '4th Adjacent', '5th': '5th Adjacent'}[spacing_type]
        
        if spacing_type == 'adjacent':
            freq_filename = f'tp3p3_adjacent_channel_spacing_frequency_error_vs_tile_combined.png'
            wl_filename = f'tp3p3_adjacent_channel_spacing_wavelength_error_vs_tile_combined.png'
        else:
            freq_filename = f'tp3p3_{spacing_type}_adjacent_channel_spacing_frequency_error_vs_tile_combined.png'
            wl_filename = f'tp3p3_{spacing_type}_adjacent_channel_spacing_wavelength_error_vs_tile_combined.png'
        
        self._create_channel_spacing_plot(tile_names, all_freq_data, expected_combinations, f'{spacing_name} Channel Spacing Frequency Error', 'GHz', freq_filename)
        self._create_channel_spacing_plot(tile_names, all_wl_data, expected_combinations, f'{spacing_name} Channel Spacing Wavelength Error', 'nm', wl_filename)
    
    def _create_simple_metric_plot(self, tile_names, values, metric_name, unit, filename):
        """Helper method for simple metrics (one value per tile)"""
        try:
            fig, ax = plt.subplots(figsize=(20, 8))
            
            # Set y-axis limits based on unit type
            if unit == 'GHz':
                ax.set_ylim(-50, 50)
            elif unit == 'nm':
                ax.set_ylim(-0.5, 0.5)
            
            # Create scatter plot and box plot
            x_positions = range(len(tile_names))
            ax.scatter(x_positions, values, alpha=0.7, s=50, color='blue')
            
            box_data = [[val] for val in values]
            bp = ax.boxplot(box_data, positions=x_positions, widths=0.6, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
            
            # Add annotations
            text_y_pos = 40 if unit == 'GHz' else 0.4
            for pos, (tile, val) in enumerate(zip(tile_names, values)):
                ax.text(pos, text_y_pos, f'{val:.2f}', 
                       fontsize=8, color='red', fontweight='bold',
                       ha='center', va='center', rotation=90)
            
            # Customize plot
            ax.set_xlabel('Tile Serial Number', fontsize=12)
            ax.set_ylabel(f'{metric_name} ({unit})', fontsize=12)
            ax.set_title(f'{metric_name} vs Tile', fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tile_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.text(0.02, 0.98, f'Mean: {mean_val:.3f} {unit}\nStd: {std_val:.3f} {unit}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot to main plots folder
            output_file = self.summary_plots_dir / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Created {metric_name} summary plot: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Error creating {metric_name} plot: {e}")

    def _create_channel_spacing_plot(self, tile_names, all_data, expected_combinations, metric_name, unit, filename):
        """Helper method for channel spacing plots with individual data points"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
            fig.suptitle(f'{metric_name} vs Tile', fontsize=16, fontweight='bold')
            
            colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(expected_combinations)))
            
            # Set y-axis limits based on unit type
            if unit == 'GHz':
                y_lim = (-50, 50)
                text_y_pos = 40
            else:
                y_lim = (-0.5, 0.5)
                text_y_pos = 0.4
            
            for bank_idx, bank_key in enumerate(['Bank0', 'Bank1']):
                ax = ax1 if bank_idx == 0 else ax2
                bank_data = all_data[bank_key]
                
                # Plot scatter points for each channel pair combination
                for combo_idx, combo in enumerate(expected_combinations):
                    combo_data = bank_data[combo]
                    
                    for tile_idx, tile in enumerate(tile_names):
                        tile_values = combo_data[tile]
                        if tile_values:  # Only plot if there are values
                            # Add small random offset to x-position for visibility
                            x_positions = [tile_idx + (combo_idx - len(expected_combinations)/2) * 0.05] * len(tile_values)
                            ax.scatter(x_positions, tile_values, 
                                     color=colors[combo_idx], alpha=0.7, s=30, 
                                     label=f'{combo}' if tile_idx == 0 else "")
                
                # Create box plot for combined data
                box_data = []
                box_positions = []
                for tile_idx, tile in enumerate(tile_names):
                    tile_all_values = []
                    for combo in expected_combinations:
                        tile_all_values.extend(bank_data[combo][tile])
                    
                    if tile_all_values:
                        box_data.append(tile_all_values)
                        box_positions.append(tile_idx)
                
                if box_data:
                    bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.3)
                
                # Add average annotations
                for tile_idx, tile in enumerate(tile_names):
                    tile_all_values = []
                    for combo in expected_combinations:
                        tile_all_values.extend(bank_data[combo][tile])
                    
                    if tile_all_values:
                        avg_val = np.mean(tile_all_values)
                        ax.text(tile_idx, text_y_pos, f'{avg_val:.2f}', 
                               fontsize=8, color='red', fontweight='bold',
                               ha='center', va='center', rotation=90)
                
                # Customize subplot
                ax.set_title(f'Bank {bank_idx}', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'{metric_name.split()[-1]} ({unit})', fontsize=12)
                ax.set_ylim(y_lim)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(len(tile_names)))
                ax.set_xticklabels(tile_names, rotation=45, ha='right')
                
                if bank_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                
                if bank_idx == 1:
                    ax.set_xlabel('Tile Serial Number', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot to main plots folder
            output_file = self.summary_plots_dir / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Created {metric_name} summary plot: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Error creating {metric_name} plot: {e}")
    
    def run_analysis(self):
        """
        Run complete TP3P3 analysis pipeline
        """
        print("=" * 80)
        print("TP3P3 - WAVELENGTH SETPOINT & FREQUENCY ERROR ANALYSIS")
        print("=" * 80)
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data. Analysis aborted.")
            return
        
        # Step 2: Create individual plots
        self.create_wavelength_setpoint_plot()
        
        # Step 3: Create summary plots
        self.create_wavelength_error_summary_plot()
        self.create_frequency_error_summary_plot()
        
        # Step 4: Create total power vs tile plot
        self.create_tp3p3_power_vs_tile_plot()
        
        # Step 5: Create TP3-3 channel power vs tile summary plot
        self.create_tp3p3_channel_power_vs_tile_summary_plot()
        
        # Step 6: Create new summary plots for centre frequency error, channel spacing errors
        self.create_new_summary_plots()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"✅ Analysis complete")
        print(f"📁 Individual plots saved to: {self.output_dir}")
        print(f"📁 Summary plots saved to: {self.summary_plots_dir}")
        print(f"📊 Individual plots show Set Laser (mA) vs Channel with Frequency Error (GHz) relative to reference grid")
        print(f"📊 Summary plots show Wavelength/Frequency Error vs Tile with boxplots and scatter plots")
        print(f"📊 Total power plot shows Power (mW) vs Tile for both banks combined (values multiplied by 2)")
        print(f"📊 Channel power vs tile summary plot shows power distribution across all tiles and channels")
        print(f"📊 New summary plots include centre frequency/wavelength errors and adjacent/4th/5th channel spacing errors")
        print(f"📊 All TP3-3 data processed: {len(self.processed_data)} total records from {len(self.processed_data['TileSerialNumber'].unique())} tiles")


def main():
    """
    Main function to run TP3P3 analysis
    """
    # Initialize analyzer
    analyzer = TP3p3CombinedAnalyzers()
    
    # Run analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 