#!/usr/bin/env python3
"""
TP2P4 - Test Point 2-4 Data Analysis Script
==========================================

This script analyzes TP2P4 data and creates wavelength setpoint plots with frequency error.
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


class TP2p4CombinedAnalyzers:
    """
    TP2P4 Combined Analysis Class
    
    This class provides analysis capabilities for TP2P4 data with dual y-axis plotting.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize TP2P4 combined analysis class
        
        Parameters
        ----------
        data_path : Optional[str]
            Path to TP2P4 data directory
        """
        script_dir = Path(__file__).parent
        self.data_path = Path(data_path) if data_path else script_dir / "../TP2-4"
        self.output_dir = script_dir / "plots" / "TP2-4"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load wavelength grid for frequency error calculation
        self.wavelength_grid = load_wavelength_grid()
        
        # Data storage
        self.raw_data = []
        self.processed_data = None
        
        print(f"TP2P4 Combined Analyzer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Output directory: {self.output_dir}")
    
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
        fig.suptitle('TP2-4 Analysis with Wavelength Locking Algorithm - Wavelength Error vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        channels = sorted(self.processed_data['Channel'].unique())
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
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
        
        # Save plot
        plot_path = self.output_dir.parent / "tp2p4_wavelength_error_vs_tile_combined.png"
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
        fig.suptitle('TP2-4 Analysis with Wavelength Locking Algorithm - Frequency Error vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        channels = sorted(self.processed_data['Channel'].unique())
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
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
                    ax1.text(pos, 35, f'{avg_error:.2f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Frequency Error (GHz)')
            ax1.set_ylim(-50, 50)
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
                    ax2.text(pos, 35, f'{avg_error:.2f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='center', rotation=90)
            
            ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Tile Serial Number')
            ax2.set_ylabel('Frequency Error (GHz)')
            ax2.set_ylim(-50, 50)
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis labels
            ax2.set_xticks(range(len(tiles)))
            ax2.set_xticklabels(tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir.parent / "tp2p4_frequency_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Frequency error summary plot saved: {plot_path}")
        plt.close()
    
    def load_data(self) -> bool:
        """
        Load data from TP2P4 CSV files and extract Y-numbers from filenames
        
        Returns
        -------
        bool
            True if data loaded successfully, False otherwise
        """
        print("🔍 Loading TP2P4 data...")
        
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
                ax1_right.set_ylim(-50, 50)  # Set frequency error y-axis range
                
                # Add legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_right.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax1.set_title('Bank 0 - No Data', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Channel')
                ax1.set_ylabel('Set Laser (mA)')
                ax1.text(0.5, 0.5, 'No data available for Bank 0', 
                        ha='center', va='center', transform=ax1.transAxes)
            
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
                ax2_right.set_ylim(-50, 50)  # Set frequency error y-axis range
                
                # Add legend
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_right.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax2.set_title('Bank 1 - No Data', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Channel')
                ax2.set_ylabel('Set Laser (mA)')
                ax2.text(0.5, 0.5, 'No data available for Bank 1', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Save plot with Y-number as filename
            plot_path = self.output_dir / f"Wavelength_Setpoint_{tile}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Wavelength_Setpoint_{tile}.png saved: {plot_path}")
            plt.close()
        
        print(f"✅ Created {len(tiles)} individual tile plots")
    
    # ============================================================================
    # OSA ANALYSIS SECTION
    # ============================================================================
    
    def scan_osa_tp2_data(self):
        """
        Scan OSA-TP2 directory to extract tile serial numbers and data
        
        Returns
        -------
        dict
            Dictionary containing OSA-TP2 tile serial numbers and folder paths
        """
        print("🔍 Scanning OSA-TP2 data...")
        
        # Define OSA-TP2 data path
        osa_tp2_path = self.data_path.parent / "OSA-TP2"
        
        if not osa_tp2_path.exists():
            print(f"❌ OSA-TP2 directory does not exist: {osa_tp2_path}")
            return {}
        
        # Find all folders in OSA-TP2
        osa_folders = [d for d in osa_tp2_path.iterdir() if d.is_dir()]
        
        osa_tile_data = {}
        
        for folder in osa_folders:
            # Extract Y-number from folder name using regex
            y_number_match = re.search(r'Y\d+', folder.name)
            if y_number_match:
                y_number = y_number_match.group(0)
                osa_tile_data[y_number] = folder
                print(f"✅ Found OSA-TP2 data for tile: {y_number}")
            else:
                print(f"⚠️  Could not extract Y-number from folder: {folder.name}")
        
        print(f"✅ Found {len(osa_tile_data)} OSA-TP2 tiles")
        return osa_tile_data
    
    def get_tp2_4_tile_serial_numbers(self):
        """
        Extract tile serial numbers from TP2-4 CSV files
        
        Returns
        -------
        set
            Set of tile serial numbers from TP2-4 data
        """
        print("🔍 Extracting TP2-4 tile serial numbers...")
        
        if not self.data_path.exists():
            print(f"❌ TP2-4 directory does not exist: {self.data_path}")
            return set()
        
        # Find all CSV files in TP2-4
        csv_files = list(self.data_path.glob("*.csv"))
        
        tp2_4_tiles = set()
        
        for file_path in csv_files:
            # Extract Y-number from filename
            y_number_match = re.search(r'Y\d+', file_path.name)
            if y_number_match:
                y_number = y_number_match.group(0)
                tp2_4_tiles.add(y_number)
                print(f"✅ Found TP2-4 data for tile: {y_number}")
            else:
                print(f"⚠️  Could not extract Y-number from file: {file_path.name}")
        
        print(f"✅ Found {len(tp2_4_tiles)} TP2-4 tiles")
        return tp2_4_tiles
    
    def compare_tile_serial_numbers(self):
        """
        Compare tile serial numbers between OSA-TP2 and TP2-4 data
        
        Returns
        -------
        dict
            Dictionary containing comparison results
        """
        print("\n" + "=" * 80)
        print("TILE SERIAL NUMBER COMPARISON")
        print("=" * 80)
        
        # Get tile serial numbers from both sources
        osa_tile_data = self.scan_osa_tp2_data()
        osa_tiles = set(osa_tile_data.keys())
        
        tp2_4_tiles = self.get_tp2_4_tile_serial_numbers()
        
        # Compare the sets
        common_tiles = osa_tiles.intersection(tp2_4_tiles)
        osa_only = osa_tiles - tp2_4_tiles
        tp2_4_only = tp2_4_tiles - osa_tiles
        
        # Print results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   OSA-TP2 tiles:     {len(osa_tiles)}")
        print(f"   TP2-4 tiles:       {len(tp2_4_tiles)}")
        print(f"   Common tiles:      {len(common_tiles)}")
        print(f"   OSA-TP2 only:      {len(osa_only)}")
        print(f"   TP2-4 only:        {len(tp2_4_only)}")
        
        if len(osa_tiles) == len(tp2_4_tiles) and len(osa_only) == 0 and len(tp2_4_only) == 0:
            print("✅ PERFECT MATCH: All tile serial numbers match between OSA-TP2 and TP2-4")
        else:
            print("⚠️  MISMATCH: Tile serial numbers do not match perfectly")
        
        # Print detailed differences if any
        if osa_only:
            print(f"\n🔍 Tiles found ONLY in OSA-TP2 ({len(osa_only)}):")
            for tile in sorted(osa_only):
                print(f"   - {tile}")
        
        if tp2_4_only:
            print(f"\n🔍 Tiles found ONLY in TP2-4 ({len(tp2_4_only)}):")
            for tile in sorted(tp2_4_only):
                print(f"   - {tile}")
        
        if common_tiles:
            print(f"\n🔍 Common tiles ({len(common_tiles)}):")
            for tile in sorted(common_tiles):
                print(f"   - {tile}")
        
        return {
            'osa_tiles': osa_tiles,
            'tp2_4_tiles': tp2_4_tiles,
            'common_tiles': common_tiles,
            'osa_only': osa_only,
            'tp2_4_only': tp2_4_only,
            'perfect_match': len(osa_tiles) == len(tp2_4_tiles) and len(osa_only) == 0 and len(tp2_4_only) == 0
        }
    
    def analyze_osa_data(self):
        """
        Analyze OSA (Optical Spectrum Analyzer) data
        
        This method will contain OSA-specific analysis functionality.
        Currently performs tile serial number comparison.
        """
        print("\n" + "=" * 80)
        print("OSA DATA ANALYSIS")
        print("=" * 80)
        
        # Perform tile serial number comparison
        comparison_results = self.compare_tile_serial_numbers()
        
        # Store results for later use
        self.osa_comparison_results = comparison_results
        
        return comparison_results
    
    def load_osa_spectrum_data(self, tile_serial_number: str) -> Dict[str, pd.DataFrame]:
        """
        Load OSA spectrum data from MaxMPDmA.xls files for a specific tile
        
        Parameters
        ----------
        tile_serial_number : str
            Tile serial number (e.g., Y25170084)
            
        Returns
        -------
        dict
            Dictionary containing spectrum data for Bank0 and Bank1
        """
        # Find the OSA-TP2 folder for this tile
        osa_tp2_path = self.data_path.parent / "OSA-TP2"
        
        # Find the folder containing this tile's data
        tile_folder = None
        for folder in osa_tp2_path.iterdir():
            if folder.is_dir() and tile_serial_number in folder.name:
                tile_folder = folder
                break
        
        if not tile_folder:
            print(f"❌ OSA-TP2 folder not found for tile: {tile_serial_number}")
            return {}
        
        spectrum_data = {}
        
        # Load Bank0 and Bank1 MaxMPDmA.xls files using glob patterns
        for bank in [0, 1]:
            # Use glob to find files with varying temperature values
            pattern = f"Bank{bank}_Channel1-8_Temp*_MaxMPDmA.xls"
            matching_files = list(tile_folder.glob(pattern))
            
            if matching_files:
                # Take the first matching file (should be only one)
                maxmpd_file = matching_files[0]
                
                try:
                    # Load the data - it's tab-separated with wavelength and spectrum columns
                    data = pd.read_csv(maxmpd_file, sep='\t', header=None, 
                                     names=['Wavelength_nm', 'OSA_Spectrum_dB'])
                    
                    # Filter to wavelength range 1300-1320nm
                    data = data[(data['Wavelength_nm'] >= 1300) & (data['Wavelength_nm'] <= 1320)]
                    
                    spectrum_data[f'Bank{bank}'] = data
                    print(f"✅ Loaded {len(data)} spectrum points for {tile_serial_number} Bank{bank} from {maxmpd_file.name}")
                    
                except Exception as e:
                    print(f"❌ Error loading {maxmpd_file}: {e}")
                    spectrum_data[f'Bank{bank}'] = pd.DataFrame(columns=['Wavelength_nm', 'OSA_Spectrum_dB'])
            else:
                print(f"⚠️  No MaxMPDmA.xls file found for {tile_serial_number} Bank{bank} (pattern: {pattern})")
                spectrum_data[f'Bank{bank}'] = pd.DataFrame(columns=['Wavelength_nm', 'OSA_Spectrum_dB'])
        
        return spectrum_data
    
    def create_osa_spectrum_plots(self):
        """
        Create OSA spectrum plots for each tile using MaxMPDmA.xls data
        """
        if not hasattr(self, 'osa_comparison_results'):
            print("❌ OSA comparison results not available. Run analyze_osa_data() first.")
            return
        
        print("📊 Creating OSA spectrum plots...")
        
        # Get the common tiles between OSA-TP2 and TP2-4
        common_tiles = self.osa_comparison_results['common_tiles']
        
        if not common_tiles:
            print("❌ No common tiles found between OSA-TP2 and TP2-4")
            return
        
        plots_created = 0
        
        # Dictionary to store all peak data for analysis
        self.osa_peak_data = {}
        
        for tile in sorted(common_tiles):
            # Load OSA spectrum data for this tile
            spectrum_data = self.load_osa_spectrum_data(tile)
            
            if not spectrum_data:
                print(f"⚠️  No spectrum data found for {tile}")
                continue
            
            # Create figure with 2 subplots (one for each bank)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f'OSA Spectrum Analysis - {tile}', fontsize=16, fontweight='bold')
            
            # Initialize peak data for this tile
            self.osa_peak_data[tile] = {'Bank0': [], 'Bank1': []}
            
            # Plot Bank0 spectrum
            bank0_data = spectrum_data.get('Bank0', pd.DataFrame())
            if not bank0_data.empty:
                ax1.plot(bank0_data['Wavelength_nm'], bank0_data['OSA_Spectrum_dB'], 
                        'b-', linewidth=1, alpha=0.8, label='OSA Spectrum')
                
                # Find and annotate peaks > -30dB
                peaks, _ = find_peaks(bank0_data['OSA_Spectrum_dB'], height=-30, distance=50)
                for peak_idx in peaks:
                    peak_wl = bank0_data.iloc[peak_idx]['Wavelength_nm']
                    peak_power = bank0_data.iloc[peak_idx]['OSA_Spectrum_dB']
                    
                    # Store peak data
                    self.osa_peak_data[tile]['Bank0'].append({
                        'wavelength_nm': peak_wl,
                        'power_dB': peak_power
                    })
                    
                    # Annotate peak
                    ax1.annotate(f'{peak_power:.1f} dB\n{peak_wl:.2f} nm', 
                               xy=(peak_wl, peak_power), xytext=(peak_wl, peak_power + 5),
                               ha='center', va='bottom', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))
                
                ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('OSA Spectrum (dB)')
                ax1.set_xlim(1300, 1320)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
            else:
                ax1.set_title('Bank 0 - No Data', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('OSA Spectrum (dB)')
                ax1.set_xlim(1300, 1320)
                ax1.text(0.5, 0.5, 'No spectrum data available for Bank 0', 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # Plot Bank1 spectrum
            bank1_data = spectrum_data.get('Bank1', pd.DataFrame())
            if not bank1_data.empty:
                ax2.plot(bank1_data['Wavelength_nm'], bank1_data['OSA_Spectrum_dB'], 
                        'r-', linewidth=1, alpha=0.8, label='OSA Spectrum')
                
                # Find and annotate peaks > -30dB
                peaks, _ = find_peaks(bank1_data['OSA_Spectrum_dB'], height=-30, distance=50)
                for peak_idx in peaks:
                    peak_wl = bank1_data.iloc[peak_idx]['Wavelength_nm']
                    peak_power = bank1_data.iloc[peak_idx]['OSA_Spectrum_dB']
                    
                    # Store peak data
                    self.osa_peak_data[tile]['Bank1'].append({
                        'wavelength_nm': peak_wl,
                        'power_dB': peak_power
                    })
                    
                    # Annotate peak
                    ax2.annotate(f'{peak_power:.1f} dB\n{peak_wl:.2f} nm', 
                               xy=(peak_wl, peak_power), xytext=(peak_wl, peak_power + 5),
                               ha='center', va='bottom', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))
                
                ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('OSA Spectrum (dB)')
                ax2.set_xlim(1300, 1320)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
            else:
                ax2.set_title('Bank 1 - No Data', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('OSA Spectrum (dB)')
                ax2.set_xlim(1300, 1320)
                ax2.text(0.5, 0.5, 'No spectrum data available for Bank 1', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"Wavelength_Spectrum_{tile}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Wavelength_Spectrum_{tile}.png saved: {plot_path}")
            plt.close()
            
            plots_created += 1
        
        # Save peak data to CSV file
        self.save_peak_data_to_csv()
        
        # Compare peak wavelengths with TP2-4 OSA_Wave data
        self.compare_peak_wavelengths_with_tp2_4()
        
        print(f"✅ Created {plots_created} OSA spectrum plots with peak annotations (> -30dB)")
    
    def create_annotate_power_plots(self):
        """
        Create AnnotatePower plots using OSA spectrum peak data
        Shows scaled power (mW) vs channel number for each tile - scaled relative to TP2-4 data
        """
        if not hasattr(self, 'osa_peak_data'):
            print("❌ No OSA peak data available. Run OSA analysis first.")
            return
        
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No TP2-4 processed data available for scaling.")
            return
        
        print("📊 Creating AnnotatePower plots with TP2-4 scaling...")
        
        plots_created = 0
        
        for tile, banks in self.osa_peak_data.items():
            # Get TP2-4 data for this tile
            tp2_4_tile_data = self.processed_data[self.processed_data['TileSerialNumber'] == tile]
            
            if tp2_4_tile_data.empty:
                print(f"⚠️  No TP2-4 data found for tile {tile}, skipping scaling")
                continue
            
            # Create figure with 2 subplots (one for each bank)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'OSA Peak Power (Scaled) - {tile}', fontsize=16, fontweight='bold')
            
            # Process Bank 0
            bank0_peaks = banks.get('Bank0', [])
            if bank0_peaks:
                # Convert dBm to mW and calculate total OSA power for Bank 0
                osa_powers_mw = []
                channels = []
                
                for peak in bank0_peaks:
                    peak_wl = peak['wavelength_nm']
                    peak_power_dbm = peak['power_dB']
                    
                    # Convert dBm to mW: P_mW = 10^(P_dBm/10)
                    peak_power_mw = 10 ** (peak_power_dbm / 10)
                    
                    # Find closest channel for this wavelength
                    closest_channel = self.find_closest_channel(peak_wl, 0)  # Bank 0
                    
                    if closest_channel is not None:
                        channels.append(closest_channel)
                        osa_powers_mw.append(peak_power_mw)
                
                # Calculate scaling factor for Bank 0
                if osa_powers_mw:
                    osa_total_power = sum(osa_powers_mw)
                    
                    # Get TP2-4 total power for Bank 0, scaled down by factor 4
                    tp2_4_bank0_data = tp2_4_tile_data[tp2_4_tile_data['Bank'] == 0]
                    if not tp2_4_bank0_data.empty:
                        tp2_4_total_power = tp2_4_bank0_data['Power(mW)'].sum()
                        tp2_4_scaled_power = tp2_4_total_power / 4  # Scale down by factor 4
                        
                        scaling_factor = tp2_4_scaled_power / osa_total_power
                        
                        # Apply scaling to individual powers
                        scaled_powers = [p * scaling_factor for p in osa_powers_mw]
                        
                        # Plot scaled powers
                        ax1.scatter(channels, scaled_powers, color='blue', s=100, alpha=0.7, marker='o')
                    else:
                        ax1.scatter(channels, osa_powers_mw, color='blue', s=100, alpha=0.7, marker='o')
            
            ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Channel Number')
            ax1.set_ylabel('Power (mW)')
            ax1.set_xlim(0.5, 8.5)
            ax1.set_xticks(range(1, 9))
            ax1.grid(True, alpha=0.3)
            
            # Process Bank 1
            bank1_peaks = banks.get('Bank1', [])
            if bank1_peaks:
                # Convert dBm to mW and calculate total OSA power for Bank 1
                osa_powers_mw = []
                channels = []
                
                for peak in bank1_peaks:
                    peak_wl = peak['wavelength_nm']
                    peak_power_dbm = peak['power_dB']
                    
                    # Convert dBm to mW: P_mW = 10^(P_dBm/10)
                    peak_power_mw = 10 ** (peak_power_dbm / 10)
                    
                    # Find closest channel for this wavelength
                    closest_channel = self.find_closest_channel(peak_wl, 1)  # Bank 1
                    
                    if closest_channel is not None:
                        channels.append(closest_channel)
                        osa_powers_mw.append(peak_power_mw)
                
                # Calculate scaling factor for Bank 1
                if osa_powers_mw:
                    osa_total_power = sum(osa_powers_mw)
                    
                    # Get TP2-4 total power for Bank 1, scaled down by factor 4
                    tp2_4_bank1_data = tp2_4_tile_data[tp2_4_tile_data['Bank'] == 1]
                    if not tp2_4_bank1_data.empty:
                        tp2_4_total_power = tp2_4_bank1_data['Power(mW)'].sum()
                        tp2_4_scaled_power = tp2_4_total_power / 4  # Scale down by factor 4
                        
                        scaling_factor = tp2_4_scaled_power / osa_total_power
                        
                        # Apply scaling to individual powers
                        scaled_powers = [p * scaling_factor for p in osa_powers_mw]
                        
                        # Plot scaled powers
                        ax2.scatter(channels, scaled_powers, color='red', s=100, alpha=0.7, marker='s')
                    else:
                        ax2.scatter(channels, osa_powers_mw, color='red', s=100, alpha=0.7, marker='s')
            
            ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Channel Number')
            ax2.set_ylabel('Power (mW)')
            ax2.set_xlim(0.5, 8.5)
            ax2.set_xticks(range(1, 9))
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"AnnotatePower_{tile}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ AnnotatePower_{tile}.png saved: {plot_path}")
            plt.close()
            
            plots_created += 1
        
        print(f"✅ Created {plots_created} AnnotatePower plots with TP2-4 scaling (factor 4 reduction)")
    
    def create_tp2p4_channel_power_vs_tile_summary_plot(self):
        """
        Create TP2-4 channel power vs tile summary plot with scatter plots and boxplots
        """
        if not hasattr(self, 'osa_peak_data'):
            print("❌ No OSA peak data available. Run OSA analysis first.")
            return
        
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No TP2-4 processed data available for scaling.")
            return
        
        print("📊 Creating TP2-4 channel power vs tile summary plot...")
        
        # Collect all scaled power data for both banks
        all_power_data = {'Bank0': [], 'Bank1': []}
        
        for tile, banks in self.osa_peak_data.items():
            # Get TP2-4 data for this tile
            tp2_4_tile_data = self.processed_data[self.processed_data['TileSerialNumber'] == tile]
            
            if tp2_4_tile_data.empty:
                continue
            
            # Process both banks
            for bank_idx in [0, 1]:
                bank_key = f'Bank{bank_idx}'
                bank_peaks = banks.get(bank_key, [])
                
                if bank_peaks:
                    # Convert dBm to mW and calculate total OSA power
                    osa_powers_mw = []
                    channels = []
                    
                    for peak in bank_peaks:
                        peak_wl = peak['wavelength_nm']
                        peak_power_dbm = peak['power_dB']
                        
                        # Convert dBm to mW: P_mW = 10^(P_dBm/10)
                        peak_power_mw = 10 ** (peak_power_dbm / 10)
                        
                        # Find closest channel for this wavelength
                        closest_channel = self.find_closest_channel(peak_wl, bank_idx)
                        
                        if closest_channel is not None:
                            channels.append(closest_channel)
                            osa_powers_mw.append(peak_power_mw)
                    
                    # Calculate scaling factor
                    if osa_powers_mw:
                        osa_total_power = sum(osa_powers_mw)
                        
                        # Get TP2-4 total power, scaled down by factor 4
                        tp2_4_bank_data = tp2_4_tile_data[tp2_4_tile_data['Bank'] == bank_idx]
                        if not tp2_4_bank_data.empty:
                            tp2_4_total_power = tp2_4_bank_data['Power(mW)'].sum()
                            tp2_4_scaled_power = tp2_4_total_power / 4  # Scale down by factor 4
                            
                            scaling_factor = tp2_4_scaled_power / osa_total_power
                            
                            # Apply scaling to individual powers and store data
                            for i, (channel, power) in enumerate(zip(channels, osa_powers_mw)):
                                scaled_power = power * scaling_factor
                                all_power_data[bank_key].append({
                                    'TileSerialNumber': tile,
                                    'Channel': channel,
                                    'Power_mW': scaled_power
                                })
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('TP2-4 Channel Power vs Tile Summary', fontsize=16, fontweight='bold')
        
        # Get unique tiles for x-axis
        tiles = sorted(self.processed_data['TileSerialNumber'].unique())
        channels = list(range(1, 9))  # Channels 1-8
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
        # Plot for Bank 0
        if all_power_data['Bank0']:
            bank0_df = pd.DataFrame(all_power_data['Bank0'])
            
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank0_df[bank0_df['Channel'] == channel]
                if not channel_data.empty:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [tiles.index(tile) for tile in channel_data['TileSerialNumber']]
                    ax1.scatter(positions, channel_data['Power_mW'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank0_df[bank0_df['TileSerialNumber'] == tile]['Power_mW'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax1.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
        
        ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Power(mW)')
        ax1.set_ylim(8, 20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis labels
        ax1.set_xticks(range(len(tiles)))
        ax1.set_xticklabels(tiles, rotation=45, ha='right')
        
        # Calculate and annotate average power for each tile at 18mW
        if all_power_data['Bank0']:
            bank0_df = pd.DataFrame(all_power_data['Bank0'])
            for pos, tile in enumerate(tiles):
                tile_data = bank0_df[bank0_df['TileSerialNumber'] == tile]['Power_mW']
                if not tile_data.empty:
                    avg_power = tile_data.mean()
                    ax1.text(pos, 18, f'{avg_power:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='bottom', rotation=90)
        
        # Plot for Bank 1
        if all_power_data['Bank1']:
            bank1_df = pd.DataFrame(all_power_data['Bank1'])
            
            # Scatter plot for each channel
            for i, channel in enumerate(channels):
                channel_data = bank1_df[bank1_df['Channel'] == channel]
                if not channel_data.empty:
                    # Convert tile serial numbers to positions for proper alignment with boxplot
                    positions = [tiles.index(tile) for tile in channel_data['TileSerialNumber']]
                    ax2.scatter(positions, channel_data['Power_mW'], 
                              color=colors[i], alpha=0.7, s=50, label=f'Channel {channel}')
            
            # Box plot
            box_data = []
            box_positions = []
            for pos, tile in enumerate(tiles):
                tile_data = bank1_df[bank1_df['TileSerialNumber'] == tile]['Power_mW'].values
                if len(tile_data) > 0:
                    box_data.append(tile_data)
                    box_positions.append(pos)
            
            if box_data:
                bp = ax2.boxplot(box_data, positions=box_positions, widths=0.6, 
                               patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.3)
        
        ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tile Serial Number')
        ax2.set_ylabel('Power(mW)')
        ax2.set_ylim(8, 20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis labels
        ax2.set_xticks(range(len(tiles)))
        ax2.set_xticklabels(tiles, rotation=45, ha='right')
        
        # Calculate and annotate average power for each tile at 18mW
        if all_power_data['Bank1']:
            bank1_df = pd.DataFrame(all_power_data['Bank1'])
            for pos, tile in enumerate(tiles):
                tile_data = bank1_df[bank1_df['TileSerialNumber'] == tile]['Power_mW']
                if not tile_data.empty:
                    avg_power = tile_data.mean()
                    ax2.text(pos, 18, f'{avg_power:.1f}', 
                            fontsize=10, color='red', fontweight='bold',
                            ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir.parent / "tp2p4_channel_power_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ TP2-4 channel power vs tile summary plot saved: {plot_path}")
        plt.close()
    
    def save_peak_data_to_csv(self):
        """
        Save OSA peak data to CSV file
        """
        if not hasattr(self, 'osa_peak_data'):
            print("❌ No OSA peak data available")
            return
        
        print("💾 Saving OSA peak data to CSV...")
        
        # Create a list to store all peak data
        peak_records = []
        
        for tile, banks in self.osa_peak_data.items():
            for bank, peaks in banks.items():
                bank_num = 0 if bank == 'Bank0' else 1
                for i, peak in enumerate(peaks):
                    peak_records.append({
                        'TileSerialNumber': tile,
                        'Bank': bank_num,
                        'PeakNumber': i + 1,
                        'Wavelength_nm': peak['wavelength_nm'],
                        'Power_dB': peak['power_dB']
                    })
        
        # Convert to DataFrame and save
        if peak_records:
            peak_df = pd.DataFrame(peak_records)
            peak_csv_path = self.output_dir.parent / "osa_peak_data.csv"
            peak_df.to_csv(peak_csv_path, index=False)
            print(f"✅ OSA peak data saved to: {peak_csv_path}")
            print(f"📊 Total peaks found: {len(peak_records)} (> -30dB)")
        else:
            print("⚠️  No peaks found to save")
    
    def find_closest_channel(self, wavelength_nm: float, bank: int) -> Optional[int]:
        """
        Find the closest channel number for a given wavelength and bank
        
        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nm
        bank : int
            Bank number (0 or 1)
            
        Returns
        -------
        int or None
            Channel number (1-8) or None if no close match found
        """
        try:
            min_diff = float('inf')
            closest_channel = None
            
            # Check all channels (1-8) for this bank
            for channel in range(1, 9):
                ref_wavelength = get_channel_value(bank, channel, 'wavelength', self.wavelength_grid)
                diff = abs(wavelength_nm - ref_wavelength)
                
                # If this is closer and within reasonable range (±2nm)
                if diff < min_diff and diff < 2.0:
                    min_diff = diff
                    closest_channel = channel
            
            return closest_channel
            
        except Exception as e:
            print(f"Error finding closest channel for wavelength {wavelength_nm} nm: {e}")
            return None
    
    def compare_peak_wavelengths_with_tp2_4(self):
        """
        Compare OSA spectrum peak wavelengths with TP2-4 OSA_Wave data
        """
        if not hasattr(self, 'osa_peak_data'):
            print("❌ No OSA peak data available")
            return
        
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No TP2-4 processed data available")
            return
        
        print("\n" + "=" * 80)
        print("OSA SPECTRUM PEAK vs TP2-4 OSA_WAVE COMPARISON")
        print("=" * 80)
        
        # Create comparison results
        comparison_results = []
        
        for tile, banks in self.osa_peak_data.items():
            # Get TP2-4 data for this tile
            tp2_4_tile_data = self.processed_data[self.processed_data['TileSerialNumber'] == tile]
            
            if tp2_4_tile_data.empty:
                print(f"⚠️  No TP2-4 data found for tile {tile}")
                continue
            
            for bank_name, peaks in banks.items():
                bank_num = 0 if bank_name == 'Bank0' else 1
                
                # Get TP2-4 data for this bank
                tp2_4_bank_data = tp2_4_tile_data[tp2_4_tile_data['Bank'] == bank_num]
                
                if tp2_4_bank_data.empty:
                    continue
                
                # Get OSA_Wave wavelengths from TP2-4 data
                tp2_4_wavelengths = tp2_4_bank_data['OSA_Wave(nm)'].values
                
                # For each peak, find closest TP2-4 wavelength
                for peak_idx, peak in enumerate(peaks):
                    peak_wl = peak['wavelength_nm']
                    peak_power = peak['power_dB']
                    
                    # Find closest TP2-4 wavelength
                    if len(tp2_4_wavelengths) > 0:
                        differences = np.abs(tp2_4_wavelengths - peak_wl)
                        closest_idx = np.argmin(differences)
                        closest_tp2_4_wl = tp2_4_wavelengths[closest_idx]
                        wavelength_diff = peak_wl - closest_tp2_4_wl
                        
                        # Get channel info for the closest match
                        closest_row = tp2_4_bank_data.iloc[closest_idx]
                        channel = closest_row['Channel']
                        
                        comparison_results.append({
                            'TileSerialNumber': tile,
                            'Bank': bank_num,
                            'PeakNumber': peak_idx + 1,
                            'OSA_Spectrum_Peak_nm': peak_wl,
                            'OSA_Spectrum_Power_dB': peak_power,
                            'TP2_4_OSA_Wave_nm': closest_tp2_4_wl,
                            'TP2_4_Channel': channel,
                            'Wavelength_Difference_nm': wavelength_diff,
                            'Abs_Difference_nm': abs(wavelength_diff)
                        })
        
        # Convert to DataFrame and save
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_csv_path = self.output_dir.parent / "osa_peak_vs_tp2_4_comparison.csv"
            comparison_df.to_csv(comparison_csv_path, index=False)
            print(f"✅ Comparison results saved to: {comparison_csv_path}")
            
            # Print summary statistics
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"   Total comparisons: {len(comparison_results)}")
            print(f"   Mean absolute difference: {comparison_df['Abs_Difference_nm'].mean():.4f} nm")
            print(f"   Std deviation: {comparison_df['Abs_Difference_nm'].std():.4f} nm")
            print(f"   Max difference: {comparison_df['Abs_Difference_nm'].max():.4f} nm")
            print(f"   Min difference: {comparison_df['Abs_Difference_nm'].min():.4f} nm")
            
            # Print matches within different tolerances
            tolerance_levels = [0.1, 0.5, 1.0, 2.0]
            for tolerance in tolerance_levels:
                matches = len(comparison_df[comparison_df['Abs_Difference_nm'] <= tolerance])
                percentage = (matches / len(comparison_results)) * 100
                print(f"   Within ±{tolerance:.1f}nm: {matches}/{len(comparison_results)} ({percentage:.1f}%)")
        else:
            print("⚠️  No comparison data available")
    
    def create_osa_plots(self):
        """
        Create OSA-specific plots
        
        This method generates OSA spectrum plots from MaxMPDmA.xls files.
        """
        print("\n" + "=" * 80)
        print("OSA SPECTRUM PLOTS")
        print("=" * 80)
        
        # Create OSA spectrum plots
        self.create_osa_spectrum_plots()
    
    def calculate_osa_metrics(self):
        """
        Calculate OSA-specific metrics
        
        This method will calculate metrics specific to OSA data analysis.
        Currently empty - to be implemented.
        """
        # TODO: Implement OSA metrics calculation
        pass
    
    def export_osa_results(self):
        """
        Export OSA analysis results
        
        This method will export OSA analysis results to files.
        Currently empty - to be implemented.
        """
        # TODO: Implement OSA results export functionality
        pass
    
    def create_tp2p4_power_vs_tile_plot(self):
        """
        Create total power vs tile plot from TP2-4 data (similar to tp2p0_all_channel_on_power_vs_tile_combined.png)
        """
        if self.processed_data is None or self.processed_data.empty:
            print("❌ No processed data available. Run load_data() first.")
            return
        
        print("📊 Creating TP2-4 total power vs tile plot...")
        
        # Extract total power data and multiply by 2
        power_data = self.processed_data.copy()
        power_data['TotalPower_mW'] = power_data['Power(mW)'] * 2
        
        # Create figure with single plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        fig.suptitle('TP2-4 Analysis - Total Power vs Tile', 
                    fontsize=16, fontweight='bold')
        
        # Get unique tiles and channels
        tiles = sorted(power_data['TileSerialNumber'].unique())
        channels = sorted(power_data['Channel'].unique())
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(channels)))
        
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
        ax.set_ylim(0, 150)
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
        
        # Save plot
        plot_path = self.output_dir.parent / "tp2p4_total_power_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ TP2-4 total power plot saved: {plot_path}")
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
    
    # ============================================================================
    # END OSA ANALYSIS SECTION
    # ============================================================================
    
    def calculate_centre_frequency_error(self, tile_data):
        """
        Calculate centre frequency error - average of all wavelengths in Bank0 and Bank1
        compared to average of all wavelengths in wavelength grid for bank0 and bank1
        
        Returns: dict with 'frequency_error_ghz' and 'wavelength_error_nm'
        """
        try:
            # Get measured wavelengths for both banks
            bank0_wavelengths = []
            bank1_wavelengths = []
            
            for _, row in tile_data.iterrows():
                if row['Bank'] == 0:
                    bank0_wavelengths.append(row['OSA_Wave(nm)'])
                elif row['Bank'] == 1:
                    bank1_wavelengths.append(row['OSA_Wave(nm)'])
            
            # Calculate average measured wavelengths
            avg_measured_bank0 = np.mean(bank0_wavelengths) if bank0_wavelengths else 0
            avg_measured_bank1 = np.mean(bank1_wavelengths) if bank1_wavelengths else 0
            avg_measured_total = (avg_measured_bank0 + avg_measured_bank1) / 2
            
            # Get reference wavelengths from grid
            bank0_ref_wavelengths = []
            bank1_ref_wavelengths = []
            
            for channel in range(1, 9):  # Channels 1-8
                bank0_ref = get_channel_value(0, channel, 'wavelength', self.wavelength_grid)
                bank1_ref = get_channel_value(1, channel, 'wavelength', self.wavelength_grid)
                bank0_ref_wavelengths.append(bank0_ref)
                bank1_ref_wavelengths.append(bank1_ref)
            
            avg_ref_bank0 = np.mean(bank0_ref_wavelengths)
            avg_ref_bank1 = np.mean(bank1_ref_wavelengths)
            avg_ref_total = (avg_ref_bank0 + avg_ref_bank1) / 2
            
            # Calculate errors
            wavelength_error_nm = avg_measured_total - avg_ref_total
            
            # Calculate frequency error using the same formula as other methods
            c = 299792458  # Speed of light in m/s
            wavelength_error_m = wavelength_error_nm * 1e-9
            ref_wavelength_m = avg_ref_total * 1e-9
            frequency_error_hz = -c * wavelength_error_m / (ref_wavelength_m ** 2)
            frequency_error_ghz = frequency_error_hz / 1e9
            
            return {
                'frequency_error_ghz': frequency_error_ghz,
                'wavelength_error_nm': wavelength_error_nm
            }
            
        except Exception as e:
            print(f"Error calculating centre frequency error: {e}")
            return {'frequency_error_ghz': 0.0, 'wavelength_error_nm': 0.0}
    
    def calculate_channel_spacing_errors(self, tile_data, spacing_type='adjacent'):
        """
        Calculate channel spacing frequency errors
        
        Parameters:
        spacing_type: 'adjacent' (1vs2, 2vs3, etc), '4th' (1vs5, 2vs6, etc), '5th' (1vs6, 2vs7, etc)
        
        Returns: dict with lists of frequency and wavelength errors, and channel pair info
        """
        try:
            if spacing_type == 'adjacent':
                channel_pairs = [(i, i+1) for i in range(1, 8)]  # 1vs2, 2vs3, ..., 7vs8
            elif spacing_type == '4th':
                channel_pairs = [(i, i+4) for i in range(1, 5)]  # 1vs5, 2vs6, 3vs7, 4vs8
            elif spacing_type == '5th':
                channel_pairs = [(i, i+5) for i in range(1, 4)]  # 1vs6, 2vs7, 3vs8
            else:
                raise ValueError(f"Unknown spacing_type: {spacing_type}")
            
            frequency_errors = []
            wavelength_errors = []
            channel_pair_info = []  # Track which channel pair each error belongs to
            bank_info = []  # Track which bank each error belongs to
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for ch1, ch2 in channel_pairs:
                    # Get measured wavelengths
                    ch1_data = bank_data[bank_data['Channel'] == ch1-1]  # Convert to 0-based
                    ch2_data = bank_data[bank_data['Channel'] == ch2-1]
                    
                    if len(ch1_data) > 0 and len(ch2_data) > 0:
                        measured_wl1 = ch1_data['OSA_Wave(nm)'].iloc[0]
                        measured_wl2 = ch2_data['OSA_Wave(nm)'].iloc[0]
                        measured_spacing = measured_wl2 - measured_wl1
                        
                        # Get reference wavelengths
                        ref_wl1 = get_channel_value(bank, ch1, 'wavelength', self.wavelength_grid)
                        ref_wl2 = get_channel_value(bank, ch2, 'wavelength', self.wavelength_grid)
                        ref_spacing = ref_wl2 - ref_wl1
                        
                        # Calculate spacing errors
                        wavelength_spacing_error = measured_spacing - ref_spacing
                        wavelength_errors.append(wavelength_spacing_error)
                        
                        # Calculate frequency spacing error
                        # Convert wavelength spacing to frequency spacing
                        c = 299792458  # Speed of light in m/s
                        avg_ref_wl = (ref_wl1 + ref_wl2) / 2
                        
                        # Δf = c * Δλ / λ²
                        wavelength_spacing_error_m = wavelength_spacing_error * 1e-9
                        avg_ref_wl_m = avg_ref_wl * 1e-9
                        frequency_spacing_error_hz = c * wavelength_spacing_error_m / (avg_ref_wl_m ** 2)
                        frequency_spacing_error_ghz = frequency_spacing_error_hz / 1e9
                        frequency_errors.append(frequency_spacing_error_ghz)
                        
                        # Store channel pair and bank info
                        channel_pair_info.append(f'{ch1}vs{ch2}')
                        bank_info.append(bank)
            
            return {
                'frequency_errors_ghz': frequency_errors,
                'wavelength_errors_nm': wavelength_errors,
                'channel_pairs': channel_pair_info,
                'banks': bank_info
            }
            
        except Exception as e:
            print(f"Error calculating {spacing_type} channel spacing errors: {e}")
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
        
        # Create plots (these have one value per tile, so use the original simple method)
        self._create_simple_metric_plot(tile_names, centre_freq_errors, 'Centre Frequency Error', 'GHz', 'tp2p4_centre_frequency_error_vs_tile_combined.png')
        self._create_simple_metric_plot(tile_names, centre_wl_errors, 'Centre Wavelength Error', 'nm', 'tp2p4_centre_wavelength_error_vs_tile_combined.png')
    
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
            freq_filename = f'tp2p4_adjacent_channel_spacing_frequency_error_vs_tile_combined.png'
            wl_filename = f'tp2p4_adjacent_channel_spacing_wavelength_error_vs_tile_combined.png'
        else:
            freq_filename = f'tp2p4_{spacing_type}_adjacent_channel_spacing_frequency_error_vs_tile_combined.png'
            wl_filename = f'tp2p4_{spacing_type}_adjacent_channel_spacing_wavelength_error_vs_tile_combined.png'
        
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
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            output_file = self.output_dir.parent / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Created {metric_name} summary plot: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Failed to create {metric_name} plot: {e}")
    
    def _create_channel_spacing_plot(self, tile_names, all_data, channel_pairs, metric_name, unit, filename):
        """Helper method for channel spacing metrics (multiple values per tile, separated by bank)"""
        try:
            # Create figure with 2 subplots (Bank 0 and Bank 1)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
            fig.suptitle(f'{metric_name} vs Tile', fontsize=16, fontweight='bold')
            
            # Set y-axis limits
            ylim = (-50, 50) if unit == 'GHz' else (-0.5, 0.5)
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)
            
            # Create color map for different channel pair combinations
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(channel_pairs)))
            color_map = {pair: colors[i] for i, pair in enumerate(channel_pairs)}
            
            # Plot for Bank 0
            bank0_data = all_data['Bank0']
            if bank0_data:
                # Scatter plot - show all individual channel pair data points with different colors
                for pair in channel_pairs:
                    pair_color = color_map[pair]
                    x_positions = []
                    y_values = []
                    
                    for pos, tile in enumerate(tile_names):
                        tile_errors = bank0_data[pair].get(tile, [])
                        if tile_errors:
                            # Create multiple points at same x position with slight jitter
                            x_pos = [pos + (i-len(tile_errors)/2)*0.02 for i in range(len(tile_errors))]
                            x_positions.extend(x_pos)
                            y_values.extend(tile_errors)
                    
                    if x_positions:
                        ax1.scatter(x_positions, y_values, alpha=0.7, s=50, color=pair_color, label=pair)
                
                # Box plot - combine all pairs for each tile
                box_data = []
                box_positions = []
                for pos, tile in enumerate(tile_names):
                    all_tile_errors = []
                    for pair in channel_pairs:
                        tile_errors = bank0_data[pair].get(tile, [])
                        all_tile_errors.extend(tile_errors)
                    if all_tile_errors:
                        box_data.append(all_tile_errors)
                        box_positions.append(pos)
                
                if box_data:
                    bp = ax1.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.3)
                
                # Add annotations with average values
                text_y_pos = 40 if unit == 'GHz' else 0.4
                for pos, tile in enumerate(tile_names):
                    all_tile_errors = []
                    for pair in channel_pairs:
                        tile_errors = bank0_data[pair].get(tile, [])
                        all_tile_errors.extend(tile_errors)
                    if all_tile_errors:
                        avg_error = np.mean(all_tile_errors)
                        ax1.text(pos, text_y_pos, f'{avg_error:.2f}', 
                               fontsize=8, color='red', fontweight='bold',
                               ha='center', va='center', rotation=90)
            
            ax1.set_title('Bank 0', fontsize=14, fontweight='bold')
            ax1.set_ylabel(f'{metric_name} ({unit})', fontsize=12)
            ax1.set_xticks(range(len(tile_names)))
            ax1.set_xticklabels(tile_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot for Bank 1
            bank1_data = all_data['Bank1']
            if bank1_data:
                # Scatter plot - show all individual channel pair data points with different colors
                for pair in channel_pairs:
                    pair_color = color_map[pair]
                    x_positions = []
                    y_values = []
                    
                    for pos, tile in enumerate(tile_names):
                        tile_errors = bank1_data[pair].get(tile, [])
                        if tile_errors:
                            # Create multiple points at same x position with slight jitter
                            x_pos = [pos + (i-len(tile_errors)/2)*0.02 for i in range(len(tile_errors))]
                            x_positions.extend(x_pos)
                            y_values.extend(tile_errors)
                    
                    if x_positions:
                        ax2.scatter(x_positions, y_values, alpha=0.7, s=50, color=pair_color, label=pair)
                
                # Box plot - combine all pairs for each tile
                box_data = []
                box_positions = []
                for pos, tile in enumerate(tile_names):
                    all_tile_errors = []
                    for pair in channel_pairs:
                        tile_errors = bank1_data[pair].get(tile, [])
                        all_tile_errors.extend(tile_errors)
                    if all_tile_errors:
                        box_data.append(all_tile_errors)
                        box_positions.append(pos)
                
                if box_data:
                    bp = ax2.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.3)
                
                # Add annotations with average values
                text_y_pos = 40 if unit == 'GHz' else 0.4
                for pos, tile in enumerate(tile_names):
                    all_tile_errors = []
                    for pair in channel_pairs:
                        tile_errors = bank1_data[pair].get(tile, [])
                        all_tile_errors.extend(tile_errors)
                    if all_tile_errors:
                        avg_error = np.mean(all_tile_errors)
                        ax2.text(pos, text_y_pos, f'{avg_error:.2f}', 
                               fontsize=8, color='red', fontweight='bold',
                               ha='center', va='center', rotation=90)
            
            ax2.set_title('Bank 1', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Tile Serial Number', fontsize=12)
            ax2.set_ylabel(f'{metric_name} ({unit})', fontsize=12)
            ax2.set_xticks(range(len(tile_names)))
            ax2.set_xticklabels(tile_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            output_file = self.output_dir.parent / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Created {metric_name} summary plot: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Failed to create {metric_name} plot: {e}")
    
    def run_analysis(self):
        """
        Run complete TP2P4 analysis pipeline
        """
        print("=" * 80)
        print("TP2P4 - WAVELENGTH SETPOINT & FREQUENCY ERROR ANALYSIS")
        print("=" * 80)
        
        # Step 1: Perform OSA tile serial number comparison
        osa_results = self.analyze_osa_data()
        
        # Step 2: Load data
        if not self.load_data():
            print("❌ Failed to load data. Analysis aborted.")
            return
        
        # Step 3: Create individual plots
        self.create_wavelength_setpoint_plot()
        
        # Step 4: Create summary plots
        self.create_wavelength_error_summary_plot()
        self.create_frequency_error_summary_plot()
        
        # Step 5: Create total power vs tile plot
        self.create_tp2p4_power_vs_tile_plot()
        
        # Step 6: Create OSA spectrum plots
        self.create_osa_plots()
        
        # Step 7: Create AnnotatePower plots
        self.create_annotate_power_plots()

        # Step 8: Create TP2-4 channel power vs tile summary plot
        self.create_tp2p4_channel_power_vs_tile_summary_plot()
        
        # Step 9: Create new summary plots for centre frequency error, channel spacing errors
        self.create_new_summary_plots()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"✅ Analysis complete")
        print(f"📁 Individual plots saved to: {self.output_dir}")
        print(f"📁 Summary plots saved to: {self.output_dir.parent}")
        print(f"📊 Individual plots show Set Laser (mA) vs Channel with Frequency Error (GHz) relative to reference grid")
        print(f"📊 Summary plots show Wavelength/Frequency Error vs Tile with boxplots and scatter plots")
        print(f"📊 Total power plot shows Power (mW) vs Tile for both banks combined (values multiplied by 2)")
        print(f"📊 OSA spectrum plots show wavelength vs spectrum (1300-1320nm) for each tile")
        print(f"📊 AnnotatePower plots show Power (mW) vs Channel Number - OSA data scaled relative to TP2-4 (factor 4 reduction)")
        print(f"📊 Channel power vs tile summary plot shows scaled power distribution across all tiles and channels")
        print(f"📊 New summary plots include centre frequency/wavelength errors and adjacent/4th/5th channel spacing errors")
        print(f"📊 All OSA data processed: {len(self.osa_comparison_results['common_tiles']) if hasattr(self, 'osa_comparison_results') else 0} tiles")


def main():
    """
    Main function to run TP2P4 analysis
    """
    # Initialize analyzer
    analyzer = TP2p4CombinedAnalyzers()
    
    # Run analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 