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
from pathlib import Path
from datetime import datetime
import glob
import re
from typing import Dict, List, Optional, Tuple, Any
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
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))
        
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
        colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))
        
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
    
    def run_analysis(self):
        """
        Run complete TP2P4 analysis pipeline
        """
        print("=" * 80)
        print("TP2P4 - WAVELENGTH SETPOINT & FREQUENCY ERROR ANALYSIS")
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
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"✅ Analysis complete")
        print(f"📁 Individual plots saved to: {self.output_dir}")
        print(f"📁 Summary plots saved to: {self.output_dir.parent}")
        print(f"📊 Individual plots show Set Laser (mA) vs Channel with Frequency Error (GHz) relative to reference grid")
        print(f"📊 Summary plots show Wavelength/Frequency Error vs Tile with boxplots and scatter plots")


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