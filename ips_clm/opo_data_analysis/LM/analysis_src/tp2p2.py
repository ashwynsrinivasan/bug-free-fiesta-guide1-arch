"""
TP2P2 Combined Analysis Class
Combined analysis of TP2P2 data with comprehensive analysis methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import glob
import re
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TP2P2CombinerAnalyzers:
    """
    TP2P2 Combined Analysis Class
    
    This class provides comprehensive analysis capabilities for TP2P2 data,
    combining multiple analysis methods and providing a unified interface.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize TP2P2 analysis class
        
        Parameters
        ----------
        data_path : Optional[str]
            Path to TP2-2 data directory
        """
        script_dir = Path(__file__).parent
        self.data_path = Path(data_path) if data_path else script_dir / "../TP2-2"
        self.output_dir = script_dir / "plots" / "TP2-2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.temp_scan_files = []
        self.temp_scan_data = None
        self.tile_metadata = {}
        self.results = {}
        self.metadata = {}
        
        # Color palette for plotting
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def load_data(self, data_path: Optional[str] = None) -> None:
        """
        Load data for analysis
        
        Parameters
        ----------
        data_path : Optional[str]
            Path to data directory
        """
        if data_path:
            self.data_path = Path(data_path)
        self.load_temp_scan_files()
        self.load_temp_scan_data()
        
    def extract_serial_number(self, filename: str) -> Optional[str]:
        """Extract serial number from filename"""
        match = re.search(r'-Y(\d+)-TP2-2 Temp Scan\.csv$', filename)
        if match:
            return f"Y{match.group(1)}"
        return None
        
    def load_temp_scan_files(self) -> List[str]:
        """Load temperature scan files"""
        pattern = str(self.data_path / "*-TP2-2 Temp Scan.csv")
        self.temp_scan_files = sorted(glob.glob(pattern))
        print(f"Found {len(self.temp_scan_files)} TP2-2 Temp Scan files")
        
        if len(self.temp_scan_files) == 0:
            print(f"⚠️  No TP2-2 Temp Scan files found in {self.data_path}")
            all_csv_files = sorted(glob.glob(str(self.data_path / "*.csv")))
            if all_csv_files:
                print(f"   However, found {len(all_csv_files)} other CSV files:")
                for csv_file in all_csv_files[:5]:
                    print(f"     • {Path(csv_file).name}")
                if len(all_csv_files) > 5:
                    print(f"     • ... and {len(all_csv_files) - 5} more")
                    
        return self.temp_scan_files
        
    def load_temp_scan_data(self) -> None:
        """Load and process temperature scan data"""
        if not self.temp_scan_files:
            print("No temperature scan files to load")
            return
            
        dfs = []
        for file_path in self.temp_scan_files:
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
                    
                # Process the data into a long format
                if sn_from_filename:
                    processed_df = self.process_temp_scan_data(df, sn_from_filename)
                    if processed_df is not None:
                        dfs.append(processed_df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        if dfs:
            self.temp_scan_data = pd.concat(dfs, ignore_index=True)
            print(f"Combined temp scan data shape: {self.temp_scan_data.shape}")
            print(f"Captured metadata for {len(self.tile_metadata)} tiles")
            print(f"Available columns: {list(self.temp_scan_data.columns)}")
        else:
            print("No temp scan data loaded successfully")
            
    def process_temp_scan_data(self, df: pd.DataFrame, tile_sn: str) -> Optional[pd.DataFrame]:
        """Process temperature scan data from wide to long format"""
        if df.empty:
            return None
            
        processed_rows = []
        
        for _, row in df.iterrows():
            bank = row['Bank']
            
            # Process each channel (0-7)
            for channel in range(8):
                processed_row = {
                    'Tile_SN': tile_sn,
                    'filename': row.get('filename', ''),
                    'Bank': bank,
                    'Channel': channel,
                    'Set_PIC_Temp_C': row['Set PIC Temp(C)'],
                    'Set_MUX_Temp_C': row['Set MUX Temp(C)'],
                    'Set_Laser_mA': row['Set Laser(mA)'],
                    'T_PIC_C': row['T_PIC(C)'],
                    'T_MUX_C': row['T_MUX(C)'],
                    'Power_mW': row['Power(mW)'],
                    'MPD_PIC_uA': row[f'MPD_PIC_{channel}(uA)'],
                    'MPD_MUX_uA': row[f'MPD_MUX_{channel}(uA)'],
                    'PeakWave_nm': row[f'PeakWave_{channel}(nm)'],
                    'PeakPower_dBm': row[f'PeakPower_{channel}(dBm)'],
                    'Time': row['Time']
                }
                processed_rows.append(processed_row)
                
        processed_df = pd.DataFrame(processed_rows)
        
        # Convert numeric columns and handle missing values
        numeric_cols = ['Set_PIC_Temp_C', 'Set_MUX_Temp_C', 'Set_Laser_mA', 
                       'T_PIC_C', 'T_MUX_C', 'Power_mW', 'MPD_PIC_uA', 
                       'MPD_MUX_uA', 'PeakWave_nm', 'PeakPower_dBm']
        
        for col in numeric_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].replace('-', pd.NA)
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
        # Convert time
        processed_df['Time'] = pd.to_datetime(processed_df['Time'], format='mixed', errors='coerce')
        processed_df = processed_df.dropna(subset=['Time'])
        processed_df = processed_df.sort_values('Time')
        
        return processed_df
        
    def preprocess(self) -> None:
        """
        Preprocess the data before analysis
        """
        if self.temp_scan_data is None:
            print("No data to preprocess")
            return
            
        # Additional preprocessing steps can be added here
        print("Data preprocessing completed")
        
    def plot_temperature_scan_per_tile(self) -> None:
        """Create temperature scan plots for each tile"""
        if self.temp_scan_data is None:
            print("No temperature scan data available for plotting")
            return
            
        unique_tiles = sorted(self.temp_scan_data['Tile_SN'].unique())
        print(f"Creating temperature scan plots for {len(unique_tiles)} tiles")
        
        for tile_sn in unique_tiles:
            print(f"Creating temperature scan plot for tile {tile_sn}...")
            
            tile_data = self.temp_scan_data[self.temp_scan_data['Tile_SN'] == tile_sn]
            
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'TP2-2 Temperature Scan - Tile {tile_sn}', fontsize=16)
            
            # Plot 1: PeakWave vs Temperature for each bank
            for bank in [0, 1]:
                ax = axs[0, bank]
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    if len(channel_data) > 0:
                        color_idx = channel % len(self.colors)
                        ax.plot(channel_data['T_PIC_C'], channel_data['PeakWave_nm'], 
                               marker='o', linewidth=1.5, markersize=4, 
                               color=self.colors[color_idx], 
                               label=f'Ch{channel}')
                
                ax.set_xlabel('PIC Temperature (°C)')
                ax.set_ylabel('Peak Wavelength (nm)')
                ax.set_title(f'Bank {bank} - Wavelength vs Temperature')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
            # Plot 2: MPD vs Temperature for each bank
            for bank in [0, 1]:
                ax = axs[1, bank]
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    if len(channel_data) > 0:
                        color_idx = channel % len(self.colors)
                        ax.plot(channel_data['T_PIC_C'], channel_data['MPD_PIC_uA'], 
                               marker='o', linewidth=1.5, markersize=4, 
                               color=self.colors[color_idx], 
                               label=f'Ch{channel}')
                
                ax.set_xlabel('PIC Temperature (°C)')
                ax.set_ylabel('MPD PIC (µA)')
                ax.set_title(f'Bank {bank} - MPD vs Temperature')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"TempScan_{tile_sn}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Temperature scan plot saved: {plot_filename}")
            plt.close()
            
    def plot_wavelength_tuning_summary(self) -> None:
        """Create a summary plot of wavelength tuning across all tiles"""
        if self.temp_scan_data is None:
            print("No temperature scan data available for plotting")
            return
            
        # Calculate tuning efficiency (wavelength change per temperature change)
        tuning_data = []
        
        for tile_sn in self.temp_scan_data['Tile_SN'].unique():
            tile_data = self.temp_scan_data[self.temp_scan_data['Tile_SN'] == tile_sn]
            
            for bank in [0, 1]:
                bank_data = tile_data[tile_data['Bank'] == bank]
                
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    if len(channel_data) > 1:
                        # Calculate tuning efficiency using linear regression
                        temp_col = channel_data['T_PIC_C']
                        wave_col = channel_data['PeakWave_nm']
                        
                        # Remove NaN values
                        valid_mask = ~(pd.isna(temp_col) | pd.isna(wave_col))
                        temp_valid = temp_col[valid_mask]
                        wave_valid = wave_col[valid_mask]
                        
                        if len(temp_valid) > 1 and len(wave_valid) > 1:
                            temps = np.array(temp_valid)
                            waves = np.array(wave_valid)
                            slope, intercept = np.polyfit(temps, waves, 1)
                            tuning_data.append({
                                'Tile_SN': tile_sn,
                                'Bank': bank,
                                'Channel': channel,
                                'Tuning_Efficiency_nm_per_C': slope,
                                'Baseline_Wavelength_nm': intercept + slope * 42,  # at 42°C
                                'Temp_Range_C': float(temps.max() - temps.min())
                            })
                            
        if tuning_data:
            tuning_df = pd.DataFrame(tuning_data)
            
            # Create summary plot
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle('TP2-2 Wavelength Tuning Summary', fontsize=16)
            
            # Plot tuning efficiency by bank
            for bank in [0, 1]:
                ax = axs[bank]
                bank_data = tuning_df[tuning_df['Bank'] == bank]
                
                # Get unique tiles for this bank
                unique_tiles = sorted(set(bank_data['Tile_SN']))
                
                # Create boxplot data for each tile
                boxplot_data = []
                boxplot_positions = []
                tile_averages = []
                
                for i, tile_sn in enumerate(unique_tiles):
                    tile_data = bank_data[bank_data['Tile_SN'] == tile_sn]
                    tile_efficiencies = np.array(tile_data['Tuning_Efficiency_nm_per_C'])
                    
                    if len(tile_efficiencies) > 0:
                        boxplot_data.append(tile_efficiencies)
                        boxplot_positions.append(i)
                        tile_averages.append(np.mean(tile_efficiencies))
                
                # Create boxplots for each tile
                if boxplot_data:
                    bp = ax.boxplot(boxplot_data, positions=boxplot_positions, 
                                   widths=0.6, patch_artist=True, showfliers=True)
                    
                    # Style boxplots
                    for box in bp['boxes']:
                        box.set_facecolor('lightblue')
                        box.set_alpha(0.5)
                
                # Scatter plot for individual channels (overlay on boxplots)
                for channel in range(8):
                    channel_data = bank_data[bank_data['Channel'] == channel]
                    if len(channel_data) > 0:
                        color_idx = channel % len(self.colors)
                        
                        # Map tile names to positions
                        x_positions = []
                        y_values = []
                        
                        for tile_sn in channel_data['Tile_SN']:
                            tile_idx = unique_tiles.index(tile_sn)
                            x_positions.append(tile_idx)
                        
                        y_values = list(channel_data['Tuning_Efficiency_nm_per_C'])
                        
                        ax.scatter(x_positions, y_values,
                                 color=self.colors[color_idx], label=f'Ch{channel}', 
                                 s=50, alpha=0.8, zorder=10)
                
                # Add average annotations at 0.12 nm/C for each tile
                for i, (tile_sn, avg_value) in enumerate(zip(unique_tiles, tile_averages)):
                    ax.annotate(f'{avg_value:.3f}', 
                               xy=(i, 0.12),
                               xytext=(i, 0.12),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold', color='red',
                               rotation=90)
                
                ax.set_xlabel('Tile Serial Number')
                ax.set_ylabel('Tuning Efficiency (nm/°C)')
                ax.set_title(f'Bank {bank}')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.05, 0.15)  # Set y-axis scale from 0.05 to 0.15 nm/C
                
                # Set x-axis labels to tile serial numbers
                ax.set_xticks(range(len(unique_tiles)))
                ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
                
            plt.tight_layout()
            
            # Save the plot
            plot_filename = "tp2p2_wavelength_tuning_summary.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Wavelength tuning summary saved: {plot_filename}")
            plt.close()
            
            # Save tuning data
            tuning_csv_path = self.data_dir / "tp2p2_wavelength_tuning.csv"
            tuning_df.to_csv(tuning_csv_path, index=False)
            print(f"✅ Wavelength tuning data saved: {tuning_csv_path}")
            
        else:
            print("No tuning data available for summary plot")
            
    def analyze(self) -> Dict[str, Any]:
        """
        Perform the combined analysis
        
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        results = {}
        
        if self.temp_scan_data is not None:
            results['total_tiles'] = len(self.temp_scan_data['Tile_SN'].unique())
            results['total_measurements'] = len(self.temp_scan_data)
            results['temperature_range'] = {
                'min': self.temp_scan_data['T_PIC_C'].min(),
                'max': self.temp_scan_data['T_PIC_C'].max()
            }
            results['wavelength_range'] = {
                'min': self.temp_scan_data['PeakWave_nm'].min(),
                'max': self.temp_scan_data['PeakWave_nm'].max()
            }
            
        self.results = results
        return results
        
    def visualize(self) -> None:
        """
        Visualize the analysis results
        """
        print("Creating TP2-2 analysis visualizations...")
        self.plot_temperature_scan_per_tile()
        self.plot_wavelength_tuning_summary()
        print("✅ All visualizations completed")
        
    def export_results(self, filepath: str) -> None:
        """
        Export analysis results to file
        
        Parameters
        ----------
        filepath : str
            Path to save results
        """
        if self.temp_scan_data is not None:
            export_path = Path(filepath)
            self.temp_scan_data.to_csv(export_path, index=False)
            print(f"✅ Results exported to: {export_path}")
        else:
            print("No data to export")
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics and results
        """
        summary = {
            'analysis_type': 'TP2-2 Temperature Scan',
            'data_loaded': self.temp_scan_data is not None,
            'tiles_analyzed': len(self.tile_metadata),
            'files_processed': len(self.temp_scan_files)
        }
        
        if self.temp_scan_data is not None:
            summary.update({
                'total_measurements': len(self.temp_scan_data),
                'unique_tiles': len(self.temp_scan_data['Tile_SN'].unique()),
                'temperature_range_C': f"{self.temp_scan_data['T_PIC_C'].min():.1f} - {self.temp_scan_data['T_PIC_C'].max():.1f}",
                'wavelength_range_nm': f"{self.temp_scan_data['PeakWave_nm'].min():.2f} - {self.temp_scan_data['PeakWave_nm'].max():.2f}"
            })
            
        return summary
        
    def run_all(self) -> None:
        """Run complete analysis pipeline"""
        print("=" * 80)
        print("TP2-2 TEMPERATURE SCAN DATA ANALYSIS")
        print("=" * 80)
        print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)
        
        # Load and process data
        self.load_data()
        self.preprocess()
        
        # Perform analysis
        results = self.analyze()
        print("\nAnalysis Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
        # Create visualizations
        self.visualize()
        
        # Export processed data
        if self.temp_scan_data is not None:
            self.export_results(self.data_dir / "tp2p2_processed_data.csv")
            
        print("\n" + "=" * 80)
        print("TP2-2 ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    def __str__(self) -> str:
        """String representation of the analysis class"""
        return f"TP2P2CombinerAnalyzers - Data: {self.temp_scan_data is not None}"
        
    def __repr__(self) -> str:
        """Representation of the analysis class"""
        return f"TP2P2CombinerAnalyzers(data={self.temp_scan_data is not None})"


def main():
    """Main function to run TP2-2 analysis"""
    analyzer = TP2P2CombinerAnalyzers()
    analyzer.run_all()


if __name__ == "__main__":
    main() 