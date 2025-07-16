"""
TP2P0 - Lensing Station Data Extractor
=====================================

This script searches for the latest xlsx file in the lensing station folder,
identifies all sheets in the Excel file (which are TileSN), and extracts
data from Test Station sections with AWG 50C for both:

1. Single Channel ON: Coupling loss and wavelength data (per channel)
2. ALL Channel ON: Wavelength and power data (per bank)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import glob
from wavelength_grid_utils import load_wavelength_grid, get_channel_value


class LensingStationDataExtractor:
    """
    Extracts TileSN data from the latest Excel file in the lensing station folder
    """
    
    def __init__(self, lensing_station_path=None):
        """
        Initialize the lensing station data extractor
        
        Parameters
        ----------
        lensing_station_path : str, optional
            Path to the lensing station folder. If None, will search common locations.
        """
        self.lensing_station_path = self._find_lensing_station_path(lensing_station_path)
        self.latest_file = None
        self.tile_sns = []
        self.coupling_loss_data = pd.DataFrame()
        self.all_channel_data = pd.DataFrame()
        self.all_channel_bank_data = pd.DataFrame()
        
        # Create plots directory
        script_dir = Path(__file__).parent
        self.plots_dir = script_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def _find_lensing_station_path(self, provided_path):
        """
        Find the lensing station folder path
        
        Parameters
        ----------
        provided_path : str or None
            User-provided path to lensing station folder
            
        Returns
        -------
        Path
            Path to the lensing station folder
        """
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path
            else:
                raise FileNotFoundError(f"Provided lensing station path does not exist: {provided_path}")
        
        # Common locations to search for lensing station folder
        script_dir = Path(__file__).parent
        search_paths = [
            script_dir / "lensing_station",
            script_dir / "../lensing_station", 
            script_dir / "../data/lensing_station",
            script_dir / "../../lensing_station",
            script_dir / "../Lensing_Station",
            script_dir / "../LENSING_STATION",
            script_dir / "../../LM/lensing station",  # Add the actual location
            script_dir / "../LM/lensing station"     # Alternative location
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        # If not found, return a default path and let user know
        default_path = script_dir / "../lensing_station"
        print(f"⚠️  Lensing station folder not found. Using default path: {default_path}")
        print("   Please ensure the folder exists or provide the correct path.")
        return default_path
    
    def find_latest_xlsx_file(self):
        """
        Find the latest xlsx file in the lensing station folder
        
        Returns
        -------
        Path or None
            Path to the latest xlsx file, or None if no files found
        """
        if not self.lensing_station_path.exists():
            print(f"❌ Lensing station folder does not exist: {self.lensing_station_path}")
            return None
        
        # Search for xlsx files, but exclude temporary files (starting with ~$)
        xlsx_files = [f for f in self.lensing_station_path.glob("*.xlsx") if not f.name.startswith("~$")]
        
        if not xlsx_files:
            print(f"❌ No xlsx files found in: {self.lensing_station_path}")
            return None
        
        # Sort by modification time to get the latest file
        latest_file = max(xlsx_files, key=lambda x: x.stat().st_mtime)
        
        self.latest_file = latest_file
        
        # Get file info
        file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        file_size = latest_file.stat().st_size
        
        print(f"✅ Found latest xlsx file: {latest_file.name}")
        print(f"   Modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Path: {latest_file}")
        
        return latest_file
    
    def extract_tile_sns(self):
        """
        Extract all TileSN (sheet names) from the latest xlsx file
        
        Returns
        -------
        list
            List of TileSN (sheet names)
        """
        if not self.latest_file:
            print("❌ No xlsx file loaded. Call find_latest_xlsx_file() first.")
            return []
        
        try:
            # Read the Excel file to get all sheet names
            excel_file = pd.ExcelFile(self.latest_file)
            all_sheet_names = excel_file.sheet_names
            
            # Filter out non-TileSN sheets
            # TileSN typically start with "Y" followed by numbers (e.g., Y25170076)
            tile_sns = []
            excluded_sheets = []
            
            for sheet_name in all_sheet_names:
                # Check if it looks like a TileSN (starts with Y followed by digits)
                # Strip whitespace first to handle cases like "Y25170106 "
                clean_name = sheet_name.strip()
                if clean_name.startswith('Y') and len(clean_name) > 1 and clean_name[1:].isdigit():
                    tile_sns.append(clean_name)
                else:
                    excluded_sheets.append(sheet_name)
            
            self.tile_sns = tile_sns
            
            print(f"✅ Found {len(all_sheet_names)} total sheets, {len(tile_sns)} are TileSN:")
            for i, tile_sn in enumerate(tile_sns, 1):
                print(f"   {i:2d}. {tile_sn}")
            
            if excluded_sheets:
                print(f"\n📋 Excluded {len(excluded_sheets)} non-TileSN sheets:")
                for sheet in excluded_sheets:
                    print(f"   • {sheet}")
            
            return tile_sns
            
        except Exception as e:
            print(f"❌ Error reading Excel file: {e}")
            return []
    
    def extract_coupling_loss_data(self):
        """
        Extract coupling loss data and WL data from Test Station Single Ch ON sections (AWG 50C only)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Tile_SN, Channel, Coupling_Loss_dB, WL_nm_Lens_coupling, Bank
        """
        if not self.latest_file or not self.tile_sns:
            print("❌ No data loaded. Run extract_tile_sns() first.")
            return pd.DataFrame()
        
        print("🔍 Extracting coupling loss and WL data from Test Station Single Ch ON sections...")
        
        excel_file = pd.ExcelFile(self.latest_file)
        all_data = []
        
        for tile_sn in self.tile_sns:
            try:
                # Handle trailing space issue
                sheet_name = tile_sn if tile_sn in excel_file.sheet_names else f"{tile_sn} "
                df = pd.read_excel(self.latest_file, sheet_name=sheet_name)
                
                # Search entire sheet for AWG 50C Test Station sections
                awg_50c_sections = []
                
                for row_idx in range(len(df)):
                    cell_content = str(df.iloc[row_idx, 0])  # Column 1
                    if ('Test station' in cell_content and 'Single' in cell_content and 
                        'AWG 50C' in cell_content):
                        awg_50c_sections.append(row_idx)
                
                if not awg_50c_sections:
                    print(f"⚠️  {tile_sn}: No Test Station Single Ch ON with AWG 50C found")
                    continue
                
                # Extract data from all AWG 50C sections found
                tile_data_count = 0
                for section_start in awg_50c_sections:
                    # Extract coupling loss data starting from the row after the header
                    for row_idx in range(section_start + 1, len(df)):
                        channel = str(df.iloc[row_idx, 1])  # Column 2 (Channel)
                        coupling_loss = str(df.iloc[row_idx, 7])  # Column 8 (coupling loss)
                        wl_lens_coupling = str(df.iloc[row_idx, 4])  # Column 5 (WL(nm)_Lens coupling)
                        
                        # Stop when we reach empty cells or non-channel data
                        if channel == 'nan' or not channel.startswith('CH'):
                            break
                        
                        if coupling_loss != 'nan' and wl_lens_coupling != 'nan':
                            # Extract channel number
                            ch_num = int(channel[2:])  # Remove 'CH' prefix
                            
                            # Determine bank and renumber channels
                            if 1 <= ch_num <= 8:
                                bank = 0
                                bank_channel = ch_num
                            elif 9 <= ch_num <= 16:
                                bank = 1
                                bank_channel = ch_num - 8  # Renumber 9-16 to 1-8
                            else:
                                continue  # Skip invalid channels
                            
                            all_data.append({
                                'Tile_SN': tile_sn,
                                'Original_Channel': channel,
                                'Channel': bank_channel,
                                'Bank': bank,
                                'Coupling_Loss_dB': abs(float(coupling_loss)),  # Take absolute value
                                'WL_nm_Lens_coupling': float(wl_lens_coupling)
                            })
                            tile_data_count += 1
                
                print(f"✅ {tile_sn}: Extracted {tile_data_count} measurements from {len(awg_50c_sections)} AWG 50C section(s)")
                
            except Exception as e:
                print(f"❌ {tile_sn}: Error - {e}")
                continue
        
        self.coupling_loss_data = pd.DataFrame(all_data)
        
        if not self.coupling_loss_data.empty:
            print(f"\n🎉 Successfully extracted {len(self.coupling_loss_data)} measurements")
            print(f"   • {self.coupling_loss_data['Tile_SN'].nunique()} tiles")
            print(f"   • Bank 0: {len(self.coupling_loss_data[self.coupling_loss_data['Bank'] == 0])} measurements")
            print(f"   • Bank 1: {len(self.coupling_loss_data[self.coupling_loss_data['Bank'] == 1])} measurements")
        
        return self.coupling_loss_data
    
    def extract_all_channel_on_data(self):
        """
        Extract WL and Power data from Test Station ALL channel ON sections (AWG 50C only)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Tile_SN, Bank, WL_nm_All_Channel, Power_mW_Bank
        """
        if not self.latest_file or not self.tile_sns:
            print("❌ No data loaded. Run extract_tile_sns() first.")
            return pd.DataFrame()
        
        print("🔍 Extracting ALL channel ON data from Test Station sections...")
        
        excel_file = pd.ExcelFile(self.latest_file)
        all_data = []
        
        for tile_sn in self.tile_sns:
            try:
                # Handle trailing space issue
                sheet_name = tile_sn if tile_sn in excel_file.sheet_names else f"{tile_sn} "
                df = pd.read_excel(self.latest_file, sheet_name=sheet_name)
                
                # Search entire sheet for ALL channel ON AWG 50C Test Station sections
                all_channel_sections = []
                
                for row_idx in range(len(df)):
                    cell_content = str(df.iloc[row_idx, 0])  # Column 1
                    if ('Test station' in cell_content and 'ALL CH ON' in cell_content and 
                        'AWG 50C' in cell_content):
                        all_channel_sections.append(row_idx)
                
                if not all_channel_sections:
                    print(f"⚠️  {tile_sn}: No Test Station ALL channel ON with AWG 50C found")
                    continue
                
                # Extract data from ALL channel ON sections
                for section_start in all_channel_sections:
                    # Extract data from each channel row
                    for row_idx in range(section_start + 1, min(section_start + 20, len(df))):
                        channel = str(df.iloc[row_idx, 1])    # Column 2 (Channel)
                        power_value = str(df.iloc[row_idx, 2]) # Column 3 (Power mW)
                        wl_value = str(df.iloc[row_idx, 3])    # Column 4 (WL nm)
                        
                        # Stop when we reach empty cells or non-channel data
                        if channel == 'nan' or not channel.startswith('CH'):
                            break
                        
                        # Extract valid data
                        if wl_value != 'nan':
                            # Extract channel number to determine bank
                            ch_num = int(channel[2:])  # Remove 'CH' prefix
                            
                            # Determine bank: CH1-8 = Bank 0, CH9-16 = Bank 1
                            if 1 <= ch_num <= 8:
                                bank = 0
                            elif 9 <= ch_num <= 16:
                                bank = 1
                            else:
                                continue  # Skip invalid channels
                            
                            # For power, we need to sum all channels per bank
                            # But first, let's extract individual channel data
                            power_val = float(power_value) if power_value != 'nan' else 0.0
                            
                            all_data.append({
                                'Tile_SN': tile_sn,
                                'Channel': channel,
                                'Bank': bank,
                                'WL_nm_All_Channel': float(wl_value),
                                'Power_mW_Channel': power_val
                            })
                
                extracted_count = len([d for d in all_data if d['Tile_SN'] == tile_sn])
                if extracted_count > 0:
                    print(f"✅ {tile_sn}: Extracted {extracted_count} ALL channel measurements")
                
            except Exception as e:
                print(f"❌ {tile_sn}: Error - {e}")
                continue
        
        self.all_channel_data = pd.DataFrame(all_data)
        
        # Create bank-level power aggregation for plotting
        if not self.all_channel_data.empty:
            # Group by Tile_SN and Bank, sum the power
            bank_power_data = self.all_channel_data.groupby(['Tile_SN', 'Bank']).agg({
                'Power_mW_Channel': 'sum',
                'WL_nm_All_Channel': 'first'  # Take first WL value for each bank
            }).reset_index()
            
            # Rename columns for clarity
            bank_power_data = bank_power_data.rename(columns={
                'Power_mW_Channel': 'Power_mW_Bank',
                'WL_nm_All_Channel': 'WL_nm_Bank'
            })
            
            self.all_channel_bank_data = bank_power_data
            
            print(f"\n🎉 Successfully extracted {len(self.all_channel_data)} ALL channel measurements")
            print(f"   • {self.all_channel_data['Tile_SN'].nunique()} tiles")
            print(f"   • Bank 0: {len(self.all_channel_data[self.all_channel_data['Bank'] == 0])} measurements")
            print(f"   • Bank 1: {len(self.all_channel_data[self.all_channel_data['Bank'] == 1])} measurements")
        else:
            self.all_channel_bank_data = pd.DataFrame()
        
        return self.all_channel_data
    
    def plot_coupling_loss_vs_tile_combined(self):
        """
        Create coupling loss vs tile combined plot with scatter and boxplot
        """
        if self.coupling_loss_data.empty:
            print("❌ No coupling loss data available. Run extract_coupling_loss_data() first.")
            return
        
        print("📊 Creating coupling loss vs tile combined plot...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('Testing with Single Channel ON and AWG at 50C - Coupling Loss', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = self.coupling_loss_data[self.coupling_loss_data['Bank'] == 0]
        bank1_data = self.coupling_loss_data[self.coupling_loss_data['Bank'] == 1]
        
        # Plot both banks
        self._plot_bank_coupling_data(ax1, bank0_data, "Bank 0 (Channels 1-8)")
        self._plot_bank_coupling_data(ax2, bank1_data, "Bank 1 (Channels 9-16)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_single_channel_on_coupling_loss_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Coupling loss plot saved: {plot_path}")
        plt.close()
    
    def _plot_bank_coupling_data(self, ax, data, title):
        """
        Plot coupling loss data for a specific bank
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        data : pd.DataFrame
            Data for the bank
        title : str
            Title for the plot
        """
        if data.empty:
            ax.set_title(f"{title} - No Data")
            return
        
        # Get unique tiles and channels
        unique_tiles = sorted(data['Tile_SN'].unique())
        unique_channels = sorted(data['Channel'].unique())
        
        # Create a mapping from tile name to numeric position
        tile_to_position = {tile: i for i, tile in enumerate(unique_tiles)}
        
        # Colors for different channels
        colors = plt.colormaps['Set3'](np.linspace(0, 1, len(unique_channels)))
        
        # Create scatter plot for each channel using numeric positions
        for i, channel in enumerate(unique_channels):
            channel_data = data[data['Channel'] == channel]
            if not channel_data.empty:
                # Convert tile names to numeric positions
                x_positions = [tile_to_position[tile] for tile in channel_data['Tile_SN']]
                ax.scatter(x_positions, channel_data['Coupling_Loss_dB'], 
                          alpha=0.7, s=60, color=colors[i], 
                          label=f'CH{channel}', edgecolors='black', linewidth=0.5)
        
        # Add box plots for each tile
        box_data = []
        box_positions = []
        
        for i, tile in enumerate(unique_tiles):
            tile_data = data[data['Tile_SN'] == tile]
            if not tile_data.empty:
                box_data.append(tile_data['Coupling_Loss_dB'].values)
                box_positions.append(i)
        
        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, 
                           patch_artist=True)
            
            # Style the box plot
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.3)
        
        # Customize plot
        ax.set_xlabel('Tile Serial Number', fontsize=12)
        ax.set_ylabel('Coupling Loss (dB)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks and labels to match the numeric positions
        ax.set_xticks(range(len(unique_tiles)))
        ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
        
        # Add statistics text
        if not data.empty:
            mean_loss = data['Coupling_Loss_dB'].mean()
            std_loss = data['Coupling_Loss_dB'].std()
            min_loss = data['Coupling_Loss_dB'].min()
            max_loss = data['Coupling_Loss_dB'].max()
            
            stats_text = f'Mean: {mean_loss:.2f} dB\nStd: {std_loss:.2f} dB\nMin: {min_loss:.2f} dB\nMax: {max_loss:.2f} dB'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_wl_vs_tile_combined(self):
        """
        Create WL vs tile combined plot with scatter and boxplot
        """
        if self.coupling_loss_data.empty:
            print("❌ No WL data available. Run extract_coupling_loss_data() first.")
            return
        
        print("📊 Creating WL vs tile combined plot...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('Testing with Single Channel ON and AWG at 50C - Wavelength', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = self.coupling_loss_data[self.coupling_loss_data['Bank'] == 0]
        bank1_data = self.coupling_loss_data[self.coupling_loss_data['Bank'] == 1]
        
        # Plot both banks
        self._plot_bank_wl_data(ax1, bank0_data, "Bank 0 (Channels 1-8)")
        self._plot_bank_wl_data(ax2, bank1_data, "Bank 1 (Channels 9-16)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_single_channel_on_wl_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ WL plot saved: {plot_path}")
        plt.close()
    
    def _plot_bank_wl_data(self, ax, data, title):
        """
        Plot WL data for a specific bank
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        data : pd.DataFrame
            Data for the bank
        title : str
            Title for the plot
        """
        if data.empty:
            ax.set_title(f"{title} - No Data")
            return
        
        # Get unique tiles and channels
        unique_tiles = sorted(data['Tile_SN'].unique())
        unique_channels = sorted(data['Channel'].unique())
        
        # Create a mapping from tile name to numeric position
        tile_to_position = {tile: i for i, tile in enumerate(unique_tiles)}
        
        # Colors for different channels
        colors = plt.colormaps['Set3'](np.linspace(0, 1, len(unique_channels)))
        
        # Create scatter plot for each channel using numeric positions
        for i, channel in enumerate(unique_channels):
            channel_data = data[data['Channel'] == channel]
            if not channel_data.empty:
                # Convert tile names to numeric positions
                x_positions = [tile_to_position[tile] for tile in channel_data['Tile_SN']]
                ax.scatter(x_positions, channel_data['WL_nm_Lens_coupling'], 
                          alpha=0.7, s=60, color=colors[i], 
                          label=f'CH{channel}', edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Tile Serial Number', fontsize=12)
        ax.set_ylabel('WL(nm)_Lens coupling', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis range to 1300-1320 nm
        ax.set_ylim(1300, 1320)
        
        # Set x-axis ticks and labels to match the numeric positions
        ax.set_xticks(range(len(unique_tiles)))
        ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
    
    def plot_all_channel_wl_vs_tile(self):
        """
        Create WL vs tile plot for ALL channel ON data
        """
        if not hasattr(self, 'all_channel_data') or self.all_channel_data.empty:
            print("❌ No ALL channel ON data available. Run extract_all_channel_on_data() first.")
            return
        
        print("📊 Creating ALL channel ON WL vs tile plot...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('Testing with ALL Channel ON and AWG at 50C - Wavelength', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = self.all_channel_data[self.all_channel_data['Bank'] == 0]
        bank1_data = self.all_channel_data[self.all_channel_data['Bank'] == 1]
        
        # Plot both banks
        self._plot_all_channel_wl_data(ax1, bank0_data, "Bank 0")
        self._plot_all_channel_wl_data(ax2, bank1_data, "Bank 1")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_all_channel_on_wl_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ ALL channel WL plot saved: {plot_path}")
        plt.close()
    
    def _plot_all_channel_wl_data(self, ax, data, title):
        """
        Plot ALL channel WL data for a specific bank
        """
        if data.empty:
            ax.set_title(f"{title} - No Data")
            return
        
        # Get unique tiles and channels
        unique_tiles = sorted(data['Tile_SN'].unique())
        unique_channels = sorted(data['Channel'].unique())
        
        # Create a mapping from tile name to numeric position
        tile_to_position = {tile: i for i, tile in enumerate(unique_tiles)}
        
        # Colors for different channels
        colors = plt.colormaps['Set3'](np.linspace(0, 1, len(unique_channels)))
        
        # Create scatter plot for each channel using numeric positions
        for i, channel in enumerate(unique_channels):
            channel_data = data[data['Channel'] == channel]
            if not channel_data.empty:
                # Convert tile names to numeric positions
                x_positions = [tile_to_position[tile] for tile in channel_data['Tile_SN']]
                
                # Map bank1 legend labels from CH9-CH16 to CH1-CH8
                legend_label = channel
                if "Bank 1" in title and channel.startswith('CH'):
                    try:
                        ch_num = int(channel[2:])  # Extract channel number
                        if 9 <= ch_num <= 16:
                            legend_label = f"CH{ch_num - 8}"  # Convert CH9->CH1, CH10->CH2, etc.
                    except ValueError:
                        pass  # Keep original label if parsing fails
                
                ax.scatter(x_positions, channel_data['WL_nm_All_Channel'], 
                          alpha=0.7, s=60, color=colors[i], 
                          label=legend_label, edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Tile Serial Number', fontsize=12)
        ax.set_ylabel('WL(nm)_All Channel', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis range to 1300-1320 nm
        ax.set_ylim(1300, 1320)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(unique_tiles)))
        ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
    
    def plot_all_channel_power_vs_tile(self):
        """
        Create Power vs tile plot for ALL channel ON data (bank-level, merged)
        """
        if not hasattr(self, 'all_channel_bank_data') or self.all_channel_bank_data.empty:
            print("❌ No ALL channel ON bank data available. Run extract_all_channel_on_data() first.")
            return
        
        print("📊 Creating ALL channel ON Power vs tile plot...")
        
        # Create figure with single plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Add a title for the figure
        fig.suptitle('Testing with ALL Channel ON and AWG at 50C - Power (Bank Total)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Get unique tiles
        unique_tiles = sorted(self.all_channel_bank_data['Tile_SN'].unique())
        
        # Create a mapping from tile name to numeric position
        tile_to_position = {tile: i for i, tile in enumerate(unique_tiles)}
        
        # Plot both banks with different colors
        bank0_data = self.all_channel_bank_data[self.all_channel_bank_data['Bank'] == 0]
        bank1_data = self.all_channel_bank_data[self.all_channel_bank_data['Bank'] == 1]
        
        if not bank0_data.empty:
            x_positions = [tile_to_position[tile] for tile in bank0_data['Tile_SN']]
            ax.scatter(x_positions, bank0_data['Power_mW_Bank'], 
                      alpha=0.7, s=80, color='darkblue', 
                      label='Bank 0', edgecolors='black', linewidth=0.5)
        
        if not bank1_data.empty:
            x_positions = [tile_to_position[tile] for tile in bank1_data['Tile_SN']]
            ax.scatter(x_positions, bank1_data['Power_mW_Bank'], 
                      alpha=0.7, s=80, color='darkred', 
                      label='Bank 1', edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Tile Serial Number', fontsize=12)
        ax.set_ylabel('Power (mW) - Bank Total', fontsize=12)
        ax.set_title('Combined Banks', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(unique_tiles)))
        ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_all_channel_on_power_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ ALL channel Power plot saved: {plot_path}")
        plt.close()
    
    def plot_single_channel_wl_error_vs_tile(self):
        """
        Create wavelength error vs tile plot for single channel ON data
        """
        if self.coupling_loss_data.empty:
            print("❌ No single channel data available. Run extract_coupling_loss_data() first.")
            return
        
        print("📊 Creating single channel wavelength error vs tile plot...")
        
        # Load wavelength grid for expected values
        try:
            grid_data = load_wavelength_grid()
        except Exception as e:
            print(f"❌ Could not load wavelength grid: {e}")
            return
        
        # Calculate wavelength errors
        error_data = []
        for _, row in self.coupling_loss_data.iterrows():
            bank = int(row['Bank'])
            channel = int(row['Channel'])  # This is already 1-8 for both banks
            measured_wl = float(row['WL_nm_Lens_coupling'])
            
            try:
                expected_wl = get_channel_value(bank, channel, 'wavelength', grid_data)
                wl_error = measured_wl - expected_wl
                
                error_data.append({
                    'Tile_SN': row['Tile_SN'],
                    'Bank': bank,
                    'Channel': channel,
                    'Original_Channel': row['Original_Channel'],
                    'WL_Error_nm': wl_error,
                    'Measured_WL': measured_wl,
                    'Expected_WL': expected_wl
                })
            except Exception as e:
                print(f"⚠️ Error calculating wavelength error for {row['Tile_SN']}, channel {channel}: {e}")
                continue
        
        if not error_data:
            print("❌ No wavelength error data calculated")
            return
        
        error_df = pd.DataFrame(error_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('Single Channel ON - Wavelength Error vs Expected Grid Values', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = error_df[error_df['Bank'] == 0]
        bank1_data = error_df[error_df['Bank'] == 1]
        
        # Plot both banks
        self._plot_error_data(ax1, bank0_data, "Bank 0 (Set A)", "WL_Error_nm", "Wavelength Error (nm)")
        self._plot_error_data(ax2, bank1_data, "Bank 1 (Set B)", "WL_Error_nm", "Wavelength Error (nm)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_single_channel_on_wl_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Single channel wavelength error plot saved: {plot_path}")
        plt.close()
    
    def plot_single_channel_freq_error_vs_tile(self):
        """
        Create frequency error vs tile plot for single channel ON data
        """
        if self.coupling_loss_data.empty:
            print("❌ No single channel data available. Run extract_coupling_loss_data() first.")
            return
        
        print("📊 Creating single channel frequency error vs tile plot...")
        
        # Load wavelength grid for expected values
        try:
            grid_data = load_wavelength_grid()
        except Exception as e:
            print(f"❌ Could not load wavelength grid: {e}")
            return
        
        # Calculate frequency errors
        error_data = []
        for _, row in self.coupling_loss_data.iterrows():
            bank = int(row['Bank'])
            channel = int(row['Channel'])  # This is already 1-8 for both banks
            measured_wl = float(row['WL_nm_Lens_coupling'])
            
            try:
                expected_freq = get_channel_value(bank, channel, 'frequency', grid_data)
                # Convert measured wavelength to frequency (c = 299792458 m/s)
                measured_freq = 299792458 / (measured_wl * 1e-9) / 1e12  # Convert to THz
                freq_error = measured_freq - expected_freq
                
                error_data.append({
                    'Tile_SN': row['Tile_SN'],
                    'Bank': bank,
                    'Channel': channel,
                    'Original_Channel': row['Original_Channel'],
                    'Freq_Error_THz': freq_error,
                    'Measured_Freq': measured_freq,
                    'Expected_Freq': expected_freq
                })
            except Exception as e:
                print(f"⚠️ Error calculating frequency error for {row['Tile_SN']}, channel {channel}: {e}")
                continue
        
        if not error_data:
            print("❌ No frequency error data calculated")
            return
        
        error_df = pd.DataFrame(error_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('Single Channel ON - Frequency Error vs Expected Grid Values', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = error_df[error_df['Bank'] == 0]
        bank1_data = error_df[error_df['Bank'] == 1]
        
        # Plot both banks
        self._plot_error_data(ax1, bank0_data, "Bank 0 (Set A)", "Freq_Error_THz", "Frequency Error (THz)")
        self._plot_error_data(ax2, bank1_data, "Bank 1 (Set B)", "Freq_Error_THz", "Frequency Error (THz)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_single_channel_on_freq_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Single channel frequency error plot saved: {plot_path}")
        plt.close()
    
    def plot_all_channel_wl_error_vs_tile(self):
        """
        Create wavelength error vs tile plot for ALL channel ON data
        """
        if not hasattr(self, 'all_channel_data') or self.all_channel_data.empty:
            print("❌ No ALL channel ON data available. Run extract_all_channel_on_data() first.")
            return
        
        print("📊 Creating ALL channel wavelength error vs tile plot...")
        
        # Load wavelength grid for expected values
        try:
            grid_data = load_wavelength_grid()
        except Exception as e:
            print(f"❌ Could not load wavelength grid: {e}")
            return
        
        # Calculate wavelength errors
        error_data = []
        for _, row in self.all_channel_data.iterrows():
            bank = int(row['Bank'])
            channel_str = str(row['Channel'])  # e.g., "CH1", "CH9", etc.
            measured_wl = float(row['WL_nm_All_Channel'])
            
            try:
                # Extract channel number from string like "CH1", "CH9"
                channel_num = int(channel_str[2:])
                
                # Convert to 1-8 range for grid lookup
                if bank == 0:
                    grid_channel = channel_num  # CH1-CH8 -> 1-8
                else:
                    grid_channel = channel_num - 8  # CH9-CH16 -> 1-8
                
                expected_wl = get_channel_value(bank, grid_channel, 'wavelength', grid_data)
                wl_error = measured_wl - expected_wl
                
                error_data.append({
                    'Tile_SN': row['Tile_SN'],
                    'Bank': bank,
                    'Channel': channel_str,
                    'Grid_Channel': grid_channel,
                    'WL_Error_nm': wl_error,
                    'Measured_WL': measured_wl,
                    'Expected_WL': expected_wl
                })
            except Exception as e:
                print(f"⚠️ Error calculating wavelength error for {row['Tile_SN']}, channel {channel_str}: {e}")
                continue
        
        if not error_data:
            print("❌ No wavelength error data calculated")
            return
        
        error_df = pd.DataFrame(error_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('ALL Channel ON - Wavelength Error vs Expected Grid Values', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = error_df[error_df['Bank'] == 0]
        bank1_data = error_df[error_df['Bank'] == 1]
        
        # Plot both banks
        self._plot_error_data(ax1, bank0_data, "Bank 0 (Set A)", "WL_Error_nm", "Wavelength Error (nm)")
        self._plot_error_data(ax2, bank1_data, "Bank 1 (Set B)", "WL_Error_nm", "Wavelength Error (nm)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_all_channel_on_wl_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ ALL channel wavelength error plot saved: {plot_path}")
        plt.close()
    
    def plot_all_channel_freq_error_vs_tile(self):
        """
        Create frequency error vs tile plot for ALL channel ON data
        """
        if not hasattr(self, 'all_channel_data') or self.all_channel_data.empty:
            print("❌ No ALL channel ON data available. Run extract_all_channel_on_data() first.")
            return
        
        print("📊 Creating ALL channel frequency error vs tile plot...")
        
        # Load wavelength grid for expected values
        try:
            grid_data = load_wavelength_grid()
        except Exception as e:
            print(f"❌ Could not load wavelength grid: {e}")
            return
        
        # Calculate frequency errors
        error_data = []
        for _, row in self.all_channel_data.iterrows():
            bank = int(row['Bank'])
            channel_str = str(row['Channel'])  # e.g., "CH1", "CH9", etc.
            measured_wl = float(row['WL_nm_All_Channel'])
            
            try:
                # Extract channel number from string like "CH1", "CH9"
                channel_num = int(channel_str[2:])
                
                # Convert to 1-8 range for grid lookup
                if bank == 0:
                    grid_channel = channel_num  # CH1-CH8 -> 1-8
                else:
                    grid_channel = channel_num - 8  # CH9-CH16 -> 1-8
                
                expected_freq = get_channel_value(bank, grid_channel, 'frequency', grid_data)
                # Convert measured wavelength to frequency (c = 299792458 m/s)
                measured_freq = 299792458 / (measured_wl * 1e-9) / 1e12  # Convert to THz
                freq_error = measured_freq - expected_freq
                
                error_data.append({
                    'Tile_SN': row['Tile_SN'],
                    'Bank': bank,
                    'Channel': channel_str,
                    'Grid_Channel': grid_channel,
                    'Freq_Error_THz': freq_error,
                    'Measured_Freq': measured_freq,
                    'Expected_Freq': expected_freq
                })
            except Exception as e:
                print(f"⚠️ Error calculating frequency error for {row['Tile_SN']}, channel {channel_str}: {e}")
                continue
        
        if not error_data:
            print("❌ No frequency error data calculated")
            return
        
        error_df = pd.DataFrame(error_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Add a common title for the entire figure
        fig.suptitle('ALL Channel ON - Frequency Error vs Expected Grid Values', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Filter data for Bank 0 and Bank 1
        bank0_data = error_df[error_df['Bank'] == 0]
        bank1_data = error_df[error_df['Bank'] == 1]
        
        # Plot both banks
        self._plot_error_data(ax1, bank0_data, "Bank 0 (Set A)", "Freq_Error_THz", "Frequency Error (THz)")
        self._plot_error_data(ax2, bank1_data, "Bank 1 (Set B)", "Freq_Error_THz", "Frequency Error (THz)")
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.plots_dir / "tp2p0_all_channel_on_freq_error_vs_tile_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ ALL channel frequency error plot saved: {plot_path}")
        plt.close()
    
    def _plot_error_data(self, ax, data, title, error_column, ylabel):
        """
        Plot error data for a specific bank
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        data : pd.DataFrame
            Error data for the bank
        title : str
            Title for the plot
        error_column : str
            Column name containing error values
        ylabel : str
            Y-axis label
        """
        if data.empty:
            ax.set_title(f"{title} - No Data")
            return
        
        # Get unique tiles and channels
        unique_tiles = sorted(data['Tile_SN'].unique())
        unique_channels = sorted(data['Channel'].unique()) if 'Channel' in data.columns else sorted(data['Grid_Channel'].unique())
        
        # Create a mapping from tile name to numeric position
        tile_to_position = {tile: i for i, tile in enumerate(unique_tiles)}
        
        # Colors for different channels
        colors = plt.colormaps['Set3'](np.linspace(0, 1, len(unique_channels)))
        
        # Create scatter plot for each channel using numeric positions
        for i, channel in enumerate(unique_channels):
            if 'Channel' in data.columns:
                channel_data = data[data['Channel'] == channel]
                legend_label = channel
                # Map bank1 legend labels from CH9-CH16 to CH1-CH8
                if "Bank 1" in title and isinstance(channel, str) and channel.startswith('CH'):
                    try:
                        ch_num = int(channel[2:])
                        if 9 <= ch_num <= 16:
                            legend_label = f"CH{ch_num - 8}"
                    except ValueError:
                        pass
            else:
                channel_data = data[data['Grid_Channel'] == channel]
                legend_label = f'CH{channel}'
            
            if not channel_data.empty:
                # Convert tile names to numeric positions
                x_positions = [tile_to_position[tile] for tile in channel_data['Tile_SN']]
                ax.scatter(x_positions, channel_data[error_column], 
                          alpha=0.7, s=60, color=colors[i], 
                          label=legend_label, edgecolors='black', linewidth=0.5)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Expected Value')
        
        # Customize plot
        ax.set_xlabel('Tile Serial Number', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(unique_tiles)))
        ax.set_xticklabels(unique_tiles, rotation=45, ha='right')
        
        # Add statistics text
        if not data.empty:
            mean_error = data[error_column].mean()
            std_error = data[error_column].std()
            min_error = data[error_column].min()
            max_error = data[error_column].max()
            
            if 'nm' in ylabel:
                stats_text = f'Mean: {mean_error:.3f} nm\nStd: {std_error:.3f} nm\nMin: {min_error:.3f} nm\nMax: {max_error:.3f} nm'
            else:
                stats_text = f'Mean: {mean_error:.3f} THz\nStd: {std_error:.3f} THz\nMin: {min_error:.3f} THz\nMax: {max_error:.3f} THz'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def get_tile_sns(self):
        """
        Get the stored TileSN list
        
        Returns
        -------
        list
            List of TileSN
        """
        return self.tile_sns
    
    def run(self):
        """
        Run the complete extraction process
        
        Returns
        -------
        list
            List of TileSN extracted from the latest xlsx file
        """
        print("=" * 60)
        print("TP2P0 - LENSING STATION DATA EXTRACTOR")
        print("=" * 60)
        print(f"Search Path: {self.lensing_station_path}")
        print("-" * 60)
        
        # Find the latest xlsx file
        latest_file = self.find_latest_xlsx_file()
        if not latest_file:
            return []
        
        print("-" * 60)
        
        # Extract TileSN from sheet names
        tile_sns = self.extract_tile_sns()
        
        print("-" * 60)
        print(f"✅ Extraction complete. Found {len(tile_sns)} TileSN.")
        
        # Print detailed summary
        if tile_sns:
            print("\n" + "=" * 60)
            print("EXTRACTED TILESN SUMMARY")
            print("=" * 60)
            print(f"📄 Source File: {latest_file.name}")
            print(f"📊 Total Number of Tiles: {len(tile_sns)}")
            print("-" * 60)
            print("📋 All TileSN Found:")
            print("-" * 60)
            
            # Print TileSN in a formatted way (5 per row for better readability)
            for i in range(0, len(tile_sns), 5):
                row_tiles = tile_sns[i:i+5]
                formatted_row = "   ".join(f"{tile_sn:>12}" for tile_sn in row_tiles)
                print(f"   {formatted_row}")
            
            print("-" * 60)
            print(f"✅ TOTAL TILES EXTRACTED: {len(tile_sns)}")
        
        print("=" * 60)
        
        return tile_sns
    
    def run_coupling_analysis(self):
        """
        Run complete coupling loss and ALL channel analysis pipeline
        """
        print("=" * 80)
        print("TP2P0 - COUPLING LOSS & WAVELENGTH ANALYSIS")
        print("Test Station Single Ch ON and ALL Ch ON (AWG 50C)")
        print("=" * 80)
        
        # Extract TileSN
        tile_sns = self.run()
        
        if not tile_sns:
            print("❌ No TileSN found. Cannot proceed with analysis.")
            return
        
        print("\n" + "=" * 80)
        print("SINGLE CHANNEL ON - DATA EXTRACTION")
        print("=" * 80)
        
        # Extract single channel data
        single_channel_data = self.extract_coupling_loss_data()
        
        print("\n" + "=" * 80)
        print("ALL CHANNEL ON - DATA EXTRACTION")
        print("=" * 80)
        
        # Extract ALL channel data
        all_channel_data = self.extract_all_channel_on_data()
        
        print("\n" + "=" * 80)
        print("CREATING PLOTS")
        print("=" * 80)
        
        # Create Single Channel ON plots
        if not single_channel_data.empty:
            print("\n--- Single Channel ON Plots ---")
            self.plot_coupling_loss_vs_tile_combined()
            self.plot_wl_vs_tile_combined()
            
            print("\n--- Single Channel ON Error Plots ---")
            self.plot_single_channel_wl_error_vs_tile()
            self.plot_single_channel_freq_error_vs_tile()
        
        # Create ALL Channel ON plots
        if not all_channel_data.empty:
            print("\n--- ALL Channel ON Plots ---")
            self.plot_all_channel_wl_vs_tile()
            self.plot_all_channel_power_vs_tile()
            
            print("\n--- ALL Channel ON Error Plots ---")
            self.plot_all_channel_wl_error_vs_tile()
            self.plot_all_channel_freq_error_vs_tile()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Summary
        if not single_channel_data.empty:
            print(f"✅ Single Channel ON: {len(single_channel_data)} measurements")
            print(f"   • Coupling loss plot: {self.plots_dir / 'tp2p0_single_channel_on_coupling_loss_vs_tile_combined.png'}")
            print(f"   • WL plot: {self.plots_dir / 'tp2p0_single_channel_on_wl_vs_tile_combined.png'}")
            print(f"   • WL error plot: {self.plots_dir / 'tp2p0_single_channel_on_wl_error_vs_tile_combined.png'}")
            print(f"   • Frequency error plot: {self.plots_dir / 'tp2p0_single_channel_on_freq_error_vs_tile_combined.png'}")
        
        if not all_channel_data.empty:
            print(f"✅ ALL Channel ON: {len(all_channel_data)} measurements")
            print(f"   • WL plot: {self.plots_dir / 'tp2p0_all_channel_on_wl_vs_tile_combined.png'}")
            print(f"   • Power plot: {self.plots_dir / 'tp2p0_all_channel_on_power_vs_tile_combined.png'}")
            print(f"   • WL error plot: {self.plots_dir / 'tp2p0_all_channel_on_wl_error_vs_tile_combined.png'}")
            print(f"   • Frequency error plot: {self.plots_dir / 'tp2p0_all_channel_on_freq_error_vs_tile_combined.png'}")
        
        if single_channel_data.empty and all_channel_data.empty:
            print("❌ No data extracted from either Single Channel ON or ALL Channel ON sections.")


def main():
    """
    Main function to run the lensing station data extractor
    """
    # You can specify a custom path here if needed
    # extractor = LensingStationDataExtractor("/path/to/lensing/station")
    extractor = LensingStationDataExtractor()
    
    # Run the coupling analysis
    extractor.run_coupling_analysis()


if __name__ == "__main__":
    main() 