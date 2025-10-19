"""
CLM Manufacturing Data Analysis Script
Test Point Analysis for clm_mfg_data_v1 and clm_mfg_data_v2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


class tpanalysis:
    """
    Test Point Analysis class for CLM manufacturing data.
    Contains methods for analyzing data from each test point (TP).
    """
    
    def __init__(self, base_path=None):
        """Initialize the test point analysis class."""
        if base_path is None:
            # Default to the parent directory of analysis_src
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.v1_path = self.base_path / "clm_mfg_data_v1"
        self.v2_path = self.base_path / "clm_mfg_data_v2"
        self.results_path = self.base_path / "analysis_results"
        
        # Create results directory if it doesn't exist
        self.results_path.mkdir(exist_ok=True)
        
        # Load filter criteria
        import yaml
        filter_file = self.base_path / "analysis_src" / "filter.yaml"
        with open(filter_file, 'r') as f:
            self.filters = yaml.safe_load(f)['filters']
        
        print(f"Loaded filters: Power >{self.filters['optical_power']['min_mw']}mW, "
              f"Freq error: {self.filters['frequency_error']['min_ghz']} to "
              f"{self.filters['frequency_error']['max_ghz']} GHz")
    
    # TP1 Series Methods
    def tp1p1_analysis(self):
        """Analysis for TP1-1 test point data."""
        pass
    
    def tp1p2_analysis(self):
        """Analysis for TP1-2 test point data - LIV curves."""
        print("Starting TP1-2 LIV Analysis...")
        print("Applying filters from filter.yaml...")
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load LIV data for all valid tiles
        liv_data_v1 = self._load_tp1p2_liv_data(self.v1_path / "TP1-2", valid_tiles['v1'], 'v1')
        liv_data_v2 = self._load_tp1p2_liv_data(self.v2_path / "TP1-2", valid_tiles['v2'], 'v2')
        
        # Combine v1 and v2 data
        all_liv_data = pd.concat([liv_data_v1, liv_data_v2], ignore_index=True)
        
        if not all_liv_data.empty:
            # Create LIV overlay plot with insets
            output_path_summary = self.results_path / "LIV" / "tp1p2_liv_summary.png"
            self._plot_tp1p2_liv_overlay(all_liv_data, output_path_summary)
            print(f"Plot saved to: {output_path_summary}")
            
            # Create simple LIV plot without insets
            output_path_simple = self.results_path / "LIV" / "tp1p2_liv.png"
            self._plot_tp1p2_liv_simple(all_liv_data, output_path_simple)
            print(f"Plot saved to: {output_path_simple}")
            
            print("TP1-2 LIV Analysis completed!")
        else:
            print("No LIV data available for valid tiles")
        
        print("\n" + "="*60 + "\n")
    
    def tp1p3_analysis(self):
        """Analysis for TP1-3 test point data."""
        pass
    
    def tp1p4_analysis(self):
        """Analysis for TP1-4 test point data."""
        pass
    
    # TP2 Series Methods
    def tp2p1_analysis(self):
        """Analysis for TP2-1 test point data."""
        pass
    
    def tp2p2_analysis(self):
        """Analysis for TP2-2 test point data."""
        pass
    
    def tp2p3_analysis(self):
        """Analysis for TP2-3 test point data."""
        pass
    
    def tp2p4_analysis(self):
        """Analysis for TP2-4 test point data with filtering."""
        print("Starting TP2-4 Analysis...")
        print("Applying filters from filter.yaml...")
        
        import yaml
        
        # Load wavelength grid
        grid_file = self.base_path / "analysis_src" / "wavelength_grid.yaml"
        with open(grid_file, 'r') as f:
            wl_grid = yaml.safe_load(f)
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load data from both versions with filtering
        all_data = []
        
        for version, version_path in [('v1', self.v1_path), ('v2', self.v2_path)]:
            df = self._load_tp2p4_data(version_path / "TP2-4", wl_grid, valid_tiles[version])
            if not df.empty:
                df['Version'] = version
                all_data.append(df)
                print(f"Loaded {len(df)} records from {version} ({len(df['Tile_SN'].unique())} tiles)")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            output_file = self.results_path / "tp2p4_freq_error_summary.png"
            self._plot_tp2p4_freq_error(combined_df, output_file)
            print("TP2-4 Analysis completed!")
            print(f"Plot saved to: {output_file}")
        else:
            print("No TP2-4 data found!")
        
        print("\n" + "="*60 + "\n")
    
    def _get_valid_tiles(self):
        """Get valid tiles that pass all filter criteria."""
        import yaml
        
        # Load wavelength grid for frequency calculations
        grid_file = self.base_path / "analysis_src" / "wavelength_grid.yaml"
        with open(grid_file, 'r') as f:
            wl_grid = yaml.safe_load(f)
        
        valid_tiles_dict = {}
        
        for version, version_path in [('v1', self.v1_path), ('v2', self.v2_path)]:
            # Load power data from TP2-6
            df_power = self._load_tp2p6_data(version_path / "TP2-6")
            
            # Apply power filter
            power_min = self.filters['optical_power']['min_mw']
            tile_min_power = df_power.groupby('Tile_SN')['Power(mW)'].min()
            tiles_pass_power = tile_min_power[tile_min_power >= power_min].index.tolist()
            
            # Apply total power filter from TP2-5
            df_totalpower = self._load_tp2p5_totalpower_data(version_path / "TP2-5", tiles_pass_power)
            
            if not df_totalpower.empty:
                total_power_min = self.filters['total_power']['min_mw']
                # Check if all banks for each tile meet the minimum total power
                tile_min_totalpower = df_totalpower.groupby('Tile_SN')['Total_Power_mW'].min()
                tiles_pass_totalpower = tile_min_totalpower[tile_min_totalpower >= total_power_min].index.tolist()
            else:
                tiles_pass_totalpower = tiles_pass_power
            
            # Load frequency data from TP2-5
            df_freq = self._load_tp2p5_data(version_path / "TP2-5", wl_grid, tiles_pass_totalpower)
            
            if not df_freq.empty:
                # Apply frequency filter
                freq_min = self.filters['frequency_error']['min_ghz']
                freq_max = self.filters['frequency_error']['max_ghz']
                
                # Check each tile - all measurements must be within range
                tile_freq_valid = df_freq.groupby('Tile_SN')['Frequency_Error_GHz'].apply(
                    lambda x: ((x >= freq_min) & (x <= freq_max)).all()
                )
                tiles_pass_freq = tile_freq_valid[tile_freq_valid].index.tolist()
            else:
                tiles_pass_freq = []
            
            # Load TP2-4 data to check channel spacing (use tiles that passed previous filters)
            df_tp2p4 = self._load_tp2p4_data(version_path / "TP2-4", wl_grid, tiles_pass_freq)
            
            if not df_tp2p4.empty:
                # Add version column for grouping
                df_tp2p4['Version'] = version
                
                # Calculate channel spacing errors
                _, df_spacing = self._calculate_center_freq_spacing_errors(df_tp2p4, wl_grid)
                
                if not df_spacing.empty:
                    # Apply channel spacing error filter
                    spacing_max_abs = self.filters['channel_spacing_error']['max_abs_ghz']
                    
                    # Check each tile - all spacing measurements must be within ±max_abs
                    tile_spacing_valid = df_spacing.groupby('Tile_SN')['Spacing_Error_GHz'].apply(
                        lambda x: (np.abs(x) <= spacing_max_abs).all()
                    )
                    tiles_pass_spacing = tile_spacing_valid[tile_spacing_valid].index.tolist()
                else:
                    tiles_pass_spacing = tiles_pass_freq
            else:
                tiles_pass_spacing = tiles_pass_freq
            
            valid_tiles_dict[version] = tiles_pass_spacing
            print(f"  {version}: {len(tiles_pass_power)} tiles passed power filter, "
                  f"{len(tiles_pass_totalpower)} passed total power filter, "
                  f"{len(tiles_pass_freq)} passed freq filter, "
                  f"{len(tiles_pass_spacing)} passed all filters")
        
        return valid_tiles_dict
    
    def tp2p5_analysis(self):
        """Analysis for TP2-5 test point data - Frequency Error."""
        print("Starting TP2-5 Analysis...")
        print("Applying filters from filter.yaml...")
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load wavelength grid
        import yaml
        grid_file = self.base_path / "analysis_src" / "wavelength_grid.yaml"
        with open(grid_file, 'r') as f:
            wl_grid = yaml.safe_load(f)
        
        # Load TP2-5 data with the valid tiles
        df_v1 = self._load_tp2p5_data(self.v1_path / "TP2-5", wl_grid, valid_tiles['v1'])
        df_v2 = self._load_tp2p5_data(self.v2_path / "TP2-5", wl_grid, valid_tiles['v2'])
        
        print(f"Loaded {len(df_v1)} records from v1")
        print(f"Loaded {len(df_v2)} records from v2")
        
        # Create combined plot only
        df_v1['Version'] = 'v1'
        df_v2['Version'] = 'v2'
        df_combined = pd.concat([df_v1, df_v2], ignore_index=True)
        self._plot_tp2p5_freq_error(df_combined, self.results_path / "tp2p5_freq_error_summary.png")
        
        print("TP2-5 Analysis completed!")
        print(f"Plot saved to: {self.results_path / 'tp2p5_freq_error_summary.png'}")
    
    def tp2p5_totalpower_analysis(self):
        """Analysis for TP2-5 test point data - Average Power per Channel at T_MUX=50C."""
        print("Starting TP2-5 Total Power Analysis...")
        print("Applying filters from filter.yaml...")
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load TP2-5 total power data with the valid tiles
        df_v1 = self._load_tp2p5_totalpower_data(self.v1_path / "TP2-5", valid_tiles['v1'])
        df_v2 = self._load_tp2p5_totalpower_data(self.v2_path / "TP2-5", valid_tiles['v2'])
        
        print(f"Loaded {len(df_v1)} records from v1 ({len(df_v1['Tile_SN'].unique())} tiles)")
        print(f"Loaded {len(df_v2)} records from v2 ({len(df_v2['Tile_SN'].unique())} tiles)")
        
        # Create combined plot
        df_v1['Version'] = 'v1'
        df_v2['Version'] = 'v2'
        df_combined = pd.concat([df_v1, df_v2], ignore_index=True)
        self._plot_tp2p5_totalpower(df_combined, self.results_path / "tp2p5_totalpower_summary.png")
        
        print("TP2-5 Total Power Analysis completed!")
        print(f"Plot saved to: {self.results_path / 'tp2p5_totalpower_summary.png'}")
    
    def tp2p6_analysis(self):
        """Analysis for TP2-6 test point data."""
        print("Starting TP2-6 Analysis...")
        print("Applying filters from filter.yaml...")
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load data from both versions with valid tiles only
        df_v1 = self._load_tp2p6_data(self.v1_path / "TP2-6")
        df_v2 = self._load_tp2p6_data(self.v2_path / "TP2-6")
        
        # Filter to valid tiles
        df_v1 = df_v1[df_v1['Tile_SN'].isin(valid_tiles['v1'])].copy()
        df_v2 = df_v2[df_v2['Tile_SN'].isin(valid_tiles['v2'])].copy()
        
        print(f"Loaded {len(df_v1)} records from v1 ({len(valid_tiles['v1'])} tiles)")
        print(f"Loaded {len(df_v2)} records from v2 ({len(valid_tiles['v2'])} tiles)")
        
        # Create combined plot only
        df_v1['Version'] = 'v1'
        df_v2['Version'] = 'v2'
        df_combined = pd.concat([df_v1, df_v2], ignore_index=True)
        self._plot_tp2p6_power_combined(df_combined, self.results_path / "tp2p6_power_summary.png")
        
        print("TP2-6 Analysis completed!")
        print(f"Plot saved to: {self.results_path / 'tp2p6_power_summary.png'}")
    
    def _load_tp2p5_data(self, tp_path, wl_grid, valid_tiles):
        """Load all TP2-5 Scan.csv files and calculate frequency error."""
        all_data = []
        csv_files = sorted(glob.glob(str(tp_path / "*TP2-5 Scan.csv")))
        
        # Speed of light in nm/s
        c = 299792458 * 1e9  # m/s * 1e9 nm/m = nm/s
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Filter for valid tiles only
                df = df[df['Tile_SN'].isin(valid_tiles)].copy()
                
                if df.empty:
                    continue
                
                # Filter for T_MUX = 50C (±0.1C tolerance)
                df = df[(df['T_MUX(C)'] >= 49.9) & (df['T_MUX(C)'] <= 50.1)].copy()
                
                # Calculate frequency error
                def calc_freq_error(row):
                    bank = row['Bank']
                    channel = row['Channel']
                    measured_wl = row['OSA_Wave(nm)']
                    
                    # Get target wavelength from grid
                    bank_key = f'bank{bank}'
                    grid_num = channel + 1  # Grid numbering starts at 1
                    target_wl = wl_grid['banks'][bank_key]['grids'][grid_num]['wavelength_nm']
                    
                    # Wavelength error in nm
                    wl_error = measured_wl - target_wl
                    
                    # Convert to frequency error in GHz
                    # Δf = -(c/λ²) * Δλ
                    # c in nm/s, λ in nm, result in Hz, then convert to GHz
                    freq_error_hz = -(c / (target_wl ** 2)) * wl_error
                    freq_error_ghz = freq_error_hz / 1e9
                    
                    return freq_error_ghz
                
                df['Frequency_Error_GHz'] = df.apply(calc_freq_error, axis=1)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _load_tp2p6_data(self, tp_path):
        """Load all TP2-6 Test.csv files from a directory."""
        all_data = []
        csv_files = sorted(glob.glob(str(tp_path / "*TP2-6 Test.csv")))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Filter only rows where laser is on (Set Laser > 0)
                df = df[df['Set Laser(mA)'] > 0].copy()
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _plot_tp2p6_power_combined(self, df, output_path):
        """Create combined power summary plot for both v1 and v2."""
        if df.empty:
            print("No data available for combined plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
        
        # Define colors by bank (not by channel)
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}  # circle for bank 0, triangle for bank 1
        
        # Left subplot - scatter plot by tile (both v1 and v2, both banks combined)
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(df[df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(df[df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        tile_offset = 0
        for version in ['v1', 'v2']:
            df_version = df[df['Version'] == version]
            tiles = sorted(df_version['Tile_SN'].unique())
            
            for tile_idx, tile in enumerate(tiles):
                for bank in [0, 1]:
                    df_bank = df_version[df_version['Bank'] == bank]
                    df_tile = df_bank[df_bank['Tile_SN'] == tile]
                    
                    for channel in range(8):
                        df_channel = df_tile[df_tile['Channel'] == channel]
                        if not df_channel.empty:
                            powers = df_channel['Power(mW)'].values
                            
                            # Position: (tile_offset + tile_idx) * 17 + bank*8 + channel
                            pos = (tile_offset + tile_idx) * 17 + bank * 8 + channel
                            
                            # Add scatter points with bank colors and different markers
                            x_scatter = np.random.normal(pos, 0.15, size=len(powers))
                            ax_left.scatter(x_scatter, powers, color=bank_colors[bank], 
                                          alpha=0.7, s=35, marker=bank_markers[bank],
                                          edgecolors='black', linewidth=0.5)
            
            tile_offset += len(tiles)
        
        # Set x-axis labels to Tile_SN without version indicators
        tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
        ax_left.set_xticks(tile_positions)
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=7)
        
        # Labels and title
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        ax_left.set_ylabel('Power (mW)', fontsize=13, fontweight='bold')
        ax_left.set_title('Power Distribution by Tile', fontsize=14, fontweight='bold')
        ax_left.set_ylim(0, 20)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution (both v1 and v2, both banks combined)
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Collect data for violin and box plots
        box_data = []
        box_positions = []
        box_colors = []
        
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    powers = df_channel['Power(mW)'].values
                    y_pos = bank * 8 + channel
                    
                    box_data.append(powers)
                    box_positions.append(y_pos)
                    box_colors.append(bank_colors[bank])
        
        # Create horizontal violin plots
        parts = ax_right.violinplot(box_data, positions=box_positions, vert=False, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plots by bank
        for pc, color in zip(parts['bodies'], box_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay horizontal boxplots on top of violin plots
        bp = ax_right.boxplot(box_data, positions=box_positions, vert=False, widths=0.3,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                             whiskerprops=dict(color='black', linewidth=2),
                             capprops=dict(color='black', linewidth=2),
                             medianprops=dict(color='red', linewidth=2.5))
        
        # Add annotations at power = 18mW for median and 1-sigma
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    powers = df_channel['Power(mW)'].values
                    y_pos = bank * 8 + channel
                    
                    median = np.median(powers)
                    std = np.std(powers)
                    
                    # Annotate at x=18mW with units
                    annotation_text = f'μ̃={median:.1f}mW\nσ={std:.2f}mW'
                    ax_right.text(18.0, y_pos, annotation_text, 
                                fontsize=7, ha='left', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=bank_colors[bank], alpha=0.8, linewidth=1))
        
        # Calculate overall statistics across all data
        all_powers = df['Power(mW)'].values
        overall_median = np.median(all_powers)
        overall_std = np.std(all_powers)
        
        # Count total tiles
        total_tiles = len(all_tiles)
        
        # Y-axis labels showing both banks
        yticks = list(range(16))
        yticklabels = [f'B0-Ch{i}' for i in range(8)] + [f'B1-Ch{i}' for i in range(8)]
        ax_right.set_yticks(yticks)
        ax_right.set_yticklabels(yticklabels, fontsize=9)
        ax_right.set_xlabel('Power (mW)', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Bank-Channel', fontsize=12, fontweight='bold')
        
        # Title with overall statistics
        title_text = f'Statistical Distribution\nμ̃={overall_median:.2f}mW, σ={overall_std:.2f}mW'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='x')
        ax_right.set_ylim(-0.5, 15.5)
        ax_right.set_xlim(0, 20)  # Set x-axis limits
        
        # Add horizontal line to separate banks
        ax_right.axhline(y=7.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add bank markers and colors to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ])
        
        # Add legend to the left subplot
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _plot_tp2p5_freq_error(self, df, output_path):
        """Create frequency error summary plot for both v1 and v2."""
        if df.empty:
            print("No data available for combined plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
        
        # Define colors by bank
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}
        
        # Left subplot - wavelength error by tile
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(df[df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(df[df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        tile_offset = 0
        for version in ['v1', 'v2']:
            df_version = df[df['Version'] == version]
            tiles = sorted(df_version['Tile_SN'].unique())
            
            for tile_idx, tile in enumerate(tiles):
                for bank in [0, 1]:
                    df_bank = df_version[df_version['Bank'] == bank]
                    df_tile = df_bank[df_bank['Tile_SN'] == tile]
                    
                    for channel in range(8):
                        df_channel = df_tile[df_tile['Channel'] == channel]
                        if not df_channel.empty:
                            errors = df_channel['Frequency_Error_GHz'].values
                            
                            pos = (tile_offset + tile_idx) * 17 + bank * 8 + channel
                            
                            x_scatter = np.random.normal(pos, 0.15, size=len(errors))
                            ax_left.scatter(x_scatter, errors, color=bank_colors[bank], 
                                          alpha=0.7, s=35, marker=bank_markers[bank],
                                          edgecolors='black', linewidth=0.5)
            
            tile_offset += len(tiles)
        
        # Set x-axis labels to Tile_SN
        tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
        ax_left.set_xticks(tile_positions)
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=7)
        
        # Labels and title
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        ax_left.set_ylabel('Frequency Error (GHz)', fontsize=13, fontweight='bold')
        ax_left.set_title('Frequency Error by Tile', fontsize=14, fontweight='bold')
        ax_left.set_ylim(-50, 50)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Collect data for violin and box plots
        box_data = []
        box_positions = []
        box_colors = []
        
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    errors = df_channel['Frequency_Error_GHz'].values
                    y_pos = bank * 8 + channel
                    
                    box_data.append(errors)
                    box_positions.append(y_pos)
                    box_colors.append(bank_colors[bank])
        
        # Create horizontal violin plots
        parts = ax_right.violinplot(box_data, positions=box_positions, vert=False, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plots by bank
        for pc, color in zip(parts['bodies'], box_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay horizontal boxplots on top of violin plots
        bp = ax_right.boxplot(box_data, positions=box_positions, vert=False, widths=0.3,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                             whiskerprops=dict(color='black', linewidth=2),
                             capprops=dict(color='black', linewidth=2),
                             medianprops=dict(color='red', linewidth=2.5))
        
        # Add annotations for median and 1-sigma
        annotation_x = 30  # Position annotations at 30 GHz
        
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    errors = df_channel['Frequency_Error_GHz'].values
                    y_pos = bank * 8 + channel
                    
                    median = np.median(errors)
                    std = np.std(errors)
                    
                    annotation_text = f'μ̃={median:.2f}GHz\nσ={std:.2f}GHz'
                    ax_right.text(annotation_x, y_pos, annotation_text, 
                                fontsize=7, ha='left', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=bank_colors[bank], alpha=0.8, linewidth=1))
        
        # Calculate overall statistics
        all_errors = df['Frequency_Error_GHz'].values
        overall_median = np.median(all_errors)
        overall_std = np.std(all_errors)
        
        # Count total tiles
        total_tiles = len(all_tiles)
        
        # Y-axis labels showing both banks
        yticks = list(range(16))
        yticklabels = [f'B0-Ch{i}' for i in range(8)] + [f'B1-Ch{i}' for i in range(8)]
        ax_right.set_yticks(yticks)
        ax_right.set_yticklabels(yticklabels, fontsize=9)
        ax_right.set_xlabel('Frequency Error (GHz)', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Bank-Channel', fontsize=12, fontweight='bold')
        
        # Title with overall statistics
        title_text = f'Statistical Distribution\nμ̃={overall_median:.2f}GHz, σ={overall_std:.2f}GHz'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='x')
        ax_right.set_ylim(-0.5, 15.5)
        ax_right.set_xlim(-50, 50)  # Set x-axis limits
        
        # Add horizontal line to separate banks
        ax_right.axhline(y=7.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add bank markers and colors to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ])
        
        # Add legend to the left subplot
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _load_tp2p5_totalpower_data(self, tp_path, valid_tiles):
        """Load all TP2-5 Scan.csv files and calculate average power per channel per bank at T_MUX=50C."""
        all_data = []
        csv_files = sorted(glob.glob(str(tp_path / "*TP2-5 Scan.csv")))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Filter for valid tiles only
                df = df[df['Tile_SN'].isin(valid_tiles)].copy()
                
                if df.empty:
                    continue
                
                # Filter for T_MUX = 50C (±0.1C tolerance)
                df = df[(df['T_MUX(C)'] >= 49.9) & (df['T_MUX(C)'] <= 50.1)].copy()
                
                if df.empty:
                    continue
                
                # Calculate average power per tile per bank
                # Group by Tile_SN and Bank, average the Power(mW) across all 8 channels
                grouped = df.groupby(['Tile_SN', 'Bank'])['Power(mW)'].mean().reset_index()
                grouped.rename(columns={'Power(mW)': 'Total_Power_mW'}, inplace=True)
                
                all_data.append(grouped)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _plot_tp2p5_totalpower(self, df, output_path):
        """Create average power per channel summary plot for both v1 and v2."""
        if df.empty:
            print("No data available for combined plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 0.38], wspace=0.25)
        
        # Define colors by bank
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}
        
        # Left subplot - total power by tile
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(df[df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(df[df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        tile_offset = 0
        for version in ['v1', 'v2']:
            df_version = df[df['Version'] == version]
            tiles = sorted(df_version['Tile_SN'].unique())
            
            for tile_idx, tile in enumerate(tiles):
                for bank in [0, 1]:
                    df_tile_bank = df_version[(df_version['Tile_SN'] == tile) & (df_version['Bank'] == bank)]
                    
                    if not df_tile_bank.empty:
                        powers = df_tile_bank['Total_Power_mW'].values
                        
                        pos = (tile_offset + tile_idx) * 3 + bank
                        
                        x_scatter = np.random.normal(pos, 0.1, size=len(powers))
                        ax_left.scatter(x_scatter, powers, color=bank_colors[bank], 
                                      alpha=0.7, s=50, marker=bank_markers[bank],
                                      edgecolors='black', linewidth=0.5)
            
            tile_offset += len(tiles)
        
        # Set x-axis labels to Tile_SN
        tile_positions = [(i * 3 + 0.5) for i in range(len(all_tiles))]
        ax_left.set_xticks(tile_positions)
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=7)
        
        # Labels and title
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        ax_left.set_ylabel('Total Power in Fiber (mW)', fontsize=13, fontweight='bold')
        ax_left.set_title('Total Power in Fiber (mW)', fontsize=14, fontweight='bold')
        ax_left.set_ylim(0, 200)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution (vertical violin plot)
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Combine both banks for violin plot
        all_powers = df['Total_Power_mW'].values
        
        # Create vertical violin plot
        parts = ax_right.violinplot([all_powers], positions=[0], vert=True, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('purple')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay box plot on top of violin plot
        bp = ax_right.boxplot([all_powers], positions=[0], widths=0.3, 
                              patch_artist=True, showfliers=False,
                              boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                              whiskerprops=dict(color='black', linewidth=2),
                              capprops=dict(color='black', linewidth=2),
                              medianprops=dict(color='red', linewidth=2.5))
        
        # Calculate overall statistics
        overall_median = np.median(all_powers)
        overall_std = np.std(all_powers)
        
        # Count total tiles (unique Tile_SN)
        total_tiles = len(all_tiles)
        
        # Add annotation inside the subplot (upper right corner)
        annotation_y = all_powers.max() - (all_powers.max() - all_powers.min()) * 0.08
        annotation_text = f'μ̃={overall_median:.2f}mW\nσ={overall_std:.2f}mW'
        ax_right.text(0.3, annotation_y, annotation_text, 
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='purple', alpha=0.9, linewidth=2))
        
        # X-axis labels
        ax_right.set_xticks([0])
        ax_right.set_xticklabels(['Both Banks'], fontsize=11, fontweight='bold')
        ax_right.set_ylabel('Total Power in Fiber (mW)', fontsize=12, fontweight='bold')
        ax_right.set_xlabel('', fontsize=12, fontweight='bold')
        
        # Title
        title_text = f'Statistical Distribution'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.set_xlim(-0.5, 0.5)
        ax_right.set_ylim(0, 200)  # Set y-axis limits
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add bank markers and colors to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ])
        
        # Add legend to the left subplot
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _load_tp2p4_data(self, tp_path, wl_grid, valid_tiles):
        """Load all TP2-4 Scan.csv files and calculate frequency error with filtering."""
        all_data = []
        csv_files = sorted(glob.glob(str(tp_path / "*TP2-4 Scan.csv")))
        
        # Speed of light in nm/s
        c = 299792458 * 1e9  # m/s * 1e9 nm/m = nm/s
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Extract Tile_SN from filename (more reliable than Mmid column)
                # Filename format: YYYY-MM-DDTHH_MM_SS-TILE_SN-TP2-4 Scan.csv
                from pathlib import Path
                filename = Path(csv_file).stem  # Get filename without extension
                parts = filename.split('-')
                
                # Look for the tile SN part (starts with Y25, Y26, Y27, etc.)
                tile_sn = None
                for part in parts:
                    if part.startswith('Y25') or part.startswith('Y26') or part.startswith('Y27') or part.startswith('Y29') or part.startswith('Y2532') or part.startswith('Y2534'):
                        tile_sn = part
                        break
                
                if tile_sn is None:
                    print(f"Warning: Could not extract tile SN from filename {csv_file}")
                    continue
                
                df['Tile_SN'] = tile_sn
                
                # Filter for valid tiles only
                if tile_sn not in valid_tiles:
                    continue
                
                # Filter for T_MUX = 50C (±0.1C tolerance)
                df = df[(df['T_MUX(C)'] >= 49.9) & (df['T_MUX(C)'] <= 50.1)].copy()
                
                # Calculate frequency error
                def calc_freq_error(row):
                    bank = row['Bank']
                    channel = row['Channel']
                    measured_wl = row['OSA_Wave(nm)']
                    
                    # Get target wavelength from grid
                    bank_key = f'bank{bank}'
                    grid_num = channel + 1  # Grid numbering starts at 1
                    target_wl = wl_grid['banks'][bank_key]['grids'][grid_num]['wavelength_nm']
                    
                    # Calculate wavelength error
                    wl_error_nm = measured_wl - target_wl
                    
                    # Convert to frequency error using Δf = -(c/λ²) * Δλ
                    freq_error_hz = -(c / (target_wl ** 2)) * wl_error_nm
                    freq_error_ghz = freq_error_hz / 1e9  # Convert Hz to GHz
                    
                    return pd.Series({
                        'Wavelength_Error_nm': wl_error_nm,
                        'Frequency_Error_GHz': freq_error_ghz
                    })
                
                # Apply calculation
                df[['Wavelength_Error_nm', 'Frequency_Error_GHz']] = df.apply(calc_freq_error, axis=1)
                
                # Keep only the columns we need
                df = df[['Tile_SN', 'Bank', 'Channel', 'T_MUX(C)', 'OSA_Wave(nm)', 
                        'Wavelength_Error_nm', 'Frequency_Error_GHz']].copy()
                
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _plot_tp2p4_freq_error(self, df, output_path):
        """Create frequency error summary plot for TP2-4 (both v1 and v2, no filtering)."""
        if df.empty:
            print("No data available for TP2-4 plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
        
        # Define colors by bank
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}
        
        # Left subplot - scatter plot by tile
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(df[df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(df[df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        tile_offset = 0
        for version in ['v1', 'v2']:
            df_version = df[df['Version'] == version]
            tiles = sorted(df_version['Tile_SN'].unique())
            
            for tile_idx, tile in enumerate(tiles):
                for bank in [0, 1]:
                    df_bank = df_version[df_version['Bank'] == bank]
                    df_tile = df_bank[df_bank['Tile_SN'] == tile]
                    
                    for channel in range(8):
                        df_channel = df_tile[df_tile['Channel'] == channel]
                        if not df_channel.empty:
                            freq_errors = df_channel['Frequency_Error_GHz'].values
                            
                            # Position: (tile_offset + tile_idx) * 17 + bank*8 + channel
                            pos = (tile_offset + tile_idx) * 17 + bank * 8 + channel
                            
                            # Add scatter points
                            x_scatter = np.random.normal(pos, 0.15, size=len(freq_errors))
                            ax_left.scatter(x_scatter, freq_errors, color=bank_colors[bank], 
                                          alpha=0.7, s=35, marker=bank_markers[bank],
                                          edgecolors='black', linewidth=0.5)
            
            tile_offset += len(tiles)
        
        # Set x-axis labels
        tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
        ax_left.set_xticks(tile_positions)
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=7)
        
        # Labels and title
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        ax_left.set_ylabel('Frequency Error (GHz)', fontsize=13, fontweight='bold')
        ax_left.set_title('Frequency Error by Tile', fontsize=14, fontweight='bold')
        ax_left.set_ylim(-50, 50)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Collect data for violin and box plots
        box_data = []
        box_positions = []
        box_colors = []
        
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    freq_errors = df_channel['Frequency_Error_GHz'].values
                    y_pos = bank * 8 + channel
                    
                    box_data.append(freq_errors)
                    box_positions.append(y_pos)
                    box_colors.append(bank_colors[bank])
        
        # Create horizontal violin plots
        parts = ax_right.violinplot(box_data, positions=box_positions, vert=False, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plots by bank
        for pc, color in zip(parts['bodies'], box_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay horizontal boxplots on top of violin plots
        bp = ax_right.boxplot(box_data, positions=box_positions, vert=False, widths=0.3,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                             whiskerprops=dict(color='black', linewidth=2),
                             capprops=dict(color='black', linewidth=2),
                             medianprops=dict(color='red', linewidth=2.5))
        
        # Add annotations for median and 1-sigma
        annotation_x = 30  # Position annotations at 30 GHz
        
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            for channel in range(8):
                df_channel = df_bank[df_bank['Channel'] == channel]
                if not df_channel.empty:
                    freq_errors = df_channel['Frequency_Error_GHz'].values
                    y_pos = bank * 8 + channel
                    
                    median = np.median(freq_errors)
                    std = np.std(freq_errors)
                    
                    # Annotate at x=30 GHz with units
                    annotation_text = f'μ̃={median:.1f}GHz\nσ={std:.2f}GHz'
                    ax_right.text(annotation_x, y_pos, annotation_text, 
                                fontsize=7, ha='left', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=bank_colors[bank], alpha=0.8, linewidth=1))
        
        # Calculate overall statistics
        all_freq_errors = df['Frequency_Error_GHz'].values
        overall_median = np.median(all_freq_errors)
        overall_std = np.std(all_freq_errors)
        
        # Count total tiles
        total_tiles = len(all_tiles)
        
        # Y-axis labels
        yticks = list(range(16))
        yticklabels = [f'B0-Ch{i}' for i in range(8)] + [f'B1-Ch{i}' for i in range(8)]
        ax_right.set_yticks(yticks)
        ax_right.set_yticklabels(yticklabels, fontsize=9)
        ax_right.set_xlabel('Frequency Error (GHz)', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Bank-Channel', fontsize=12, fontweight='bold')
        
        # Title with overall statistics
        title_text = f'Statistical Distribution\nμ̃={overall_median:.2f}GHz, σ={overall_std:.2f}GHz'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='x')
        ax_right.set_ylim(-0.5, 15.5)
        ax_right.set_xlim(-50, 50)  # Set x-axis limits
        
        # Add horizontal line to separate banks
        ax_right.axhline(y=7.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add bank markers and colors to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ])
        
        # Add legend to the left subplot
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def center_freq_spacing_analysis(self):
        """Analysis for center frequency error and channel spacing error using TP2-4 and TP2-5 data."""
        print("Starting Center Frequency & Channel Spacing Analysis...")
        print("Applying filters from filter.yaml...")
        
        import yaml
        
        # Load wavelength grid
        grid_file = self.base_path / "analysis_src" / "wavelength_grid.yaml"
        with open(grid_file, 'r') as f:
            wl_grid = yaml.safe_load(f)
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Process both TP2-4 and TP2-5 data
        for tp_name, tp_folder in [('TP2-4', 'TP2-4'), ('TP2-5', 'TP2-5')]:
            print(f"\nProcessing {tp_name} data...")
            all_data = []
            
            for version, version_path in [('v1', self.v1_path), ('v2', self.v2_path)]:
                if tp_name == 'TP2-4':
                    df = self._load_tp2p4_data(version_path / tp_folder, wl_grid, valid_tiles[version])
                else:
                    df = self._load_tp2p5_data(version_path / tp_folder, wl_grid, valid_tiles[version])
                
                if not df.empty:
                    df['Version'] = version
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Calculate center frequency and channel spacing errors
                center_freq_df, spacing_df = self._calculate_center_freq_spacing_errors(combined_df, wl_grid)
                
                # Create plots
                output_file_center = self.results_path / f"{tp_folder.lower().replace('-', 'p')}_center_freq_error_summary.png"
                output_file_spacing = self.results_path / f"{tp_folder.lower().replace('-', 'p')}_channel_spacing_error_summary.png"
                
                self._plot_center_freq_error(center_freq_df, output_file_center)
                self._plot_channel_spacing_error(center_freq_df, spacing_df, output_file_spacing)
                
                print(f"  Saved: {output_file_center}")
                print(f"  Saved: {output_file_spacing}")
        
        print("\nCenter Frequency & Channel Spacing Analysis completed!")
        print("\n" + "="*60 + "\n")
    
    def _calculate_center_freq_spacing_errors(self, df, wl_grid):
        """Calculate center frequency error and channel spacing error for each tile-bank combination."""
        center_freq_results = []
        spacing_results = []
        
        # Speed of light in nm/s for frequency calculations
        c = 299792458 * 1e9  # m/s * 1e9 nm/m = nm/s
        
        # Group by Tile_SN, Version, and Bank
        for (tile_sn, version, bank), group in df.groupby(['Tile_SN', 'Version', 'Bank']):
            # Sort by channel
            group = group.sort_values('Channel')
            
            # Get target center frequency from grid
            bank_key = f'bank{bank}'
            target_center_freq_thz = wl_grid['banks'][bank_key]['center_frequency_thz']
            target_spacing_thz = wl_grid['banks'][bank_key]['channel_spacing_thz']
            
            # Convert measured wavelengths to frequencies (THz)
            wavelengths = group['OSA_Wave(nm)'].values
            channels = group['Channel'].values
            frequencies_thz = (c / wavelengths) / 1e12  # Convert Hz to THz
            
            # Calculate center frequency (average of all channels)
            measured_center_freq_thz = np.mean(frequencies_thz)
            center_freq_error_ghz = (measured_center_freq_thz - target_center_freq_thz) * 1000  # THz to GHz
            
            # Calculate channel spacings (differences between adjacent channels)
            if len(frequencies_thz) >= 2:
                spacings_thz = np.diff(frequencies_thz)
                # Note: spacing is negative because frequency decreases with increasing wavelength
                avg_spacing_thz = np.mean(spacings_thz)
                target_spacing_signed = -target_spacing_thz  # Target is -400 GHz (negative because freq decreases)
                spacing_error_ghz = (avg_spacing_thz - target_spacing_signed) * 1000  # THz to GHz
                spacing_std_ghz = np.std(spacings_thz) * 1000  # THz to GHz
                
                # Store individual channel-to-channel spacings
                for i in range(len(spacings_thz)):
                    spacing_ghz = spacings_thz[i] * 1000  # THz to GHz (keep signed value)
                    spacing_error = spacing_ghz - (target_spacing_signed * 1000)
                    spacing_results.append({
                        'Tile_SN': tile_sn,
                        'Version': version,
                        'Bank': bank,
                        'Channel_From': channels[i],
                        'Channel_To': channels[i+1],
                        'Spacing_GHz': spacing_ghz,
                        'Spacing_Error_GHz': spacing_error
                    })
            else:
                avg_spacing_thz = 0
                spacing_error_ghz = 0
                spacing_std_ghz = 0
            
            center_freq_results.append({
                'Tile_SN': tile_sn,
                'Version': version,
                'Bank': bank,
                'Target_Center_Freq_THz': target_center_freq_thz,
                'Measured_Center_Freq_THz': measured_center_freq_thz,
                'Center_Freq_Error_GHz': center_freq_error_ghz,
                'Target_Spacing_THz': target_spacing_thz,
                'Measured_Spacing_THz': avg_spacing_thz,
                'Spacing_Error_GHz': spacing_error_ghz,
                'Spacing_Std_GHz': spacing_std_ghz
            })
        
        return pd.DataFrame(center_freq_results), pd.DataFrame(spacing_results)
    
    def _plot_center_freq_error(self, df, output_path):
        """Create center frequency error summary plot with statistical distribution."""
        if df.empty:
            print("No data available for center frequency error plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 0.38], wspace=0.25)
        
        # Define colors and markers by bank
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}
        
        # Left subplot - scatter plot by tile
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(df[df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(df[df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        # Create x-axis positions
        tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}
        
        # Plot data points
        for bank in [0, 1]:
            df_bank = df[df['Bank'] == bank]
            
            x_pos = [tile_to_pos[tile] for tile in df_bank['Tile_SN']]
            y_vals = df_bank['Center_Freq_Error_GHz'].values
            
            ax_left.scatter(x_pos, y_vals, color=bank_colors[bank], marker=bank_markers[bank],
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.8,
                          label=f'Bank {bank}')
        
        # Set x-axis
        ax_left.set_xticks(range(len(all_tiles)))
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=8)
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        
        # Set y-axis
        ax_left.set_ylabel('Center Frequency Error (GHz)', fontsize=13, fontweight='bold')
        ax_left.set_title('Center Frequency Error by Tile', fontsize=14, fontweight='bold')
        ax_left.set_ylim(-50, 50)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution (vertical violin plot)
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Combine both banks for violin plot
        all_errors = df['Center_Freq_Error_GHz'].values
        
        # Create vertical violin plot
        parts = ax_right.violinplot([all_errors], positions=[0], vert=True, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('purple')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay box plot on top of violin plot
        bp = ax_right.boxplot([all_errors], positions=[0], widths=0.3, 
                              patch_artist=True, showfliers=False,
                              boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                              whiskerprops=dict(color='black', linewidth=2),
                              capprops=dict(color='black', linewidth=2),
                              medianprops=dict(color='red', linewidth=2.5))
        
        # Calculate overall statistics
        overall_mean = np.mean(all_errors)
        overall_std = np.std(all_errors)
        
        # Count total tiles
        total_tiles = len(all_tiles)
        
        # Add annotation inside the subplot (upper right corner)
        annotation_y = 20 - (20 - (-20)) * 0.08
        annotation_text = f'μ={overall_mean:.2f}GHz\nσ={overall_std:.2f}GHz'
        ax_right.text(0.3, annotation_y, annotation_text, 
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='purple', alpha=0.9, linewidth=2))
        
        # X-axis labels
        ax_right.set_xticks([0])
        ax_right.set_xticklabels(['Both Banks'], fontsize=11, fontweight='bold')
        ax_right.set_ylabel('Center Frequency Error (GHz)', fontsize=12, fontweight='bold')
        ax_right.set_xlabel('', fontsize=12, fontweight='bold')
        
        # Title
        title_text = f'Statistical Distribution'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.set_ylim(-50, 50)  # Set y-axis limits for center freq error
        ax_right.set_xlim(-0.5, 0.5)
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ]
        
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
    
    def _plot_channel_spacing_error(self, summary_df, spacing_df, output_path):
        """Create channel spacing error summary plot with statistical distribution by channel transition."""
        if summary_df.empty or spacing_df.empty:
            print("No data available for channel spacing error plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
        
        # Define colors and markers by bank
        bank_colors = {0: 'blue', 1: 'red'}
        bank_markers = {0: 'o', 1: '^'}
        
        # Left subplot - scatter plot by tile with individual channel transition data points
        ax_left = fig.add_subplot(gs[0, 0])
        
        # Get all unique tiles sorted (v1 first, then v2)
        v1_tiles = sorted(summary_df[summary_df['Version'] == 'v1']['Tile_SN'].unique())
        v2_tiles = sorted(summary_df[summary_df['Version'] == 'v2']['Tile_SN'].unique())
        all_tiles = v1_tiles + v2_tiles
        
        # Create x-axis positions
        tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}
        
        # Plot individual channel transition data points with jitter
        for bank in [0, 1]:
            df_bank = spacing_df[spacing_df['Bank'] == bank]
            
            for tile in all_tiles:
                df_tile = df_bank[df_bank['Tile_SN'] == tile]
                if not df_tile.empty:
                    x_pos = tile_to_pos[tile]
                    y_vals = df_tile['Spacing_Error_GHz'].values
                    
                    # Add scatter points with small jitter
                    x_scatter = np.random.normal(x_pos, 0.15, size=len(y_vals))
                    ax_left.scatter(x_scatter, y_vals, color=bank_colors[bank], 
                                  marker=bank_markers[bank], s=30, alpha=0.5, 
                                  edgecolors='black', linewidth=0.3)
        
        # Set x-axis
        ax_left.set_xticks(range(len(all_tiles)))
        ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=8)
        ax_left.set_xlabel('Tile_SN', fontsize=13, fontweight='bold')
        
        # Set y-axis
        ax_left.set_ylabel('Channel Spacing Error (GHz)', fontsize=13, fontweight='bold')
        ax_left.set_title('Channel Spacing Error by Tile', fontsize=14, fontweight='bold')
        ax_left.set_ylim(-50, 50)  # Set y-axis limits
        ax_left.grid(True, alpha=0.3)
        
        # Right subplot - statistical distribution by channel transition
        ax_right = fig.add_subplot(gs[0, 1])
        
        # Collect data for violin and box plots (by channel transition)
        box_data = []
        box_positions = []
        box_colors = []
        
        for bank in [0, 1]:
            df_bank = spacing_df[spacing_df['Bank'] == bank]
            
            for ch_from in range(7):  # ch0→ch1, ch1→ch2, ..., ch6→ch7
                df_transition = df_bank[(df_bank['Channel_From'] == ch_from) & 
                                       (df_bank['Channel_To'] == ch_from + 1)]
                
                if not df_transition.empty:
                    errors = df_transition['Spacing_Error_GHz'].values
                    y_pos = bank * 7 + ch_from
                    
                    box_data.append(errors)
                    box_positions.append(y_pos)
                    box_colors.append(bank_colors[bank])
        
        # Create horizontal violin plots
        parts = ax_right.violinplot(box_data, positions=box_positions, vert=False, widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plots by bank
        for pc, color in zip(parts['bodies'], box_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay horizontal boxplots on top of violin plots
        bp = ax_right.boxplot(box_data, positions=box_positions, vert=False, widths=0.3,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                             whiskerprops=dict(color='black', linewidth=2),
                             capprops=dict(color='black', linewidth=2),
                             medianprops=dict(color='red', linewidth=2.5))
        
        # Add annotations for mean and standard deviation
        annotation_x = 25  # Position at 25 GHz (within -40 to 40 range)
        
        for bank in [0, 1]:
            df_bank = spacing_df[spacing_df['Bank'] == bank]
            
            for ch_from in range(7):
                df_transition = df_bank[(df_bank['Channel_From'] == ch_from) & 
                                       (df_bank['Channel_To'] == ch_from + 1)]
                
                if not df_transition.empty:
                    errors = df_transition['Spacing_Error_GHz'].values
                    y_pos = bank * 7 + ch_from
                    
                    mean_val = np.mean(errors)
                    std_val = np.std(errors)
                    
                    # Annotate with mean and std
                    annotation_text = f'μ={mean_val:.2f}GHz\nσ={std_val:.2f}GHz'
                    ax_right.text(annotation_x, y_pos, annotation_text, 
                                fontsize=7, ha='left', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=bank_colors[bank], alpha=0.8, linewidth=1))
        
        # Calculate overall statistics
        all_errors = spacing_df['Spacing_Error_GHz'].values
        overall_mean = np.mean(all_errors)
        overall_std = np.std(all_errors)
        
        # Count total tiles
        total_tiles = len(all_tiles)
        
        # Y-axis labels showing channel transitions
        yticks = list(range(14))
        yticklabels = [f'B0: Ch{i}→Ch{i+1}' for i in range(7)] + [f'B1: Ch{i}→Ch{i+1}' for i in range(7)]
        ax_right.set_yticks(yticks)
        ax_right.set_yticklabels(yticklabels, fontsize=9)
        ax_right.set_xlabel('Channel Spacing Error (GHz)', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Channel Transition', fontsize=12, fontweight='bold')
        
        # Title with overall statistics
        title_text = f'Statistical Distribution\nμ={overall_mean:.2f}GHz, σ={overall_std:.2f}GHz'
        ax_right.set_title(title_text, fontsize=13, fontweight='bold')
        
        ax_right.grid(True, alpha=0.3, axis='x')
        ax_right.set_ylim(-0.5, 13.5)
        ax_right.set_xlim(-50, 50)  # Set x-axis limits for channel spacing error
        
        # Add horizontal line to separate banks
        ax_right.axhline(y=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Create legend for banks (inside the left subplot)
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 0'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', linewidth=1.5, label='Bank 1')
        ]
        
        ax_left.legend(handles=legend_elements, loc='upper right', ncol=2, 
                      fontsize=10, frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
    
    # TP3 Series Methods
    def tp3p1_analysis(self):
        """Analysis for TP3-1 test point data."""
        pass
    
    def tp3p2_analysis(self):
        """Analysis for TP3-2 test point data."""
        pass
    
    def tp3p3_analysis(self):
        """Analysis for TP3-3 test point data."""
        pass
    
    # TP4 Series Methods
    def tp4p1_analysis(self):
        """Analysis for TP4-1 test point data."""
        pass
    
    def tp4p2_analysis(self):
        """Analysis for TP4-2 test point data."""
        pass
    
    def tp4p3_analysis(self):
        """Analysis for TP4-3 test point data."""
        pass
    
    def _load_tp1p2_liv_data(self, tp_path, valid_tiles, version):
        """Load all TP1-2 LIV CSV files for valid tiles."""
        all_data = []
        
        if not tp_path.exists():
            print(f"  Warning: {tp_path} does not exist")
            return pd.DataFrame()
        
        for csv_file in tp_path.glob("*LIV.csv"):
            # Extract Tile_SN from filename (e.g., "2025-07-22T16_16_42-Y2529000038-TP1-2 LIV.csv")
            filename = csv_file.name
            parts = filename.split('-')
            
            # Find the part that starts with 'Y' followed by digits (Tile_SN pattern)
            tile_sn = None
            for part in parts:
                if part.startswith('Y') and len(part) > 1 and part[1:].isdigit():
                    tile_sn = part
                    break
            
            if tile_sn is None:
                continue
            
            # Only process valid tiles
            if tile_sn not in valid_tiles:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                df['Tile_SN'] = tile_sn
                df['Version'] = version
                
                # Convert columns to numeric, handling any non-numeric values
                df['Set Laser(mA)'] = pd.to_numeric(df['Set Laser(mA)'], errors='coerce')
                df['Power(mW)'] = pd.to_numeric(df['Power(mW)'], errors='coerce')
                df['Bank'] = pd.to_numeric(df['Bank'], errors='coerce')
                df['Channel'] = pd.to_numeric(df['Channel'], errors='coerce')
                
                # Remove rows with NaN values in critical columns
                df = df.dropna(subset=['Set Laser(mA)', 'Power(mW)', 'Bank', 'Channel'])
                
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file.name}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"  Loaded LIV data for {len(valid_tiles)} {version} tiles")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _plot_tp1p2_liv_overlay(self, df, output_path):
        """Plot the most representative (typical) LIV curve with filtered data."""
        if df.empty:
            print("No data available for LIV plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Calculate median LIV curve across all emitters (including zero-power points)
        # For each current value, find the median power
        median_curve = df.groupby('Set Laser(mA)', as_index=False)['Power(mW)'].median()
        
        # Find the emitter whose LIV curve is closest to the median
        best_emitter = None
        min_deviation = float('inf')
        
        for (tile_sn, bank, channel), group_df in df.groupby(['Tile_SN', 'Bank', 'Channel']):
            # Average power for each unique current (keep all points including zeros)
            group_df_unique = group_df.groupby('Set Laser(mA)', as_index=False)['Power(mW)'].mean()
            
            if len(group_df_unique) < 2:
                continue
            
            # Merge with median curve to compare at the same current points
            merged = pd.merge(group_df_unique, median_curve, on='Set Laser(mA)', 
                             suffixes=('_emitter', '_median'), how='inner')
            
            if len(merged) > 5:  # Need enough points for comparison
                # Calculate deviation from median (RMS error)
                deviation = np.sqrt(np.mean((merged['Power(mW)_emitter'] - merged['Power(mW)_median'])**2))
                
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_emitter = (tile_sn, bank, channel, group_df_unique)
        
        # Plot the most representative emitter
        representative_powers = {}
        if best_emitter is not None:
            tile_sn, bank, channel, emitter_data = best_emitter
            
            # Plot in black
            ax.plot(emitter_data['Set Laser(mA)'], emitter_data['Power(mW)'],
                   color='black', linewidth=2.5, linestyle='-', alpha=0.9, zorder=10)
            
            # Get power values at specific currents and mark them
            currents_of_interest = [120, 135, 150, 165]
            for current in currents_of_interest:
                power_at_current = emitter_data[emitter_data['Set Laser(mA)'] == float(current)]
                if not power_at_current.empty:
                    power_val = power_at_current.iloc[0]['Power(mW)']
                    representative_powers[current] = power_val
                    # Mark the point on the main curve
                    ax.plot(current, power_val, 'o', color='red', 
                           markersize=8, zorder=15, markeredgecolor='black', markeredgewidth=1.5)
            
            print(f"Most representative emitter: Tile {tile_sn}, Bank {bank}, Channel {channel}")
            print(f"  RMS deviation from median: {min_deviation:.3f} mW")
            for current, power in representative_powers.items():
                print(f"  Power at {current}mA: {power:.2f} mW")
        
        # Keep linear scale for y-axis
        ax.set_yscale('linear')
        
        # Labels (no title)
        ax.set_xlabel('Current (mA)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Optical Power (mW)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set axis limits - dynamically based on data or fixed to 170 mA
        max_current = df['Set Laser(mA)'].max() if not df.empty else 170
        max_power = df['Power(mW)'].max() if not df.empty else 40
        ax.set_xlim(0, max(170, max_current))
        ax.set_ylim(0, max(40, max_power * 1.05))
        
        # Add inset distribution plots for power at multiple currents
        if len(representative_powers) > 0:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from matplotlib.patches import ConnectionPatch
            
            # Define currents and their positions (x, y) in axes coordinates
            current_positions = {
                120: (0.10, 0.25),
                135: (0.30, 0.35),
                150: (0.50, 0.45),
                165: (0.70, 0.55)
            }
            
            for current, (pos_x, pos_y) in current_positions.items():
                if current not in representative_powers:
                    continue
                    
                representative_power = representative_powers[current]
                
                # Get all power values at this current
                df_current = df[df['Set Laser(mA)'] == float(current)].copy()
                
                if not df_current.empty and len(df_current) > 5:
                    # Position inset at specified coordinates
                    ax_inset = inset_axes(ax, width="30%", height="60%",
                                         bbox_to_anchor=(pos_x, pos_y, 0.30, 0.60), 
                                         bbox_transform=ax.transAxes, loc='lower left', borderpad=0)
                    
                    # Plot violin + box plot for distribution
                    parts = ax_inset.violinplot([df_current['Power(mW)'].values], positions=[0], 
                                               widths=0.5, showmeans=False, showmedians=False, showextrema=False)
                    
                    # Style violin plot
                    for pc in parts['bodies']:
                        pc.set_facecolor('#8B0000')
                        pc.set_alpha(0.6)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(1.5)
                    
                    # Overlay box plot
                    bp = ax_inset.boxplot([df_current['Power(mW)'].values], positions=[0], widths=0.15,
                                         patch_artist=True, showfliers=False,
                                         boxprops=dict(facecolor='white', color='black', linewidth=1.5),
                                         whiskerprops=dict(color='black', linewidth=1.5),
                                         capprops=dict(color='black', linewidth=1.5),
                                         medianprops=dict(color='red', linewidth=2))
                    
                    # Mark the representative emitter's value
                    ax_inset.plot(0, representative_power, 'o', color='red', 
                                 markersize=6, zorder=10, markeredgecolor='black', markeredgewidth=1.5)
                    
                    # Style inset
                    ax_inset.set_xlim(-0.6, 0.6)
                    ax_inset.set_ylim(0, 40)  # Fixed y-axis range
                    ax_inset.set_xticks([])
                    ax_inset.set_ylabel(f'{current}mA', fontsize=9, fontweight='bold')
                    ax_inset.tick_params(labelsize=8)
                    ax_inset.grid(True, alpha=0.3, axis='y')
                    
                    # Add annotation showing statistics
                    median_val = df_current['Power(mW)'].median()
                    std_val = df_current['Power(mW)'].std()
                    ax_inset.text(0.98, 0.05, f'μ={median_val:.1f}mW\nσ={std_val:.1f}mW',
                                transform=ax_inset.transAxes, fontsize=7,
                                verticalalignment='bottom', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                    
                    # Add arrow pointing from inset to current point on main curve
                    con = ConnectionPatch((0, representative_power), (current, representative_power),
                                         coordsA=ax_inset.transData, coordsB=ax.transData,
                                         arrowstyle='->', shrinkA=5, shrinkB=5, linewidth=1.5, color='red', alpha=0.7)
                    fig.add_artist(con)
                    
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _plot_tp1p2_liv_simple(self, df, output_path):
        """Plot a simple LIV curve without insets - just the representative emitter."""
        if df.empty:
            print("No data available for LIV plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Calculate median LIV curve across all emitters (including zero-power points)
        # For each current value, find the median power
        median_curve = df.groupby('Set Laser(mA)', as_index=False)['Power(mW)'].median()
        
        # Find the emitter whose LIV curve is closest to the median
        best_emitter = None
        min_deviation = float('inf')
        
        for (tile_sn, bank, channel), group_df in df.groupby(['Tile_SN', 'Bank', 'Channel']):
            # Average power for each unique current (keep all points including zeros)
            group_df_unique = group_df.groupby('Set Laser(mA)', as_index=False)['Power(mW)'].mean()
            
            if len(group_df_unique) < 2:
                continue
            
            # Merge with median curve to compare at the same current points
            merged = pd.merge(group_df_unique, median_curve, on='Set Laser(mA)', 
                             suffixes=('_emitter', '_median'), how='inner')
            
            if len(merged) > 5:  # Need enough points for comparison
                # Calculate deviation from median (RMS error)
                deviation = np.sqrt(np.mean((merged['Power(mW)_emitter'] - merged['Power(mW)_median'])**2))
                
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_emitter = (tile_sn, bank, channel, group_df_unique)
        
        # Plot the most representative emitter
        if best_emitter is not None:
            tile_sn, bank, channel, emitter_data = best_emitter
            
            # Plot in black
            ax.plot(emitter_data['Set Laser(mA)'], emitter_data['Power(mW)'],
                   color='black', linewidth=2.5, linestyle='-', alpha=0.9)
            
            print(f"Most representative emitter: Tile {tile_sn}, Bank {bank}, Channel {channel}")
            print(f"  RMS deviation from median: {min_deviation:.3f} mW")
        
        # Keep linear scale for y-axis
        ax.set_yscale('linear')
        
        # Labels (no title)
        ax.set_xlabel('Current (mA)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Optical Power (mW)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set axis limits - dynamically based on data or fixed to 170 mA
        max_current = df['Set Laser(mA)'].max() if not df.empty else 170
        max_power = df['Power(mW)'].max() if not df.empty else 40
        ax.set_xlim(0, max(170, max_current))
        ax.set_ylim(0, max(40, max_power * 1.05))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def ofc_plotter(self):
        """
        OFC plotter for frequency error and power distribution.
        Creates simplified violin plots for OFC presentation.
        Uses OFC-specific filters from filter.yaml.
        """
        print("Starting OFC Plotter...")
        print("Applying OFC-specific filters from filter.yaml...")
        
        # Create ofc folder
        ofc_path = self.results_path / "ofc"
        ofc_path.mkdir(exist_ok=True)
        
        # Load OFC-specific filters
        import yaml
        filter_file = self.base_path / "analysis_src" / "filter.yaml"
        with open(filter_file, 'r') as f:
            filter_config = yaml.safe_load(f)
            ofc_filters = filter_config['ofc_filters']
        
        print(f"OFC Filters: Power {ofc_filters['optical_power']['min_mw']}-{ofc_filters['optical_power']['max_mw']}mW, "
              f"Freq error: {ofc_filters['frequency_error']['min_ghz']} to {ofc_filters['frequency_error']['max_ghz']} GHz")
        
        # Get valid tiles that pass all filters
        valid_tiles = self._get_valid_tiles()
        
        # Load wavelength grid for frequency error
        grid_file = self.base_path / "analysis_src" / "wavelength_grid.yaml"
        with open(grid_file, 'r') as f:
            wl_grid = yaml.safe_load(f)
        
        # Load TP2-5 data for frequency error
        print("\nLoading TP2-5 data for frequency error...")
        df_freq_v1 = self._load_tp2p5_data(self.v1_path / "TP2-5", wl_grid, valid_tiles['v1'])
        df_freq_v2 = self._load_tp2p5_data(self.v2_path / "TP2-5", wl_grid, valid_tiles['v2'])
        df_freq_v1['Version'] = 'v1'
        df_freq_v2['Version'] = 'v2'
        df_freq = pd.concat([df_freq_v1, df_freq_v2], ignore_index=True)
        print(f"Loaded {len(df_freq)} frequency error records (before OFC filtering)")
        
        # Apply OFC frequency error filter
        freq_min = ofc_filters['frequency_error']['min_ghz']
        freq_max = ofc_filters['frequency_error']['max_ghz']
        df_freq = df_freq[(df_freq['Frequency_Error_GHz'] >= freq_min) & 
                          (df_freq['Frequency_Error_GHz'] <= freq_max)]
        print(f"After OFC freq filter: {len(df_freq)} records (within {freq_min} to {freq_max} GHz)")
        
        # Load TP2-6 data for power
        print("\nLoading TP2-6 data for power...")
        df_power_v1 = self._load_tp2p6_data(self.v1_path / "TP2-6")
        df_power_v2 = self._load_tp2p6_data(self.v2_path / "TP2-6")
        df_power_v1 = df_power_v1[df_power_v1['Tile_SN'].isin(valid_tiles['v1'])].copy()
        df_power_v2 = df_power_v2[df_power_v2['Tile_SN'].isin(valid_tiles['v2'])].copy()
        df_power_v1['Version'] = 'v1'
        df_power_v2['Version'] = 'v2'
        df_power = pd.concat([df_power_v1, df_power_v2], ignore_index=True)
        print(f"Loaded {len(df_power)} power records (before OFC filtering)")
        
        # Apply OFC power filter
        power_min = ofc_filters['optical_power']['min_mw']
        power_max = ofc_filters['optical_power']['max_mw']
        df_power = df_power[(df_power['Power(mW)'] >= power_min) & 
                            (df_power['Power(mW)'] <= power_max)]
        print(f"After OFC power filter: {len(df_power)} records (within {power_min} to {power_max} mW)")
        
        # Plot frequency error
        self._plot_ofc_freq_error(df_freq, ofc_path / "ofc_freq_error.png")
        
        # Plot power
        self._plot_ofc_power(df_power, ofc_path / "ofc_power.png")
        
        print("\nOFC Plotter completed!")
        print(f"Plots saved to: {ofc_path}")
    
    def _plot_ofc_freq_error(self, df, output_path):
        """Create OFC frequency error scatter plot with all data points."""
        if df.empty:
            print("No data available for frequency error plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Define colors by bank (Red for Set A/Bank 1, Blue for Set B/Bank 0)
        bank_colors = {1: 'red', 0: 'blue'}  # Bank 1 = Set A (Red), Bank 0 = Set B (Blue)
        bank_markers = {1: 'o', 0: '^'}  # Circle for Set A, Triangle for Set B
        
        x_labels = []
        
        for channel in range(8):
            for bank in [1, 0]:  # Set A (Bank 1) first, then Set B (Bank 0)
                df_channel = df[(df['Bank'] == bank) & (df['Channel'] == channel)]
                if not df_channel.empty:
                    errors = df_channel['Frequency_Error_GHz'].values
                    pos = channel * 2 + (0 if bank == 1 else 1)  # Set A on left, Set B on right
                    
                    # Add scatter plot with all data points
                    x_scatter = np.random.normal(pos, 0.1, size=len(errors))
                    ax.scatter(x_scatter, errors, color=bank_colors[bank], 
                              alpha=0.7, s=30, marker=bank_markers[bank],
                              edgecolors='black', linewidth=0.5)
            
            x_labels.append(f'Ch{channel+1}')
        
        # Set x-axis labels to channel numbers
        channel_positions = [i * 2 + 0.5 for i in range(8)]
        ax.set_xticks(channel_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        
        # Labels (no title)
        ax.set_xlabel('Channel', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency Error (GHz)', fontsize=11, fontweight='bold')
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, markeredgecolor='black', label='Set A', alpha=0.7),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                   markersize=8, markeredgecolor='black', label='Set B', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _plot_ofc_power(self, df, output_path):
        """Create OFC power scatter plot with all data points."""
        if df.empty:
            print("No data available for power plot")
            return
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Define colors by bank (Red for Set A/Bank 1, Blue for Set B/Bank 0)
        bank_colors = {1: 'red', 0: 'blue'}  # Bank 1 = Set A (Red), Bank 0 = Set B (Blue)
        bank_markers = {1: 'o', 0: '^'}  # Circle for Set A, Triangle for Set B
        
        x_labels = []
        
        for channel in range(8):
            for bank in [1, 0]:  # Set A (Bank 1) first, then Set B (Bank 0)
                df_channel = df[(df['Bank'] == bank) & (df['Channel'] == channel)]
                if not df_channel.empty:
                    powers = df_channel['Power(mW)'].values
                    pos = channel * 2 + (0 if bank == 1 else 1)  # Set A on left, Set B on right
                    
                    # Add scatter plot with all data points
                    x_scatter = np.random.normal(pos, 0.1, size=len(powers))
                    ax.scatter(x_scatter, powers, color=bank_colors[bank], 
                              alpha=0.7, s=30, marker=bank_markers[bank],
                              edgecolors='black', linewidth=0.5)
            
            x_labels.append(f'Ch{channel+1}')
        
        # Set x-axis labels to channel numbers
        channel_positions = [i * 2 + 0.5 for i in range(8)]
        ax.set_xticks(channel_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        
        # Labels (no title)
        ax.set_xlabel('Channel', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power (mW)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, markeredgecolor='black', label='Set A', alpha=0.7),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                   markersize=8, markeredgecolor='black', label='Set B', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    analyzer = tpanalysis()
    print("=" * 60)
    print("CLM Manufacturing Data Test Point Analysis")
    print("Available test points: TP1-1 through TP4-3")
    print("=" * 60)
    
    # Run TP1-2 LIV analysis
    analyzer.tp1p2_analysis()
    
    # Run TP2-4 analysis
    analyzer.tp2p4_analysis()
    
    # Run TP2-5 analysis
    analyzer.tp2p5_analysis()
    
    # Run TP2-5 Total Power analysis
    analyzer.tp2p5_totalpower_analysis()
    
    print("\n" + "=" * 60 + "\n")
    
    # Run TP2-6 analysis
    analyzer.tp2p6_analysis()
    
    # Run center frequency and channel spacing analysis
    analyzer.center_freq_spacing_analysis()
