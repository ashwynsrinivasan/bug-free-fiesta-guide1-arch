#!/usr/bin/env python3
"""
Tile Analysis Script - Combined TP1 to TP3 Data Analysis
=======================================================

This script combines results from tp1p1.py to tp3p3.py and creates comprehensive
tile-specific analysis plots organized by tile serial number.

Data is stored in plots/Tiles/TileSN/ where TileSN is the tile serial number.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path
import json
import glob
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import the individual analysis modules
from tp1p1 import TP1P1CombinedAnalyzer
from tp1p2 import TP1P2CombinedAnalyzer  
from tp1p3 import TP1P3CombinedAnalyzer
from tp1p4 import TP1P4CombinedAnalyzer
from tp2p0 import LensingStationDataExtractor
from tp2p1 import TP2P1CombinerAnalyzers
from tp2p2 import TP2P2CombinerAnalyzers
from tp2p4 import TP2p4CombinedAnalyzers
from tp3p1 import TP3P1CombinedAnalyzer
from tp3p3 import TP3p3CombinedAnalyzers

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


class TileAnalyzer:
    """
    Comprehensive Tile Analysis Class
    
    This class combines data from all test points (TP1-1 to TP3-3) and creates
    tile-specific analysis plots organized by tile serial number.
    """
    
    def __init__(self):
        """Initialize the Tile Analyzer"""
        script_dir = Path(__file__).parent
        self.script_dir = script_dir
        self.data_dir = script_dir / "data"
        self.plots_dir = script_dir / "plots"
        self.tile_plots_dir = self.plots_dir / "Tiles"
        self.tile_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.analyzers = {}
        self.tile_data = {}  # Store combined data for each tile
        self.available_tiles = set()
        
        print(f"🔧 Tile Analyzer initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Tile plots directory: {self.tile_plots_dir}")
    
    def initialize_analyzers(self):
        """Initialize all TP analyzers"""
        print("\n🚀 Initializing TP analyzers...")
        
        try:
            self.analyzers['tp1p1'] = TP1P1CombinedAnalyzer()
            print("   ✓ TP1-1 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP1-1 analyzer failed: {e}")
            
        try:
            self.analyzers['tp1p2'] = TP1P2CombinedAnalyzer()
            print("   ✓ TP1-2 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP1-2 analyzer failed: {e}")
            
        try:
            self.analyzers['tp1p3'] = TP1P3CombinedAnalyzer()
            print("   ✓ TP1-3 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP1-3 analyzer failed: {e}")
            
        try:
            self.analyzers['tp1p4'] = TP1P4CombinedAnalyzer()
            print("   ✓ TP1-4 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP1-4 analyzer failed: {e}")
            
        try:
            self.analyzers['tp2p0'] = LensingStationDataExtractor()
            print("   ✓ TP2-0 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP2-0 analyzer failed: {e}")
            
        try:
            self.analyzers['tp2p1'] = TP2P1CombinerAnalyzers()
            print("   ✓ TP2-1 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP2-1 analyzer failed: {e}")
            
        try:
            self.analyzers['tp2p2'] = TP2P2CombinerAnalyzers()
            print("   ✓ TP2-2 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP2-2 analyzer failed: {e}")
            
        try:
            self.analyzers['tp2p4'] = TP2p4CombinedAnalyzers()
            print("   ✓ TP2-4 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP2-4 analyzer failed: {e}")
            
        try:
            self.analyzers['tp3p1'] = TP3P1CombinedAnalyzer()
            print("   ✓ TP3-1 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP3-1 analyzer failed: {e}")
            
        try:
            self.analyzers['tp3p3'] = TP3p3CombinedAnalyzers()
            print("   ✓ TP3-3 analyzer initialized")
        except Exception as e:
            print(f"   ⚠️  TP3-3 analyzer failed: {e}")
    
    def discover_tiles_from_plots(self):
        """Discover additional tiles from existing plot files"""
        print("\n🔍 Discovering tiles from existing plot files...")
        
        plot_folders = [
            self.plots_dir / "TP1-2",
            self.plots_dir / "TP1-4", 
            self.plots_dir / "TP2-0",
            self.plots_dir / "TP2-1",
            self.plots_dir / "TP2-2",
            self.plots_dir / "TP2-4",
            self.plots_dir / "TP3-1",
            self.plots_dir / "TP3-3"
        ]
        
        plot_patterns = [
            # TP1-2 patterns
            "MPD_Responsivity_*.png",
            "MPD_*.png",
            # TP1-4 patterns  
            "LIV_*.png",
            "VOA_*.png", 
            "Wavelength_vs_VOA_*.png",
            # TP2-1 patterns
            "VOA_MPD_MUX_*.png",
            "VOA_MPD_PIC_*.png",
            "VOA_DeltaWave_*.png",
            # TP2-2 patterns
            "TempScan_*.png",
            # TP2-4 patterns
            "AnnotatePower_*.png",
            "Wavelength_Spectrum_*.png",
            "Wavelength_Setpoint_*.png",
            # TP3-1 patterns
            "LIV_*.png",
            "MPD_*.png",
            "VOA_*.png",
            "Wavelength_vs_VOA_*.png",
            # TP3-3 patterns
            "Wavelength_Setpoint_*.png"
        ]
        
        tiles_from_plots = set()
        
        for folder in plot_folders:
            if folder.exists():
                for pattern in plot_patterns:
                    for plot_file in folder.glob(pattern):
                        # Extract tile serial number from filename
                        filename = plot_file.name
                        # Look for pattern like "PlotType_TileSN.png"
                        match = re.search(r'_(Y\d+)\.png$', filename)
                        if match:
                            tile_sn = match.group(1)
                            tiles_from_plots.add(tile_sn)
                            self.available_tiles.add(tile_sn)
        
        print(f"   ✓ Found {len(tiles_from_plots)} additional tiles from plot files")
        if tiles_from_plots:
            print(f"   Plot tiles: {sorted(list(tiles_from_plots))[:10]}{'...' if len(tiles_from_plots) > 10 else ''}")
    
    def extract_tile_data(self):
        """Extract data from all TP analysis results"""
        print("\n📊 Extracting tile data from all test points...")
        
        # Load processed data files
        data_files = {
            'tp1p1': self.data_dir / "tp1p1_combined_data.nc",
            'tp1p2': self.data_dir / "tp1p2_combined_data.nc", 
            'tp1p3': self.data_dir / "tp1p3_combined_data.nc",
            'tp1p4': self.data_dir / "tp1p4_combined_data.nc",
            'tp2p0_single': self.data_dir / "single_ch_data.csv",
            'tp2p0_all': self.data_dir / "all_ch_data.csv",
            'tp2p2': self.data_dir / "tp2p2_processed_data.csv"
        }
        
        # Load CSV data files
        csv_files = [
            'tp1p4_tuning_efficiency.csv',
            'tp1p2_tuning_efficiency.csv',
            'tp1p4_voa_loss_20mA_data.csv',
            'tp1p4_voa_loss_10mA_data.csv',
            'tp1p4_wavelength_voa_slope_data.csv',
            'tp2p2_wavelength_tuning.csv'
        ]
        
        for csv_file in csv_files:
            file_path = self.data_dir / csv_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Look for various tile identifier column names
                    tile_col = None
                    if 'Tile_SN' in df.columns:
                        tile_col = 'Tile_SN'
                    elif 'Serial Number' in df.columns:
                        tile_col = 'Serial Number'
                    elif 'Tile' in df.columns:
                        tile_col = 'Tile'
                    elif 'tile_sn' in df.columns:
                        tile_col = 'tile_sn'
                    
                    if tile_col:
                        # Extract tile serial numbers
                        tiles = df[tile_col].unique()
                        
                        for tile in tiles:
                            if tile not in self.tile_data:
                                self.tile_data[tile] = {}
                            self.tile_data[tile][csv_file.replace('.csv', '')] = df[
                                df[tile_col] == tile
                            ]
                            self.available_tiles.add(tile)
                    
                    print(f"   ✓ Loaded {csv_file}: {len(df)} records, tile column: {tile_col}")
                except Exception as e:
                    print(f"   ⚠️  Failed to load {csv_file}: {e}")
        
        # Also discover tiles from existing plot files
        self.discover_tiles_from_plots()
        
        print(f"\n📈 Found data for {len(self.available_tiles)} tiles total")
        print(f"   Tiles: {sorted(list(self.available_tiles))[:10]}{'...' if len(self.available_tiles) > 10 else ''}")
    
    def create_tile_summary_plot(self, tile_sn: str) -> None:
        """Create comprehensive summary plot combining existing detailed plots for a specific tile"""
        tile_dir = self.tile_plots_dir / tile_sn
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the existing plot types and their locations
        plot_sources = [
            # TP1-2 plots
            (self.plots_dir / "TP1-2" / f"MPD_Responsivity_{tile_sn}.png", "TP1-2: MPD Responsivity"),
            (self.plots_dir / "TP1-2" / f"MPD_{tile_sn}.png", "TP1-2: MPD Analysis"),
            
            # TP1-4 plots  
            (self.plots_dir / "TP1-4" / f"LIV_{tile_sn}.png", "TP1-4: LIV Analysis"),
            (self.plots_dir / "TP1-4" / f"VOA_{tile_sn}.png", "TP1-4: VOA Analysis"),
            (self.plots_dir / "TP1-4" / f"Wavelength_vs_VOA_{tile_sn}.png", "TP1-4: Wavelength vs VOA"),
            
            # TP2-1 plots
            (self.plots_dir / "TP2-1" / f"VOA_MPD_MUX_{tile_sn}.png", "TP2-1: VOA MPD MUX"),
            (self.plots_dir / "TP2-1" / f"VOA_MPD_PIC_{tile_sn}.png", "TP2-1: VOA MPD PIC"),
            (self.plots_dir / "TP2-1" / f"VOA_DeltaWave_{tile_sn}.png", "TP2-1: VOA Delta Wave"),
            
            # TP2-2 plots
            (self.plots_dir / "TP2-2" / f"TempScan_{tile_sn}.png", "TP2-2: Temperature Scan"),
            
            # TP2-4 plots
            (self.plots_dir / "TP2-4" / f"AnnotatePower_{tile_sn}.png", "TP2-4: Annotated Power"),
            (self.plots_dir / "TP2-4" / f"Wavelength_Spectrum_{tile_sn}.png", "TP2-4: Wavelength Spectrum"),
            (self.plots_dir / "TP2-4" / f"Wavelength_Setpoint_{tile_sn}.png", "TP2-4: Wavelength Setpoint"),
            
            # TP3-1 plots
            (self.plots_dir / "TP3-1" / f"LIV_{tile_sn}.png", "TP3-1: LIV Analysis"),
            (self.plots_dir / "TP3-1" / f"MPD_{tile_sn}.png", "TP3-1: MPD Analysis"),
            (self.plots_dir / "TP3-1" / f"VOA_{tile_sn}.png", "TP3-1: VOA Analysis"),
            (self.plots_dir / "TP3-1" / f"Wavelength_vs_VOA_{tile_sn}.png", "TP3-1: Wavelength vs VOA"),
            
            # TP3-3 plots
            (self.plots_dir / "TP3-3" / f"Wavelength_Setpoint_{tile_sn}.png", "TP3-3: Wavelength Setpoint"),
        ]
        
        # Check which plots exist for this tile
        available_plots = []
        for plot_path, title in plot_sources:
            if plot_path.exists():
                available_plots.append((plot_path, title))
        
        if not available_plots:
            print(f"   ⚠️  No detailed plots found for tile {tile_sn}")
            return
        
        # Create figure with subplots for available plots
        n_plots = len(available_plots)
        
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots <= 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        elif n_plots <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        fig.suptitle(f'Detailed Analysis Summary - Tile {tile_sn}', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # Load and display each available plot
        for i, (plot_path, title) in enumerate(available_plots):
            try:
                img = mpimg.imread(plot_path)
                axes[i].imshow(img)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{title}\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, color='red')
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the combined plot
        output_file = tile_dir / f"{tile_sn}_detailed_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Created detailed analysis plot for {tile_sn}: {output_file} ({n_plots} plots combined)")
    
    def create_all_tile_plots(self):
        """Create detailed analysis plots for all available tiles"""
        print(f"\n🎨 Creating detailed analysis plots for {len(self.available_tiles)} tiles...")
        
        for i, tile_sn in enumerate(sorted(self.available_tiles)):
            print(f"   📊 Processing tile {tile_sn} ({i+1}/{len(self.available_tiles)})")
            
            try:
                self.create_tile_summary_plot(tile_sn)
            except Exception as e:
                print(f"   ⚠️  Failed to create plots for {tile_sn}: {e}")
    
    def create_overview_summary(self):
        """Create an overview summary of all tiles"""
        print("\n📋 Creating overview summary...")
        
        # Create summary statistics based on available plots
        summary_data = {
            'tile_sn': [],
            'available_plots': [],
            'plot_coverage_score': []
        }
        
        for tile_sn in sorted(self.available_tiles):
            # Count available plots for this tile
            plot_sources = [
                self.plots_dir / "TP1-2" / f"MPD_Responsivity_{tile_sn}.png",
                self.plots_dir / "TP1-2" / f"MPD_{tile_sn}.png",
                self.plots_dir / "TP1-4" / f"LIV_{tile_sn}.png",
                self.plots_dir / "TP1-4" / f"VOA_{tile_sn}.png",
                self.plots_dir / "TP1-4" / f"Wavelength_vs_VOA_{tile_sn}.png",
                self.plots_dir / "TP2-1" / f"VOA_MPD_MUX_{tile_sn}.png",
                self.plots_dir / "TP2-1" / f"VOA_MPD_PIC_{tile_sn}.png",
                self.plots_dir / "TP2-1" / f"VOA_DeltaWave_{tile_sn}.png",
                self.plots_dir / "TP2-2" / f"TempScan_{tile_sn}.png",
                self.plots_dir / "TP2-4" / f"AnnotatePower_{tile_sn}.png",
                self.plots_dir / "TP2-4" / f"Wavelength_Spectrum_{tile_sn}.png",
                self.plots_dir / "TP2-4" / f"Wavelength_Setpoint_{tile_sn}.png",
                self.plots_dir / "TP3-1" / f"LIV_{tile_sn}.png",
                self.plots_dir / "TP3-1" / f"MPD_{tile_sn}.png",
                self.plots_dir / "TP3-1" / f"VOA_{tile_sn}.png",
                self.plots_dir / "TP3-1" / f"Wavelength_vs_VOA_{tile_sn}.png",
                self.plots_dir / "TP3-3" / f"Wavelength_Setpoint_{tile_sn}.png",
            ]
            
            available_count = sum(1 for plot_path in plot_sources if plot_path.exists())
            
            summary_data['tile_sn'].append(tile_sn)
            summary_data['available_plots'].append(available_count)
            
            # Calculate plot coverage score (percentage of available plots)
            coverage_score = (available_count / len(plot_sources)) * 100
            summary_data['plot_coverage_score'].append(coverage_score)
        
        # Create overview plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Available Plots per Tile
        ax1.bar(range(len(summary_data['tile_sn'])), summary_data['available_plots'])
        ax1.set_title('Available Detailed Plots per Tile')
        ax1.set_ylabel('Number of Plots')
        ax1.set_xlabel('Tile Index')
        ax1.set_ylim(0, 17)  # Max 17 plot types now (12 original + 4 TP3-1 + 1 TP3-3)
        
        # Plot 2: Plot Coverage Score Distribution
        ax2.hist(summary_data['plot_coverage_score'], bins=20, alpha=0.7)
        ax2.set_title('Plot Coverage Distribution')
        ax2.set_xlabel('Coverage Score (%)')
        ax2.set_ylabel('Number of Tiles')
        
        plt.tight_layout()
        
        # Save overview plot
        overview_file = self.tile_plots_dir / "tiles_overview_summary.png"
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.tile_plots_dir / "tiles_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"   ✓ Overview summary saved: {overview_file}")
        print(f"   ✓ Summary CSV saved: {summary_csv}")
        
        # Print some statistics
        avg_plots = np.mean(summary_data['available_plots'])
        avg_coverage = np.mean(summary_data['plot_coverage_score'])
        print(f"   📊 Average plots per tile: {avg_plots:.1f}")
        print(f"   📊 Average coverage: {avg_coverage:.1f}%")
    
    def run_analysis(self):
        """Run the complete tile analysis"""
        print("=" * 80)
        print("🚀 COMPREHENSIVE TILE ANALYSIS - TP1 TO TP3 DATA")
        print("=" * 80)
        print("Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)
        
        # Initialize analyzers
        self.initialize_analyzers()
        
        # Extract tile data
        self.extract_tile_data()
        
        if not self.available_tiles:
            print("❌ No tile data found. Please ensure TP analysis has been run first.")
            return
        
        # Create tile-specific plots
        self.create_all_tile_plots()
        
        # Create overview summary
        self.create_overview_summary()
        
        print("\n" + "=" * 80)
        print("✅ TILE ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📊 Processed {len(self.available_tiles)} tiles")
        print(f"📁 Output directory: {self.tile_plots_dir}")
        print(f"🔍 Individual detailed analysis plots organized by serial number")
        print(f"🎨 Each tile directory contains combined plots from TP1 to TP3 analysis")
        print("=" * 80)


def main():
    """Main function to run tile analysis"""
    analyzer = TileAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 