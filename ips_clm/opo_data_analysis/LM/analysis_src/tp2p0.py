"""
TP2P0 - Lensing Station Data Extractor
=====================================

This script searches for the latest xlsx file in the lensing station folder,
identifies all sheets in the Excel file (which are TileSN), and stores them.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob


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


def main():
    """
    Main function to run the lensing station data extractor
    """
    # You can specify a custom path here if needed
    # extractor = LensingStationDataExtractor("/path/to/lensing/station")
    extractor = LensingStationDataExtractor()
    
    # Run the extraction
    tile_sns = extractor.run()
    
    # Final summary
    if tile_sns:
        print(f"\n🔍 FINAL RESULT:")
        print(f"   • Successfully extracted {len(tile_sns)} TileSN from the latest lensing station file")
        print(f"   • TileSN are stored in the extractor and ready for use")
        print(f"   • Access via: extractor.get_tile_sns()")
        
        # Show first few and last few TileSN as a preview
        if len(tile_sns) > 10:
            print(f"\n📋 TILESN PREVIEW:")
            print(f"   First 5: {tile_sns[:5]}")
            print(f"   Last 5:  {tile_sns[-5:]}")
        else:
            print(f"\n📋 ALL TILESN: {tile_sns}")
    else:
        print(f"\n❌ No TileSN were extracted. Please check the lensing station folder and xlsx file.")


if __name__ == "__main__":
    main() 