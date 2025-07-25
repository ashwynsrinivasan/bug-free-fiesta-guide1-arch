"""
Wavelength Grid Utilities
========================

Utility functions for loading and working with the wavelength grid configuration.
This module provides easy access to the wavelength grid data for all analysis scripts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


def load_wavelength_grid(grid_file: Optional[str] = None) -> Dict:
    """
    Load the wavelength grid configuration from JSON file.
    
    Parameters
    ----------
    grid_file : str, optional
        Path to the wavelength grid JSON file. If None, uses default location.
        
    Returns
    -------
    dict
        Wavelength grid configuration dictionary
    """
    if grid_file is None:
        # Default location relative to this script
        script_dir = Path(__file__).parent
        grid_file = str(script_dir / "wavelength_grid.json")
    
    if not os.path.exists(grid_file):
        raise FileNotFoundError(f"Wavelength grid file not found: {grid_file}")
    
    with open(grid_file, 'r') as f:
        return json.load(f)


def get_channel_wavelengths(bank: Union[int, str], 
                           grid_data: Optional[Dict] = None) -> List[float]:
    """
    Get wavelength values for all channels in a specific bank.
    
    Parameters
    ----------
    bank : int or str
        Bank number (0 or 1) or bank name ('Bank 0' or 'Bank 1')
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
        
    Returns
    -------
    list
        List of wavelength values in nm for the specified bank
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    # Convert bank input to Set designation
    if bank == 0 or bank == "Bank 0":
        set_name = "Set_A"
    elif bank == 1 or bank == "Bank 1":
        set_name = "Set_B"
    else:
        raise ValueError(f"Invalid bank: {bank}. Must be 0, 1, 'Bank 0', or 'Bank 1'")
    
    wavelengths = []
    for channel_data in grid_data['wavelength_grid']['channels']:
        wavelengths.append(channel_data[set_name]['wavelength_nm'])
    
    return wavelengths


def get_channel_frequencies(bank: Union[int, str], 
                           grid_data: Optional[Dict] = None) -> List[float]:
    """
    Get frequency values for all channels in a specific bank.
    
    Parameters
    ----------
    bank : int or str
        Bank number (0 or 1) or bank name ('Bank 0' or 'Bank 1')
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
        
    Returns
    -------
    list
        List of frequency values in THz for the specified bank
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    # Convert bank input to Set designation
    if bank == 0 or bank == "Bank 0":
        set_name = "Set_A"
    elif bank == 1 or bank == "Bank 1":
        set_name = "Set_B"
    else:
        raise ValueError(f"Invalid bank: {bank}. Must be 0, 1, 'Bank 0', or 'Bank 1'")
    
    frequencies = []
    for channel_data in grid_data['wavelength_grid']['channels']:
        frequencies.append(channel_data[set_name]['frequency_THz'])
    
    return frequencies


def get_channel_value(bank: Union[int, str], 
                     channel: int, 
                     value_type: str = 'wavelength',
                     grid_data: Optional[Dict] = None) -> float:
    """
    Get a specific wavelength or frequency value for a channel.
    
    Parameters
    ----------
    bank : int or str
        Bank number (0 or 1) or bank name ('Bank 0' or 'Bank 1')
    channel : int
        Channel number (1-8)
    value_type : str
        'wavelength' or 'frequency'
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
        
    Returns
    -------
    float
        Wavelength in nm or frequency in THz
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    # Convert bank input to Set designation
    if bank == 0 or bank == "Bank 0":
        set_name = "Set_A"
    elif bank == 1 or bank == "Bank 1":
        set_name = "Set_B"
    else:
        raise ValueError(f"Invalid bank: {bank}. Must be 0, 1, 'Bank 0', or 'Bank 1'")
    
    # Validate channel number
    if not (1 <= channel <= 8):
        raise ValueError(f"Invalid channel: {channel}. Must be 1-8")
    
    # Get the channel data (channels are 1-indexed in the grid)
    channel_data = grid_data['wavelength_grid']['channels'][channel - 1]
    
    if value_type == 'wavelength':
        return channel_data[set_name]['wavelength_nm']
    elif value_type == 'frequency':
        return channel_data[set_name]['frequency_THz']
    else:
        raise ValueError(f"Invalid value_type: {value_type}. Must be 'wavelength' or 'frequency'")


def get_center_wavelength(bank: Union[int, str], 
                         grid_data: Optional[Dict] = None) -> float:
    """
    Get the center wavelength for a specific bank.
    
    Parameters
    ----------
    bank : int or str
        Bank number (0 or 1) or bank name ('Bank 0' or 'Bank 1')
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
        
    Returns
    -------
    float
        Center wavelength in nm
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    # Convert bank input to Set designation
    if bank == 0 or bank == "Bank 0":
        set_name = "Set_A"
    elif bank == 1 or bank == "Bank 1":
        set_name = "Set_B"
    else:
        raise ValueError(f"Invalid bank: {bank}. Must be 0, 1, 'Bank 0', or 'Bank 1'")
    
    return grid_data['center_wavelengths'][set_name]['wavelength_nm']


def get_channel_spacing(bank: Union[int, str], 
                       grid_data: Optional[Dict] = None) -> float:
    """
    Get the channel spacing for a specific bank.
    
    Parameters
    ----------
    bank : int or str
        Bank number (0 or 1) or bank name ('Bank 0' or 'Bank 1')
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
        
    Returns
    -------
    float
        Channel spacing in nm
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    # Convert bank input to Set designation
    if bank == 0 or bank == "Bank 0":
        set_name = "Set_A"
    elif bank == 1 or bank == "Bank 1":
        set_name = "Set_B"
    else:
        raise ValueError(f"Invalid bank: {bank}. Must be 0, 1, 'Bank 0', or 'Bank 1'")
    
    return grid_data['channel_spacing'][set_name]['wavelength_nm']


def print_wavelength_grid_summary(grid_data: Optional[Dict] = None):
    """
    Print a summary of the wavelength grid configuration.
    
    Parameters
    ----------
    grid_data : dict, optional
        Wavelength grid data. If None, loads from default file.
    """
    if grid_data is None:
        grid_data = load_wavelength_grid()
    
    print("=" * 80)
    print("WAVELENGTH GRID CONFIGURATION")
    print("=" * 80)
    print(f"Description: {grid_data['metadata']['description']}")
    print(f"Bank Mapping: Set A → Bank 0, Set B → Bank 1")
    print(f"Units: {grid_data['metadata']['units']['wavelength']}, {grid_data['metadata']['units']['frequency']}")
    print()
    
    # Print channel grid
    print("CHANNEL GRID:")
    print("-" * 80)
    print("Ch# | Set A (Bank 0)      | Set B (Bank 1)      ")
    print("    | λ (nm)  | f (THz)   | λ (nm)  | f (THz)   ")
    print("-" * 80)
    
    for channel_data in grid_data['wavelength_grid']['channels']:
        ch = channel_data['channel']
        set_a_wl = channel_data['Set_A']['wavelength_nm']
        set_a_freq = channel_data['Set_A']['frequency_THz']
        set_b_wl = channel_data['Set_B']['wavelength_nm']
        set_b_freq = channel_data['Set_B']['frequency_THz']
        
        print(f" {ch}  | {set_a_wl:7.2f} | {set_a_freq:7.2f} | {set_b_wl:7.2f} | {set_b_freq:7.2f}")
    
    print("-" * 80)
    
    # Print center wavelengths
    print("\nCENTER WAVELENGTHS:")
    print("-" * 40)
    set_a_center = grid_data['center_wavelengths']['Set_A']['wavelength_nm']
    set_a_center_freq = grid_data['center_wavelengths']['Set_A']['frequency_THz']
    set_b_center = grid_data['center_wavelengths']['Set_B']['wavelength_nm']
    set_b_center_freq = grid_data['center_wavelengths']['Set_B']['frequency_THz']
    
    print(f"Set A (Bank 0): {set_a_center:.2f} nm ({set_a_center_freq:.2f} THz)")
    print(f"Set B (Bank 1): {set_b_center:.2f} nm ({set_b_center_freq:.2f} THz)")
    
    # Print spacing
    print("\nCHANNEL SPACING:")
    print("-" * 40)
    set_a_spacing = grid_data['channel_spacing']['Set_A']['wavelength_nm']
    set_a_spacing_freq = grid_data['channel_spacing']['Set_A']['frequency_THz']
    set_b_spacing = grid_data['channel_spacing']['Set_B']['wavelength_nm']
    set_b_spacing_freq = grid_data['channel_spacing']['Set_B']['frequency_THz']
    
    print(f"Set A (Bank 0): {set_a_spacing:.2f} nm ({set_a_spacing_freq:.2f} THz)")
    print(f"Set B (Bank 1): {set_b_spacing:.2f} nm ({set_b_spacing_freq:.2f} THz)")
    
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Load and print the wavelength grid
    grid = load_wavelength_grid()
    print_wavelength_grid_summary(grid)
    
    # Example usage of utility functions
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    
    # Get all wavelengths for Bank 0
    bank0_wavelengths = get_channel_wavelengths(0)
    print(f"Bank 0 wavelengths: {bank0_wavelengths}")
    
    # Get all frequencies for Bank 1
    bank1_frequencies = get_channel_frequencies(1)
    print(f"Bank 1 frequencies: {bank1_frequencies}")
    
    # Get specific channel value
    ch4_bank0_wl = get_channel_value(0, 4, 'wavelength')
    print(f"Channel 4, Bank 0 wavelength: {ch4_bank0_wl} nm")
    
    # Get center wavelengths
    center_wl_bank0 = get_center_wavelength(0)
    center_wl_bank1 = get_center_wavelength(1)
    print(f"Center wavelengths: Bank 0 = {center_wl_bank0} nm, Bank 1 = {center_wl_bank1} nm")
    
    # Get channel spacing
    spacing_bank0 = get_channel_spacing(0)
    print(f"Channel spacing Bank 0: {spacing_bank0} nm") 