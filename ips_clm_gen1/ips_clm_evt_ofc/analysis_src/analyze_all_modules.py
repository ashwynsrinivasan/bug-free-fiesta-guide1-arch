#!/usr/bin/env python3
"""
Analyze all modules in the ips_clm_evt_ofc folder for both Endeavour and Kenya specifications
Includes State Validation and Mission Mode analysis
"""

from module_analysis import module_analysis

def main():
    # Initialize analysis
    base_path = "/Users/sashwyn06/gitsrc/friendly-system-lmi/src/bug-free-fiesta-guide1-arch/ips_clm/ips_clm_evt_ofc"
    analyzer = module_analysis(base_path)
    
    # Run State Validation analysis
    analyzer.state_validation()
    
    # Run Mission Mode analysis
    analyzer.mission_mode()
    
    # Run Alabama analysis (Kenya with Endeavour freq specs)
    analyzer.mission_mode_alabama()
    
    print("\n" + "=" * 80)
    print("All analysis completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
