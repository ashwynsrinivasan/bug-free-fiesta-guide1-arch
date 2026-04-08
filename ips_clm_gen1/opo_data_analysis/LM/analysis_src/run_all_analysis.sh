#!/bin/bash

# =============================================================================
# Comprehensive LM Data Analysis Runner
# =============================================================================
# This script runs all test point analysis scripts (TP1-1 to TP3-3) and 
# generates comprehensive tile analysis plots.
#
# Usage: ./run_all_analysis.sh
# 
# The script will run the following analyses in order:
# 1. TP1-1: Basic wavelength and power analysis
# 2. TP1-2: LIV analysis and MPD responsivity
# 3. TP1-3: Temperature scan analysis
# 4. TP1-4: VOA analysis and tuning efficiency
# 5. TP2-0: Lensing station data analysis
# 6. TP2-1: VOA analysis with thermal crosstalk
# 7. TP2-2: Temperature tuning analysis
# 8. TP2-4: Wavelength setpoint and frequency error analysis
# 9. TP3-1: Combined laser and VOA analysis
# 10. TP3-3: Wavelength setpoint analysis for TP3-3 data
# 11. Tile Analysis: Comprehensive tile-specific plots
# =============================================================================

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Log file
LOGFILE="analysis_run_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output and log
log_and_print() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOGFILE"
}

# Function to run analysis with error handling
run_analysis() {
    local script_name=$1
    local description=$2
    local step_num=$3
    
    log_and_print "$BLUE" "="
    log_and_print "$BLUE" "Step $step_num: Running $description"
    log_and_print "$BLUE" "Script: $script_name"
    log_and_print "$BLUE" "Time: $(date)"
    log_and_print "$BLUE" "="
    
    if [ ! -f "$script_name" ]; then
        log_and_print "$RED" "❌ ERROR: Script $script_name not found!"
        return 1
    fi
    
    # Check if script is executable
    if [ ! -x "$script_name" ]; then
        log_and_print "$YELLOW" "⚠️  Making $script_name executable..."
        chmod +x "$script_name"
    fi
    
    # Run the script using python
    log_and_print "$CYAN" "🚀 Starting $script_name..."
    start_time=$(date +%s)
    
    if python3 "$script_name" 2>&1 | tee -a "$LOGFILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_and_print "$GREEN" "✅ $description completed successfully in ${duration}s"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_and_print "$RED" "❌ $description failed after ${duration}s"
        return 1
    fi
}

# Function to check Python environment
check_environment() {
    log_and_print "$PURPLE" "🔍 Checking Python environment..."
    
    if ! command -v python3 &> /dev/null; then
        log_and_print "$RED" "❌ ERROR: python3 not found!"
        exit 1
    fi
    
    python_version=$(python3 --version)
    log_and_print "$GREEN" "✅ Found $python_version"
    
    # Check for required modules
    required_modules=("pandas" "numpy" "matplotlib" "scipy" "plotly" "xarray")
    
    for module in "${required_modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            log_and_print "$GREEN" "✅ $module: Available"
        else
            log_and_print "$RED" "❌ $module: Missing"
            log_and_print "$YELLOW" "⚠️  Install with: pip install $module"
        fi
    done
}

# Function to print summary
print_summary() {
    local successful_runs=$1
    local total_runs=$2
    local failed_scripts=("${@:3}")
    
    log_and_print "$BLUE" ""
    log_and_print "$BLUE" "="
    log_and_print "$BLUE" "ANALYSIS COMPLETE - SUMMARY"
    log_and_print "$BLUE" "="
    log_and_print "$PURPLE" "📊 Analysis Results:"
    log_and_print "$GREEN" "   ✅ Successful: $successful_runs/$total_runs"
    
    if [ ${#failed_scripts[@]} -gt 0 ]; then
        log_and_print "$RED" "   ❌ Failed: ${#failed_scripts[@]}/$total_runs"
        log_and_print "$RED" "   Failed scripts: ${failed_scripts[*]}"
    fi
    
    log_and_print "$CYAN" "📁 Output locations:"
    log_and_print "$CYAN" "   • Individual plots: plots/TP*-*/"
    log_and_print "$CYAN" "   • Combined plots: plots/"
    log_and_print "$CYAN" "   • Tile analysis: plots/Tiles/"
    log_and_print "$CYAN" "   • Data files: data/"
    log_and_print "$CYAN" "   • Log file: $LOGFILE"
    
    log_and_print "$PURPLE" "🕒 Total runtime: $(($(date +%s) - script_start_time))s"
    log_and_print "$BLUE" "="
}

# Main execution starts here
script_start_time=$(date +%s)

log_and_print "$PURPLE" "🚀 COMPREHENSIVE LM DATA ANALYSIS RUNNER"
log_and_print "$PURPLE" "=========================================="
log_and_print "$PURPLE" "Start time: $(date)"
log_and_print "$PURPLE" "Working directory: $SCRIPT_DIR"
log_and_print "$PURPLE" "Log file: $LOGFILE"
log_and_print "$PURPLE" "=========================================="

# Check environment
check_environment

# Initialize counters
successful_runs=0
total_runs=11
failed_scripts=()

# Create necessary directories
log_and_print "$YELLOW" "📁 Creating output directories..."
mkdir -p plots data
mkdir -p plots/{TP1-1,TP1-2,TP1-3,TP1-4,TP2-0,TP2-1,TP2-2,TP2-4,TP3-1,TP3-3,Tiles}

# Run all analysis scripts in order
log_and_print "$PURPLE" ""
log_and_print "$PURPLE" "🎯 Starting analysis pipeline..."

# TP1-1: Basic analysis
if run_analysis "tp1p1.py" "TP1-1: Basic wavelength and power analysis" 1; then
    ((successful_runs++))
else
    failed_scripts+=("tp1p1.py")
fi

# TP1-2: LIV analysis  
if run_analysis "tp1p2.py" "TP1-2: LIV analysis and MPD responsivity" 2; then
    ((successful_runs++))
else
    failed_scripts+=("tp1p2.py")
fi

# TP1-3: Temperature scan
if run_analysis "tp1p3.py" "TP1-3: Temperature scan analysis" 3; then
    ((successful_runs++))
else
    failed_scripts+=("tp1p3.py")
fi

# TP1-4: VOA analysis
if run_analysis "tp1p4.py" "TP1-4: VOA analysis and tuning efficiency" 4; then
    ((successful_runs++))
else
    failed_scripts+=("tp1p4.py")
fi

# TP2-0: Lensing station
if run_analysis "tp2p0.py" "TP2-0: Lensing station data analysis" 5; then
    ((successful_runs++))
else
    failed_scripts+=("tp2p0.py")
fi

# TP2-1: VOA thermal crosstalk
if run_analysis "tp2p1.py" "TP2-1: VOA analysis with thermal crosstalk" 6; then
    ((successful_runs++))
else
    failed_scripts+=("tp2p1.py")
fi

# TP2-2: Temperature tuning
if run_analysis "tp2p2.py" "TP2-2: Temperature tuning analysis" 7; then
    ((successful_runs++))
else
    failed_scripts+=("tp2p2.py")
fi

# TP2-4: Wavelength setpoint
if run_analysis "tp2p4.py" "TP2-4: Wavelength setpoint and frequency error analysis" 8; then
    ((successful_runs++))
else
    failed_scripts+=("tp2p4.py")
fi

# TP3-1: Combined laser/VOA analysis
if run_analysis "tp3p1.py" "TP3-1: Combined laser and VOA analysis" 9; then
    ((successful_runs++))
else
    failed_scripts+=("tp3p1.py")
fi

# TP3-3: Wavelength setpoint for TP3-3
if run_analysis "tp3p3.py" "TP3-3: Wavelength setpoint analysis for TP3-3 data" 10; then
    ((successful_runs++))
else
    failed_scripts+=("tp3p3.py")
fi

# Tile Analysis: Comprehensive tile plots
if run_analysis "tile_analysis.py" "Tile Analysis: Comprehensive tile-specific plots" 11; then
    ((successful_runs++))
else
    failed_scripts+=("tile_analysis.py")
fi

# Print final summary
print_summary $successful_runs $total_runs "${failed_scripts[@]}"

# Exit with appropriate code
if [ $successful_runs -eq $total_runs ]; then
    log_and_print "$GREEN" "🎉 All analyses completed successfully!"
    exit 0
else
    log_and_print "$YELLOW" "⚠️  Some analyses failed. Check the log for details."
    exit 1
fi 