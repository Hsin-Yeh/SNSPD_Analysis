#!/usr/bin/env python3
"""
Wrapper script for Counter_sweep_power_3 analysis.
Calls plot_counter_generic.py with appropriate parameters.
"""

import sys
import os
from pathlib import Path

def main():
    """Run Counter_sweep_power_3 analysis using the generic script."""
    # Change to script directory to ensure imports work
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Import the generic script
    import plot_counter_generic
    
    # Set up arguments for Counter_sweep_power_3
    data_folder = '/Users/ya/Documents/Projects/SNSPD/SNSPD_data/SMSPD_3/Counter_sweep_power_3/2-7/6K'
    bias_voltages = '66,68,70,72,74'
    remove_lowest = '3'
    tolerance = '1.5'
    
    # Override sys.argv to simulate command line arguments
    sys.argv = [
        'plot_counter_generic.py',
        data_folder,
        '--bias', bias_voltages,
        '--remove-lowest', remove_lowest,
        '--tolerance', tolerance
    ]
    
    # Call the generic main function
    plot_counter_generic.main()

if __name__ == "__main__":
    main()
