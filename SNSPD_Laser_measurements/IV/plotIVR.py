#!/usr/bin/env python3
"""
IVR Plotting Script for SNSPD IV Data

Reads IV data from .txt files with columns (Voltage, Current, Resistance) and generates plots.
Output files are saved to SNSPD_analyzed_output/IV/ with a directory structure matching the input file location after SNSPD_rawdata or SNSPD_data.
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def determine_output_directory(input_filename: str, default_dir: str, stage: str = 'IV') -> str:
    """
    Determine output directory based on input file location, matching SelfTrigger.py logic.
    Uses 'IV' for IVR outputs.
    """
    abs_path = os.path.abspath(input_filename)
    path_parts = abs_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part in ['SNSPD_rawdata', 'SNSPD_data', 'SNSPD_analyzed_output']:
            base_path = os.sep.join(path_parts[:i])
            output_base_dir = os.path.join(base_path, 'SNSPD_analyzed_output', stage)
            subdirs = path_parts[i+1:-1]  # Get subdirectory structure
            if subdirs:
                return os.path.join(output_base_dir, *subdirs) + '/'
            return output_base_dir + '/'
    return default_dir + '/'

def read_iv_file(filename):
    """Read IVR data from a .txt file. Assumes columns: Voltage, Current, Resistance (tab or space separated)."""
    data = np.loadtxt(filename)
    if data.shape[1] < 3:
        raise ValueError("File must have at least 3 columns: Voltage, Current, Resistance")
    voltage = data[:, 0]
    current = data[:, 1]
    resistance = data[:, 2]
    return voltage, current, resistance

def plot_ivr(voltage, current, resistance, title, output_dir, base_name):
    if source_type == 'current':
        plt.figure(figsize=(10, 6))
        plt.plot(current, resistance, 'o-', color='green')
        plt.xlabel('Current (A)')
        plt.ylabel('Resistance (Ohm)')
        plt.title(f'R-I Curve\n{title}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_RI.png'), dpi=150)
        plt.close()
    elif source_type == 'voltage':
        plt.figure(figsize=(10, 6))
        plt.plot(voltage, resistance, 'o-', color='orange')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Resistance (Ohm)')
        plt.title(f'R-V Curve\n{title}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_RV.png'), dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot IV and RV curves from SNSPD IVR .txt files')
    parser.add_argument('in_filenames', nargs='+', help='Input IVR .txt files')
    parser.add_argument('--output_dir', '-d', default='.', help='Base output directory')
    args = parser.parse_args()

    for filename in args.in_filenames:
        base_name = Path(filename).stem
        output_dir = determine_output_directory(filename, args.output_dir, stage='IV')
        os.makedirs(output_dir, exist_ok=True)
        voltage, current, resistance = read_iv_file(filename)
        title = base_name
        # Determine source type from filename or path
        lower_path = str(filename).lower()
        if 'current_source' in lower_path:
            source_type = 'current'
        elif 'voltage_source' in lower_path:
            source_type = 'voltage'
        else:
            print(f"Warning: Could not determine source type from filename. Defaulting to voltage source (R-V plot).")
            source_type = 'voltage'
        plot_ivr(voltage, current, resistance, title, output_dir, base_name, source_type)
        print(f'✓ Plots saved to: {output_dir}')

if __name__ == '__main__':
    main()
