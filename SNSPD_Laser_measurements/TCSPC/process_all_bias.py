#!/usr/bin/env python3
"""
Process all TCSPC data files with different bias voltages and create combined plots.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

# Data file information
data_files = {
    '70mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu',
    '74mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu',
    '78mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_78mV_20260205_0230.phu',
}

output_base = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3')

def extract_bias_voltage(filename):
    """Extract bias voltage from filename (e.g., '74mV' from 'SMSPD_3_2-7_500kHz_74mV_...')"""
    match = re.search(r'(\d+mV)', filename)
    return match.group(1) if match else None

def process_single_file(phu_file, output_dir, bias_voltage):
    """Process a single .phu file and save outputs to specified directory."""
    print(f"\n{'='*80}")
    print(f"Processing: {Path(phu_file).name}")
    print(f"Bias: {bias_voltage}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run read_phu.py with temporary output redirection
    script_path = Path(__file__).parent / 'read_phu.py'
    cmd = ['python3', str(script_path), phu_file]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Move generated plots to bias-specific folder
    source_dir = Path(phu_file).parent
    for png_file in source_dir.glob('*.png'):
        # Skip if already in output directory
        if png_file.parent == output_dir:
            continue
        dest = output_dir / png_file.name
        print(f"Moving: {png_file.name} -> {dest}")
        png_file.rename(dest)
    
    return output_dir

def load_fit_results(output_dir):
    """Extract fit results from the analysis output."""
    # Look for text files or parse from plot directory
    # For now, we'll need to extract from the PNG filenames or create a results file
    pass

def create_combined_plot(output_base):
    """Create combined plots comparing all bias voltages."""
    print(f"\n{'='*80}")
    print("Creating combined plots...")
    print(f"{'='*80}\n")
    
    combined_dir = output_base / 'combined'
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # This will be implemented after we collect data from individual runs
    # For now, create placeholder
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Combined plot will be generated after processing', 
            ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    output_path = combined_dir / 'combined_bias_comparison.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Combined plot saved: {output_path}")

def main():
    """Main processing function."""
    print("TCSPC Multi-Bias Voltage Analysis")
    print("="*80)
    
    # Process each bias voltage file
    results = {}
    for bias, phu_file in data_files.items():
        phu_path = Path(phu_file)
        if not phu_path.exists():
            print(f"Warning: File not found: {phu_file}")
            continue
        
        output_dir = output_base / bias
        process_single_file(phu_file, output_dir, bias)
        results[bias] = output_dir
    
    # Create combined comparison plots
    if results:
        create_combined_plot(output_base)
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Output directory: {output_base}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
