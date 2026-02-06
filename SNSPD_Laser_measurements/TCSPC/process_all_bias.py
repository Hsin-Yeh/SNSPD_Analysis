#!/usr/bin/env python3
"""
Process all TCSPC data files with different bias voltages and create combined plots.
Runs read_phu.py on each bias voltage, then runs create_combined_plot.py for comparison.
"""

import subprocess
from pathlib import Path

# Data file information
data_files = {
    '66mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_66mV_20260205_0754.phu',
    '70mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_70mV_20260205_0122.phu',
    '73mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_73mV_20260205_1213.phu',
    '74mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_74mV_20260205_0102.phu',
    '78mV': '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_78mV_20260205_0230.phu',
}

# External block-0 reference file (0nW sweep)
block0_reference_file = '/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_0nW_20260205_0518.phu'

# Map bias voltages to block IDs in the 0nW file for Block 0 reference
block0_reference_blocks = {
    '66mV': 66,
    '70mV': 70,
    '73mV': 73,
    '74mV': 74,
    '78mV': 78,
}

output_base = Path('/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3')

def process_single_file(phu_file, bias_voltage):
    """Process a single .phu file using read_phu.py which auto-saves to organized directories."""
    print(f"\n{'='*80}")
    print(f"Processing: {Path(phu_file).name}")
    print(f"Bias: {bias_voltage}")
    print(f"{'='*80}\n")
    
    # Run read_phu.py - it will auto-create and save to ~/SNSPD_analyzed_output/TCSPC/SMSPD_3/power_sweep/{bias}/
    script_path = Path(__file__).parent / 'read_phu.py'
    cmd = ['python3', str(script_path), phu_file]
    block0_block = block0_reference_blocks.get(bias_voltage)
    if block0_block is not None:
        cmd += ['--block0-file', block0_reference_file, '--block0-block', str(block0_block)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"⚠️  Warning: read_phu.py returned exit code {result.returncode}")
        return None
    
    # Return the expected output directory (read_phu.py auto-creates this)
    expected_output = output_base / 'power_sweep' / bias_voltage
    if expected_output.exists():
        print(f"✓ Output saved to: {expected_output}")
        return expected_output
    else:
        print(f"⚠️  Expected output directory not found: {expected_output}")
        return None

def create_combined_plot(output_base):
    """Create combined plots comparing all bias voltages."""
    print(f"\n{'='*80}")
    print("Creating combined plots...")
    print(f"{'='*80}\n")
    
    # Use create_combined_plot.py which already handles this
    script_path = Path(__file__).parent / 'create_combined_plot.py'
    cmd = ['python3', str(script_path)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        combined_dir = output_base / 'power_sweep' / 'combined'
        print(f"✓ Combined plots saved to: {combined_dir}")
    else:
        print(f"⚠️  Warning: create_combined_plot.py returned exit code {result.returncode}")

def main():
    """Main processing function."""
    print("TCSPC Multi-Bias Voltage Analysis")
    print("="*80)
    print(f"Output base: {output_base}")
    print("="*80)
    
    # Process each bias voltage file
    results = {}
    for bias, phu_file in data_files.items():
        phu_path = Path(phu_file)
        if not phu_path.exists():
            print(f"⚠️  Warning: File not found: {phu_file}")
            continue
        
        output_dir = process_single_file(phu_file, bias)
        if output_dir:
            results[bias] = output_dir
    
    # Create combined comparison plots using create_combined_plot.py
    if results:
        print(f"\nProcessed {len(results)} bias voltages successfully")
        create_combined_plot(output_base)
    else:
        print("\n⚠️  No files were successfully processed")
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Individual outputs: {output_base}/power_sweep/{{66mV,70mV,73mV,74mV,78mV}}/")
    print(f"Combined plots: {output_base}/power_sweep/combined/")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
