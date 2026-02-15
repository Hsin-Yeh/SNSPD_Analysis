#!/usr/bin/env python3
"""
Run TCSPC analysis on all available bias voltage measurements
and generate consolidated result files.
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

def run_analysis(phu_file):
    """Run read_phu.py on a single PHU file."""
    cmd = ['python3', 'read_phu.py', str(phu_file)]
    print(f"\n{'='*70}")
    print(f"Processing: {phu_file.name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"✓ Analysis completed successfully")
            return True
        else:
            print(f"✗ Analysis failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error running analysis: {e}")
        return False

def find_bias_files(base_path: Path):
    """Find all PHU files with bias voltage info."""
    phu_files = {}

    for phu_file in base_path.rglob("*.phu"):
        # Extract bias voltage from filename
        filename = phu_file.stem
        # Look for pattern like "70mV"
        import re
        match = re.search(r'(\d+mV)', filename)
        if match:
            bias = match.group(1)
            if bias not in phu_files:
                phu_files[bias] = []
            phu_files[bias].append(phu_file)
    
    return phu_files

def main():
    parser = argparse.ArgumentParser(
        description="Run TCSPC analysis on all available bias voltage measurements."
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default="/Users/ya/SNSPD_rawdata",
        help="Base path to search for .phu files (default: /Users/ya/SNSPD_rawdata)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser()
    if not base_path.exists():
        print(f"Error: Base path not found: {base_path}")
        sys.exit(1)

    print("TCSPC BATCH ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all bias files
    bias_files = find_bias_files(base_path)
    
    if not bias_files:
        print("No PHU files with bias voltage found")
        sys.exit(1)
    
    # Sort by bias voltage
    sorted_biases = sorted(bias_files.keys(), key=lambda x: int(x.replace('mV', '')))
    
    results_summary = {}
    successful = 0
    failed = 0
    
    # Run analysis on each bias voltage (use first file for each)
    for bias in sorted_biases:
        phu_file = bias_files[bias][0]  # Use first file for this bias
        success = run_analysis(phu_file)
        
        if success:
            successful += 1
            results_summary[bias] = "SUCCESS"
        else:
            failed += 1
            results_summary[bias] = "FAILED"
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total biases analyzed: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults by bias:")
    for bias in sorted_biases:
        status = results_summary.get(bias, "UNKNOWN")
        print(f"  {bias:>6s}: {status}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed == 0:
        print("\n✓ All analyses completed successfully!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} analysis/analyses failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
