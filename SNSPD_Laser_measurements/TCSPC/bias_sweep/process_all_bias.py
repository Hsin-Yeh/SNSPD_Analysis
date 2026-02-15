#!/usr/bin/env python3
"""
Process all TCSPC data files with different bias voltages and create combined plots.
Runs read_phu.py on each bias voltage, then runs create_combined_plot.py for comparison.
"""

import argparse
import subprocess
from pathlib import Path
import re


def find_bias_files(base_path: Path):
    """Find all PHU files with bias voltage info under base_path."""
    phu_files = {}
    for phu_file in base_path.rglob("*.phu"):
        filename = phu_file.stem
        match = re.search(r'(\d+mV)', filename)
        if match:
            bias = match.group(1)
            phu_files.setdefault(bias, []).append(phu_file)
    return phu_files


def parse_bias_to_block_id(bias_voltage: str):
    """Convert bias voltage string like '70mV' to block ID integer (70)."""
    try:
        return int(bias_voltage.replace('mV', ''))
    except Exception:
        return None


def process_single_file(phu_file, bias_voltage, block0_reference_file=None, output_base=None):
    """Process a single .phu file using read_phu.py which auto-saves to organized directories."""
    print(f"\n{'='*80}")
    print(f"Processing: {Path(phu_file).name}")
    print(f"Bias: {bias_voltage}")
    print(f"{'='*80}\n")

    script_path = Path(__file__).parent / 'read_phu.py'
    cmd = ['python3', str(script_path), str(phu_file)]
    block0_block = parse_bias_to_block_id(bias_voltage)
    if block0_reference_file and block0_block is not None:
        cmd += ['--block0-file', str(block0_reference_file), '--block0-block', str(block0_block)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"⚠️  Warning: read_phu.py returned exit code {result.returncode}")
        return None

    if output_base is None:
        return None

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
    parser = argparse.ArgumentParser(
        description="Process TCSPC bias sweeps and generate combined plots."
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default="/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/sweep_power",
        help="Base path to search for .phu files (default: /Users/ya/SNSPD_rawdata)",
    )
    parser.add_argument(
        "--block0-file",
        default="/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/sweep_bias/SMSPD_3_2-7_500kHz_0nW_20260205_0518.phu",
        help="Optional block-0 reference .phu file",
    )
    parser.add_argument(
        "--output-base",
        default="/Users/ya/SNSPD_analyzed_output/TCSPC/SMSPD_3",
        help="Output base directory",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser()
    if not base_path.exists():
        print(f"Error: Base path not found: {base_path}")
        return

    block0_reference_file = Path(args.block0_file).expanduser() if args.block0_file else None
    if block0_reference_file and not block0_reference_file.exists():
        print(f"Warning: Block-0 reference file not found: {block0_reference_file}")
        block0_reference_file = None

    output_base = Path(args.output_base).expanduser()

    print("TCSPC Multi-Bias Voltage Analysis")
    print("="*80)
    print(f"Output base: {output_base}")
    print(f"Search base: {base_path}")
    print("="*80)

    bias_files = find_bias_files(base_path)
    if not bias_files:
        print("⚠️  No .phu files with bias voltage found")
        return

    results = {}
    sorted_biases = sorted(bias_files.keys(), key=lambda x: int(x.replace('mV', '')))
    for bias in sorted_biases:
        phu_file = bias_files[bias][0]
        output_dir = process_single_file(phu_file, bias, block0_reference_file, output_base)
        if output_dir:
            results[bias] = output_dir

    if results:
        print(f"\nProcessed {len(results)} bias voltages successfully")
        create_combined_plot(output_base)
    else:
        print("\n⚠️  No files were successfully processed")

    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Individual outputs: {output_base}/power_sweep/{{bias}}/")
    print(f"Combined plots: {output_base}/power_sweep/combined/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
