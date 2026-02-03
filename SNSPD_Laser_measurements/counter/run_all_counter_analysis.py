#!/usr/bin/env python3
"""
Run counter analysis on all measurement folders using configuration from JSON file.
"""

import subprocess
import sys
from pathlib import Path
import json


def normalize_list_arg(value):
    """Convert list-like or string arguments into CLI-friendly strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ','.join(map(str, value))
    return str(value)


def build_command(script_name, data_folder, measurement_name, bias_voltages, powers='all',
                 dark_subtract_mode='closest', linear_fit='false', fit_range='all',
                 fit_line_range='all', loglog='false', yaxis_scale='auto'):
    """Build the CLI command for a counter analysis run."""
    cmd = [
        sys.executable, script_name, data_folder,
        '--bias', normalize_list_arg(bias_voltages),
        '--powers', normalize_list_arg(powers)
    ]
    if dark_subtract_mode:
        cmd.extend(['--dark-subtract-mode', str(dark_subtract_mode)])
    if linear_fit:
        cmd.extend(['--linear-fit', str(linear_fit)])
    if fit_range:
        cmd.extend(['--fit-range', str(fit_range)])
    if fit_line_range:
        cmd.extend(['--fit-line-range', str(fit_line_range)])
    if loglog:
        cmd.extend(['--loglog', str(loglog)])
    if yaxis_scale:
        cmd.extend(['--yaxis-scale', str(yaxis_scale)])
    if measurement_name:
        cmd.extend(['--measurement-name', str(measurement_name)])
    return cmd

def load_config(config_path='counter_analysis_config.json'):
    """Load the counter analysis configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_analysis(script_name, data_folder, measurement_name, bias_voltages, powers='all',
                dark_subtract_mode='closest', linear_fit='false', fit_range='all',
                fit_line_range='all', loglog='false', yaxis_scale='auto'):
    """Run a counter analysis script and return the result."""
    cmd = build_command(
        script_name, data_folder, measurement_name, bias_voltages, powers,
        dark_subtract_mode, linear_fit, fit_range, fit_line_range, loglog, yaxis_scale
    )

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        return None

    print(result.stdout)
    return result.returncode == 0


def print_config_summary(config):
    """Print a summary of the current configuration."""
    print("="*80)
    print("Configuration loaded from counter_analysis_config.json")
    print("="*80)
    for name, settings in config['measurements'].items():
        enabled_status = "✓" if settings.get('enabled', 'true').lower() == 'true' else "✗"
        print(f"\n[{enabled_status}] {name}:")
        print(f"  Bias voltages: {settings['bias_voltages']} mV")
        print(f"  Powers: {settings.get('powers', 'all')} nW")
        print(f"  Dark subtract mode: {settings.get('dark_subtract_mode', 'closest')}")
        print(f"  Linear fit: {settings.get('linear_fit', 'false')}")
        print(f"  Fit range: {settings.get('fit_range', 'all')}")
        print(f"  Fit line range: {settings.get('fit_line_range', 'all')}")
        print(f"  Log-log scale: {settings.get('loglog', 'false')}")
        print(f"  Description: {settings['description']}")

def main():
    """Run all counter analyses from config file."""
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: counter_analysis_config.json not found!")
        print("Please create the config file first.")
        return 1
    
    base_path = Path(config['base_path'])
    
    # Build analysis list from config
    analyses = []
    for name, settings in config['measurements'].items():
        # Skip disabled measurements
        if settings.get('enabled', 'true').lower() != 'true':
            continue
        
        analyses.append({
            'name': name,
            'script': 'plot_counter_generic.py',
            'folder': base_path / settings['folder'],
            'measurement_name': name,
            'bias': settings['bias_voltages'],
            'powers': settings.get('powers', 'all'),
            'dark_subtract_mode': settings.get('dark_subtract_mode', 'closest'),
            'linear_fit': settings.get('linear_fit', 'false'),
            'fit_range': settings.get('fit_range', 'all'),
            'fit_line_range': settings.get('fit_line_range', 'all'),
            'loglog': settings.get('loglog', 'false'),
            'yaxis_scale': settings.get('yaxis_scale', 'auto')
        })
    
    # Print configuration summary
    print_config_summary(config)
    
    print("="*80)
    print("Running Counter Data Analysis on All Measurement Folders")
    print("="*80)
    
    successful = []
    failed = []
    
    for analysis in analyses:
        print(f"\n{'='*80}")
        print(f"Processing: {analysis['name']}")
        print(f"Folder: {analysis['folder']}")
        print(f"{'='*80}")
        
        if not analysis['folder'].exists():
            print(f"WARNING: Folder does not exist, skipping...")
            failed.append(analysis['name'])
            continue
        
        success = run_analysis(
            analysis['script'],
            str(analysis['folder']),
            analysis['measurement_name'],
            analysis['bias'],
            analysis['powers'],
            analysis['dark_subtract_mode'],
            analysis['linear_fit'],
            analysis['fit_range'],
            analysis['fit_line_range'],
            analysis['loglog'],
            analysis.get('yaxis_scale', 'auto')
        )
        
        if success:
            successful.append(analysis['name'])
        else:
            failed.append(analysis['name'])
    
    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"\nSuccessful ({len(successful)}):")
    for name in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
 
if __name__ == "__main__":
    main()
