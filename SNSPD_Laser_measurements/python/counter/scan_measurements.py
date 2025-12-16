#!/usr/bin/env python3
"""
Scan for counter measurement folders and display their configuration.
Can also detect new folders not yet in the config file.
"""

import json
from pathlib import Path
import argparse
import re


def save_config_compact(config, config_path):
    """Save config file with compact array formatting (arrays on single lines)"""
    output = json.dumps(config, indent=2)
    
    # Replace multi-line arrays with single-line arrays
    # Match arrays with any number of numeric elements
    def compact_array(match):
        # Extract all numbers from the array
        array_content = match.group(0)
        numbers = re.findall(r'\d+', array_content)
        return '[' + ', '.join(numbers) + ']'
    
    # Match arrays that span multiple lines with only numbers
    output = re.sub(r'\[\s*\n\s*\d+[,\s\n\d]*\]', compact_array, output, flags=re.MULTILINE)
    
    with open(config_path, 'w') as f:
        f.write(output)


def load_config(config_path):
    """Load the counter analysis configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def scan_measurement_folders(base_path):
    """
    Scan the base path for measurement folders.
    Returns a dict of {folder_name: full_path}
    """
    base = Path(base_path)
    if not base.exists():
        print(f"Warning: Base path does not exist: {base_path}")
        return {}
    
    measurements = {}
    # Look for folders that contain subdirectories matching the pattern */2-7/6K
    for item in base.iterdir():
        if item.is_dir():
            # Check if this folder has the expected structure
            potential_path = item / "2-7" / "6K"
            if potential_path.exists():
                measurements[item.name] = str(potential_path)
    
    return measurements


def get_power_levels(measurement_path):
    """
    Scan a measurement folder for power level subdirectories.
    Returns a sorted list of power values (in nW).
    """
    path = Path(measurement_path)
    if not path.exists():
        return []
    
    powers = []
    for item in path.iterdir():
        if item.is_dir() and item.name.endswith('nW'):
            try:
                # Extract power value from folder name (e.g., "100nW" -> 100)
                power_str = item.name.replace('nW', '')
                power = int(power_str)
                powers.append(power)
            except ValueError:
                continue
    
    return sorted(powers)


def get_available_bias_voltages(measurement_path):
    """
    Scan data files to find all available bias voltages.
    Returns a sorted list of unique bias voltages found in the data files.
    """
    path = Path(measurement_path)
    if not path.exists():
        return []
    
    bias_voltages = set()
    
    # Look through power folders for data files
    for power_folder in path.iterdir():
        if power_folder.is_dir() and power_folder.name.endswith('nW'):
            # Find .txt files in this power folder
            for data_file in power_folder.glob('*.txt'):
                try:
                    # Read the file and extract bias voltages from first column
                    with open(data_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#') or line.startswith('bias'):
                                continue
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                try:
                                    # Convert voltage to mV (data is in volts)
                                    bias_volts = float(parts[0])
                                    bias_mv = int(round(bias_volts * 1000))
                                    if bias_mv > 0:  # Skip 0 values
                                        bias_voltages.add(bias_mv)
                                except (ValueError, IndexError):
                                    continue
                    # Only need to read one file per folder
                    if bias_voltages:
                        break
                except Exception:
                    continue
            # If we found some bias voltages, we can stop
            if len(bias_voltages) > 10:  # Should have many bias points
                break
    
    return sorted(list(bias_voltages))


def print_config_summary(config):
    """Print a summary of the current configuration."""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nBase Path: {config['base_path']}\n")
    
    for name, settings in config['measurements'].items():
        enabled_status = "✓" if settings.get('enabled', 'true').lower() == 'true' else "✗"
        print(f"Measurement [{enabled_status}]: {name}")
        print(f"  Folder: {settings['folder']}")
        print(f"  Selected Bias Voltages: {settings['bias_voltages']} mV")
        print(f"  Selected Powers: {settings.get('powers', 'all')} nW")
        print(f"  Dark Subtract Mode: {settings.get('dark_subtract_mode', 'closest')}")
        print(f"  Linear Fit: {settings.get('linear_fit', 'false')}")
        print(f"  Fit Range: {settings.get('fit_range', 'all')}")
        print(f"  Fit Line Range: {settings.get('fit_line_range', 'all')}")
        print(f"  Log-log Scale: {settings.get('loglog', 'false')}")
        
        # Try to get actual power levels and bias voltages from filesystem
        full_path = Path(config['base_path']) / settings['folder']
        powers = get_power_levels(full_path)
        available_bias = get_available_bias_voltages(full_path)
        
        if powers:
            # Separate 0nW (dark) from signal powers
            dark = [p for p in powers if p == 0]
            signal = [p for p in powers if p != 0]
            print(f"  Available Power Levels: {signal} nW (+ {len(dark)} dark count folder(s))")
        else:
            print(f"  Available Power Levels: [Folder not found or empty]")
        
        if available_bias:
            print(f"  Available Bias Voltages: {available_bias} mV")
        
        print(f"  Description: {settings['description']}")
        print()
    
    print("="*70 + "\n")


def find_new_measurements(config):
    """
    Scan for measurement folders not in the config.
    Returns a list of folder names that are new.
    """
    # Get configured measurements
    configured = set(config['measurements'].keys())
    
    # Scan filesystem
    found = scan_measurement_folders(config['base_path'])
    found_names = set(found.keys())
    
    # Find new ones
    new_measurements = found_names - configured
    
    return sorted(new_measurements), found


def generate_config_template(measurement_name, measurement_path):
    """Generate a config template for a new measurement."""
    powers = get_power_levels(measurement_path)
    available_bias = get_available_bias_voltages(measurement_path)
    
    # Separate dark from signal powers
    signal_powers = [p for p in powers if p != 0]
    
    template = {
        "enabled": "true",
        "folder": measurement_name + "/2-7/6K",
        "bias_voltages": available_bias[:5] if available_bias else [66, 68, 70, 72, 74],  # Use first 5 or defaults
        "powers": "all",  # Default to all powers
        "dark_subtract_mode": "closest",  # Default to closest time
        "linear_fit": "false",  # Default to disabled
        "fit_range": "all",  # Default to all data
        "fit_line_range": "all",  # Default to full range
        "loglog": "false",  # Default to linear scale
        "description": f"Auto-detected: {len(powers)} power folders",
        "_available_bias_voltages": available_bias,  # Include all available for reference
        "_available_powers": signal_powers
    }
    
    return template


def main():
    parser = argparse.ArgumentParser(
        description="Scan counter measurement folders and manage configuration"
    )
    parser.add_argument(
        '--config',
        default='counter_analysis_config.json',
        help='Path to config file (default: counter_analysis_config.json)'
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan for new measurements not in config'
    )
    parser.add_argument(
        '--add-new',
        action='store_true',
        help='Add newly discovered measurements to config file'
    )
    parser.add_argument(
        '--update-metadata',
        action='store_true',
        help='Update _available_bias_voltages and _available_powers metadata for existing measurements'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create it first or specify the correct path with --config")
        return 1
    
    config = load_config(config_path)
    
    # Update metadata if requested
    if args.update_metadata:
        print("\n" + "="*70)
        print("UPDATING METADATA")
        print("="*70 + "\n")
        
        base_path = Path(config['base_path'])
        for name, settings in config['measurements'].items():
            full_path = base_path / settings['folder']
            powers = get_power_levels(full_path)
            available_bias = get_available_bias_voltages(full_path)
            
            # Separate dark from signal powers
            signal_powers = [p for p in powers if p != 0]
            
            # Update metadata fields
            config['measurements'][name]['_available_bias_voltages'] = available_bias
            config['measurements'][name]['_available_powers'] = signal_powers
            
            print(f"{name}:")
            print(f"  Available bias: {available_bias} mV")
            print(f"  Available powers: {signal_powers} nW")
        
        # Save updated config with compact formatting
        save_config_compact(config, config_path)
        
        print(f"\n✓ Updated metadata in {config_path}")
        print("="*70 + "\n")
        
        # Reload to show updated config
        config = load_config(config_path)
    
    # Print current config summary
    print_config_summary(config)
    
    # Scan for new measurements if requested
    if args.scan or args.add_new:
        new_measurements, all_found = find_new_measurements(config)
        
        if new_measurements:
            print("\n" + "="*70)
            print("NEW MEASUREMENTS DETECTED")
            print("="*70 + "\n")
            
            new_configs = {}
            for name in new_measurements:
                path = all_found[name]
                powers = get_power_levels(path)
                print(f"Measurement: {name}")
                print(f"  Path: {path}")
                print(f"  Power Levels: {powers} nW")
                print()
                
                # Generate template
                template = generate_config_template(name, path)
                new_configs[name] = template
            
            if args.add_new:
                # Add to config
                for name, template in new_configs.items():
                    config['measurements'][name] = template
                
                # Save updated config with compact formatting
                save_config_compact(config, config_path)
                
                print(f"\n✓ Added {len(new_configs)} new measurement(s) to {config_path}")
                print("  Please review and adjust bias_voltages and remove_lowest_points as needed.")
            else:
                print("\nTo add these measurements to the config, run with --add-new flag")
            
            print("="*70 + "\n")
        else:
            print("\n✓ No new measurements found. Config is up to date.\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
