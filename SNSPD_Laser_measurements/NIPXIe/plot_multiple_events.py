#!/usr/bin/env python3
"""
Plot signal waveforms from multiple single-event JSON files.

Usage:
    python plot_multiple_events.py <directory> [--event_num 1]
    
Example:
    python plot_multiple_events.py ~/SNSPD_analyzed_json/SMSPD_3/Laser/2-7/20251215/6K/Pulse/515/10000kHz/629nW/0degrees/
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import setup_atlas_style

import matplotlib.pyplot as plt
import numpy as np

# Apply ATLAS plotting style
setup_atlas_style()


def extract_resistance_from_note(note):
    """
    Extract power resistor value from Additional Note.
    
    Examples:
        "10ohm power resistor, 10k ohm shunt" -> "10Ω"
        "50ohm power resistor, ..." -> "50Ω"
    """
    if not note:
        return "Unknown"
    
    match = re.search(r'(\d+\.?\d*)ohm power resistor', note, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Ω"
    return "Unknown"


def read_event_file(filepath):
    """Read single event JSON file and extract relevant data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    event_info = data.get('event_info', {})
    waveform = data.get('waveform_data', {})
    
    # Extract resistance from Additional Note
    note = metadata.get('Additional Note', '')
    resistance_label = extract_resistance_from_note(note)
    
    # Get bias voltage and current for additional label
    bias_voltage = metadata.get('Bias Voltage (mV)', 'N/A')
    bias_current = metadata.get('Bias Current (uA)', 'N/A')
    
    # Create label
    label = f"{resistance_label} ({bias_voltage}mV, {bias_current}uA)"
    
    return {
        'time': np.array(waveform.get('time', [])),
        'signal': np.array(waveform.get('signal_channel', [])),
        'trigger': np.array(waveform.get('trigger_channel', [])),
        'label': label,
        'resistance': resistance_label,
        'bias_voltage': bias_voltage,
        'bias_current': bias_current,
        'event_number': event_info.get('event_number', -1),
        'sample_interval': event_info.get('sample_interval', 0),
        'filename': os.path.basename(filepath)
    }


def find_event_files(directory, event_number=None):
    """
    Find all event JSON files in directory.
    
    Args:
        directory: Directory to search
        event_number: If specified, only return files with this event number
    
    Returns:
        List of file paths
    """
    directory = Path(directory).expanduser()
    
    if event_number is not None:
        pattern = f"*_event{event_number}.json"
    else:
        pattern = "*_event*.json"
    
    files = list(directory.rglob(pattern))
    return sorted(files)


def plot_events(event_files, event_number=None):
    """
    Plot signal waveforms from multiple event files.
    
    Args:
        event_files: List of event JSON file paths
        event_number: Event number being plotted (for title)
    """
    if not event_files:
        print("No event files found!")
        return
    
    print(f"Found {len(event_files)} event files")
    
    # Setup ATLAS style (if available)
    try:
        import sys
        sys.path.insert(0, '../counter')
        from plot_counter_generic import setup_atlas_style
        setup_atlas_style()
    except:
        # Use default style if ATLAS style not available
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 6
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use distinguishable colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_files)))
    
    for idx, filepath in enumerate(event_files):
        print(f"  Loading: {os.path.basename(filepath)}")
        
        try:
            event_data = read_event_file(filepath)
            
            # Convert time to nanoseconds for better readability
            time_ns = event_data['time'] * 1e9
            signal = event_data['signal']
            
            # Plot signal
            ax.plot(time_ns, signal, 
                   label=event_data['label'],
                   color=colors[idx],
                   alpha=0.8,
                   linewidth=1.5)
            
            print(f"    Resistance: {event_data['resistance']}, "
                  f"Bias: {event_data['bias_voltage']}mV, {event_data['bias_current']}uA, "
                  f"Event: {event_data['event_number']}")
            
        except Exception as e:
            print(f"    Error reading {filepath}: {e}")
            continue
    
    # Configure plot
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Signal Amplitude (V)')
    
    event_str = f"Event {event_number}" if event_number is not None else "Events"
    ax.text(0.05, 0.95, f'Signal Waveforms - {event_str}', 
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top')
    
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Enable minor ticks
    ax.minorticks_on()
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f'event_comparison_event{event_number}.png' if event_number is not None else 'event_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_filename}")
    
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot signal waveforms from multiple single-event JSON files')
    parser.add_argument('directory', type=str, 
                       help='Directory containing event JSON files')
    parser.add_argument('--event_num', '-e', type=int, default=None,
                       help='Event number to plot (default: all events)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available event files and exit')
    
    args = parser.parse_args()
    
    # Find event files
    event_files = find_event_files(args.directory, args.event_num)
    
    if args.list:
        print(f"Found {len(event_files)} event files in {args.directory}:")
        for f in event_files:
            print(f"  {f}")
        return
    
    if not event_files:
        print(f"No event files found in {args.directory}")
        if args.event_num is not None:
            print(f"  Searched for: *_event{args.event_num}.json")
        else:
            print(f"  Searched for: *_event*.json")
        return
    
    # Plot events
    plot_events(event_files, args.event_num)


if __name__ == "__main__":
    main()
