#!/usr/bin/env python3
"""
Plot rotation degrees vs power for attenuation measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_rotation_data(filepath):
    """Load rotation angle and power data from file."""
    data = np.loadtxt(filepath)
    angles = data[:, 0]
    powers = data[:, 1]
    return angles, powers


def plot_rotation_power(filepath, output_dir=None, logscale=False):
    """
    Plot rotation degrees vs power.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the rotation data file
    output_dir : str or Path, optional
        Directory to save the plot (default: same as input file)
    logscale : bool
        Use log scale for power axis
    """
    filepath = Path(filepath)
    
    # Load data
    angles, powers = load_rotation_data(filepath)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data
    ax.plot(angles, powers, 'o:', linewidth=3, markersize=10, alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=18)
    ax.set_ylabel('Power (nW)', fontsize=18)
    ax.set_title(f'Rotation vs Power: {filepath.stem}', fontsize=20, fontweight='bold')
    
    # Increase tick label size
    ax.tick_params(axis='both', labelsize=14)
    
    # Apply log scale if requested
    if logscale:
        ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = filepath.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{filepath.stem}_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    # Show plot
    plt.show()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Plot rotation degrees vs power for attenuation measurements'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to rotation data file (tab-separated: angle power)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plot (default: same as input file)'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Use logarithmic scale for power axis'
    )
    
    args = parser.parse_args()
    
    plot_rotation_power(args.input_file, args.output_dir, args.log)


if __name__ == '__main__':
    main()
