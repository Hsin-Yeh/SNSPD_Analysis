"""
HEP (High Energy Physics) plotting style configuration for matplotlib
Based on ATLAS/ROOT plotting conventions

This module provides a centralized plotting style that can be imported
and used by all analysis scripts.

Usage:
    from plot_style import setup_hep_style
    setup_hep_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_hep_style():
    """
    Setup HEP (High Energy Physics) plotting style for matplotlib
    
    This function configures matplotlib to use a professional plotting style
    following HEP conventions (similar to ATLAS/ROOT plots), with:
    - Serif fonts (similar to ROOT)
    - Ticks on all four sides pointing inward
    - Bold lines and large markers
    - Minor ticks visible
    - Grid behind data
    - Appropriate margins and spacing
    """
    # Use white background with black elements
    plt.style.use('default')
    
    # Set figure and axes colors
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    
    # Set margins (as fraction of figure size)
    mpl.rcParams['figure.subplot.left'] = 0.15
    mpl.rcParams['figure.subplot.right'] = 0.88  # 1 - 0.12
    mpl.rcParams['figure.subplot.bottom'] = 0.16
    mpl.rcParams['figure.subplot.top'] = 0.95  # 1 - 0.05
    
    # Font settings - use serif font similar to ROOT
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14
    
    # Line and marker settings - bold style
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['lines.markeredgewidth'] = 1.5
    
    # Grid settings
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.8
    
    # Tick settings - ticks on all sides
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.2
    mpl.rcParams['ytick.major.width'] = 1.2
    mpl.rcParams['xtick.minor.width'] = 1.0
    mpl.rcParams['ytick.minor.width'] = 1.0
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    
    # Axes settings
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.axisbelow'] = True  # Grid behind data
    
    # Legend settings
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.95
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.edgecolor'] = 'black'
