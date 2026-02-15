#!/usr/bin/env python3
"""
Reorganize TCSPC folder structure
"""
import os
import shutil
from pathlib import Path

# Define base directory
base_dir = Path("/Users/ya/Documents/Projects/SNSPD/SNSPD_Analysis/SNSPD_Laser_measurements/TCSPC")

# Define file categories
bias_sweep_files = [
    "analyze_bias_sweep.py",
    "compare_bias_sweeps.py",
    "process_all_bias.py",
    "run_all_bias_analysis.py",
]

power_sweep_files = [
    "analyze_power_dependence.py",
    "compare_power_sweeps.py",
    "plot_photon_number_vs_power.py",
    "compare_cuts.py",
    "compare_tail_cuts.py",
    "run_power_comparison.sh",
]

photon_peak_fitting_files = [
    "fit_all_blocks_75p7_limits.py",
    "fit_all_blocks_75p7.py",
    "fit_all_blocks_constrained.py",
    "fit_all_blocks_mean_constrained.py",
    "fit_template_plus_3gaussians.py",
    "analyze_n1_spectrum.py",
    "photon_number_analysis.py",
    "create_combined_plot.py",
    "peak_position_summary.py",
]

analysis_summary_files = [
    "ANALYSIS_SUMMARY_75P7.md",
    "FITTING_SUMMARY_75P7.txt",
    "IMPLEMENTATION_SUMMARY.md",
    "PHOTON_ANALYSIS_QUICKSTART.md",
    "PHOTON_NUMBER_ANALYSIS_IMPLEMENTATION.md",
    "TCSPC_RESULTS_SUMMARY_20260206.md",
    "fit_all_blocks_output.txt",
    "commit_message.txt",
]

# Test files to remove
test_files_to_remove = [
    "test_75p7_cut.py",
    "test_75p7_subprocess.py",
    "test_block10_bounded_fit.py",
    "test_block10_gaussian_fullwindow.py",
    "test_block10_multigaussian.py",
    "test_block10_tailcut_fit.py",
    "test_block120_loose_bounds.py",
    "test_block145_gaussian.py",
    "test_block180_2peaks.py",
    "test_block180_5peaks.py",
    "test_fixed_means.py",
    "test_peak_identification_block150.py",
    "test_photon_analysis.py",
    "test_photon_peaks_block150.py",
    "test_reference_photon_peaks.py",
]

# Redundant files to remove
redundant_files = [
    "plot_block180_raw.py",
    "template_fit_block180.py",
    "extract_block145_reference.py",
    "verify_results.py",
    "results_generator.py",
    "analyze_dark_count_statistics.py",
    "generate_histograms.py",
    "generate_histograms_from_analysis.py",
]

def move_files(file_list, destination):
    """Move files to destination folder"""
    dest_path = base_dir / destination
    dest_path.mkdir(exist_ok=True)
    
    moved = []
    skipped = []
    
    for filename in file_list:
        src = base_dir / filename
        if src.exists():
            try:
                dst = dest_path / filename
                shutil.move(str(src), str(dst))
                moved.append(filename)
                print(f"✓ Moved {filename} to {destination}/")
            except Exception as e:
                print(f"✗ Error moving {filename}: {e}")
                skipped.append(filename)
        else:
            skipped.append(filename)
            print(f"⊗ Skipped {filename} (not found)")
    
    return moved, skipped

def remove_files(file_list, description):
    """Remove files"""
    removed = []
    skipped = []
    
    for filename in file_list:
        src = base_dir / filename
        if src.exists():
            try:
                src.unlink()
                removed.append(filename)
                print(f"✓ Removed {filename} ({description})")
            except Exception as e:
                print(f"✗ Error removing {filename}: {e}")
                skipped.append(filename)
        else:
            skipped.append(filename)
    
    return removed, skipped

# Main execution
print("=" * 80)
print("REORGANIZING TCSPC FOLDER")
print("=" * 80)

print("\n1. Moving bias_sweep files...")
move_files(bias_sweep_files, "bias_sweep")

print("\n2. Moving power_sweep files...")
move_files(power_sweep_files, "power_sweep")

print("\n3. Moving photon_peak_fitting files...")
move_files(photon_peak_fitting_files, "photon_peak_fitting")

print("\n4. Moving analysis_summaries files...")
move_files(analysis_summary_files, "analysis_summaries")

print("\n5. Removing test files...")
removed_test, _ = remove_files(test_files_to_remove, "test file")

print("\n6. Removing redundant files...")
removed_redundant, _ = remove_files(redundant_files, "redundant")

print("\n" + "=" * 80)
print("REORGANIZATION COMPLETE")
print("=" * 80)
print(f"Test files removed: {len(removed_test)}")
print(f"Redundant files removed: {len(removed_redundant)}")
print("\nRemaining files in TCSPC root:")
remaining = [f for f in os.listdir(base_dir) if f.endswith('.py')]
for f in sorted(remaining):
    print(f"  - {f}")
