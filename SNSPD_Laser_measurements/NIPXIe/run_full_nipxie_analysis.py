#!/usr/bin/env python3
"""
Run full NIPXIe SNSPD analysis pipeline (Stage 1 → Stage 2 → Stage 3).

Features:
- Accepts one or more input folders/files
- Recursively finds TDMS files (subfolders included)
- Runs SelfTrigger (Stage 1) → analyze_events (Stage 2) → plot_all (Stage 3)
- Lets you specify output folders
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set

# Local imports (Stage 1 & Stage 2 logic)
import SelfTrigger
import analyze_events


def _is_tdms(path: str) -> bool:
    return path.lower().endswith(".tdms")


def find_tdms_files(inputs: List[str], recursive: bool = True) -> List[str]:
    tdms_files = []
    for raw in inputs:
        path = os.path.abspath(raw)
        if os.path.isfile(path) and _is_tdms(path):
            tdms_files.append(path)
        elif os.path.isdir(path):
            if recursive:
                for root, _, files in os.walk(path):
                    for fname in files:
                        if _is_tdms(fname):
                            tdms_files.append(os.path.join(root, fname))
            else:
                for fname in os.listdir(path):
                    fpath = os.path.join(path, fname)
                    if os.path.isfile(fpath) and _is_tdms(fpath):
                        tdms_files.append(fpath)
        else:
            print(f"Warning: Input path not found or not a TDMS file: {raw}")
    return sorted(set(tdms_files))


def analysis_output_paths(tdms_files: List[str], stage1_output_dir: str) -> List[str]:
    outputs = []
    for tdms_path in tdms_files:
        basename = os.path.basename(tdms_path).replace(".tdms", "")
        out_dir = SelfTrigger.determine_output_directory(tdms_path, stage1_output_dir)
        outputs.append(os.path.join(out_dir, f"{basename}_analysis.json"))
    return outputs


def ensure_parent_dirs(paths: List[str]):
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)


def run_stage1(tdms_files: List[str], args) -> List[str]:
    if not tdms_files:
        return []

    print("=" * 70)
    print("Stage 1: SelfTrigger TDMS → event JSON")
    print("=" * 70)

    # Configure global flags in SelfTrigger
    SelfTrigger.DEBUG = args.debug_report
    SelfTrigger.DISPLAY = args.display_report

    # Build a minimal args namespace for SelfTrigger.process_tdms_file
    stage1_args = argparse.Namespace(
        report=args.report,
        subset=args.subset,
        checkSingleEvent=-1,
        find_sync_method=args.find_sync_method,
    )

    output_paths = []
    for idx, tdms_path in enumerate(tdms_files, start=1):
        print("\n" + "=" * 60)
        print(f"Processing: {tdms_path} ({idx}/{len(tdms_files)})")
        print("=" * 60)

        basename = os.path.basename(tdms_path).replace(".tdms", "")
        out_dir = SelfTrigger.determine_output_directory(tdms_path, args.stage1_output)
        SelfTrigger.createDir(out_dir)

        output_filename = os.path.join(out_dir, f"{basename}_analysis.json")
        voltage, current, resistance = SelfTrigger.extract_bias_parameters(basename)

        SelfTrigger.process_tdms_file(tdms_path, output_filename, stage1_args)
        SelfTrigger.save_analysis_results(output_filename, voltage, current, resistance)

        output_paths.append(output_filename)

    return output_paths


def run_stage2(analysis_jsons: List[str], args) -> List[str]:
    if not analysis_jsons:
        return []

    print("=" * 70)
    print("Stage 2: analyze_events event JSON → statistics JSON")
    print("=" * 70)

    stage2_args = argparse.Namespace(
        in_filenames=analysis_jsons,
        output_dir=args.stage2_output,
        no_plots=args.no_plots,
        reset=args.stage2_reset,
        scan=args.stage2_scan,
        update=args.stage2_update,
        restart=args.stage2_restart,
    )

    analysis_variables, correlations_to_plot = analyze_events.get_analysis_config()

    if stage2_args.reset:
        analyze_events.reset_outputs_for_inputs(stage2_args, analysis_variables, correlations_to_plot, stage="stage2_statistics")
        return []
    if stage2_args.scan:
        analyze_events.scan_outputs_for_inputs(stage2_args, analysis_variables, correlations_to_plot, stage="stage2_statistics")
        return []
    if stage2_args.update:
        analyze_events.update_outputs_for_inputs(stage2_args, analysis_variables, correlations_to_plot, stage="stage2_statistics")
        return []
    if stage2_args.restart:
        analyze_events.restart_outputs_for_inputs(stage2_args, analysis_variables, correlations_to_plot, stage="stage2_statistics")
        return []

    analyze_events.analyze_all(stage2_args, analysis_variables, correlations_to_plot, stage="stage2_statistics")

    # Compute expected statistics output paths
    stats_outputs = []
    for input_file in analysis_jsons:
        out_dir = analyze_events.determine_output_directory(input_file, args.stage2_output, stage="stage2_statistics")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        stats_outputs.append(os.path.join(out_dir, "json_stats", f"statistics_{base_name}.json"))
    return stats_outputs


def find_statistics_dirs(stat_files: List[str], stage2_output_dir: str, fallback_inputs: List[str]) -> List[str]:
    dirs: Set[str] = set()

    for stat in stat_files:
        if os.path.exists(stat):
            dirs.add(os.path.dirname(stat))

    if not dirs and os.path.isdir(stage2_output_dir):
        for root, _, files in os.walk(stage2_output_dir):
            if any(f.startswith("statistics_") and f.endswith(".json") for f in files):
                dirs.add(root)

    if not dirs:
        for p in fallback_inputs:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    if any(f.startswith("statistics_") and f.endswith(".json") for f in files):
                        dirs.add(root)

    return sorted(dirs)


def run_stage3(stats_dirs: List[str], args):
    if not stats_dirs:
        print("Warning: No statistics directories found for Stage 3. Skipping plot generation.")
        return

    print("=" * 70)
    print("Stage 3: plot_all statistics JSON → comparison plots")
    print("=" * 70)

    plot_all_path = str(Path(__file__).parent / "plot_all.py")

    cmd = [sys.executable, plot_all_path, "-i", *stats_dirs, "-p", args.stage3_pattern, "-m", args.stage3_mode]
    if args.log_scale:
        cmd.append("--log_scale")
    if args.loglog_fit_range:
        cmd.extend(["--loglog_fit_range", args.loglog_fit_range])

    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run full NIPXIe analysis pipeline (Stage 1 → 2 → 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run on a folder (recursively finds TDMS)
  python run_full_nipxie_analysis.py /path/to/data

  # Multiple folders
  python run_full_nipxie_analysis.py /data/run1 /data/run2

  # Custom output folders
  python run_full_nipxie_analysis.py /data --stage1-output ./stage1_events --stage2-output ./stage2_statistics

  # Only Stage 1 + Stage 2 (skip plots)
  python run_full_nipxie_analysis.py /data --skip-stage3 --no-plots
""",
    )

    parser.add_argument("inputs", nargs="+", help="Input TDMS files or folders (subfolders supported)")
    parser.add_argument("--recursive", action="store_true", default=True, help="Search for TDMS files recursively")

    # Stage control
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 (SelfTrigger)")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 (analyze_events)")
    parser.add_argument("--skip-stage3", action="store_true", help="Skip Stage 3 (plot_all)")

    # Stage 1 options
    parser.add_argument("--stage1-output", default="./stage1_events", help="Stage 1 output directory")
    parser.add_argument("--report", "-r", default=1000, type=int, help="Report every N events")
    parser.add_argument("--subset", "-s", default=-1, type=int, help="Process first N events")
    parser.add_argument("--find_sync_method", default="spline", choices=["spline", "simple"], help="Trigger calculation method")
    parser.add_argument("--debug_report", "-b", action="store_true", help="Enable debug output in Stage 1")
    parser.add_argument("--display_report", "-p", action="store_true", help="Display waveforms during Stage 1")

    # Stage 2 options
    parser.add_argument("--stage2-output", default="./stage2_statistics", help="Stage 2 output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation in Stage 2")
    parser.add_argument("--stage2-reset", action="store_true", help="Reset Stage 2 outputs")
    parser.add_argument("--stage2-scan", action="store_true", help="Scan Stage 2 outputs for missing files")
    parser.add_argument("--stage2-update", action="store_true", help="Update only missing Stage 2 outputs")
    parser.add_argument("--stage2-restart", action="store_true", help="Reset and recompute all Stage 2 outputs")

    # Stage 3 options
    parser.add_argument("--stage3-pattern", default="statistics_*.json", help="Stage 3 file pattern")
    parser.add_argument("--stage3-mode", default="all", choices=["all", "vs_bias", "vs_power", "pulse"], help="Stage 3 plot mode")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for power plots in Stage 3")
    parser.add_argument("--loglog_fit_range", default=None, help="Fit range for log-log power plots (min,max in nW)")

    args = parser.parse_args()

    tdms_files = find_tdms_files(args.inputs, recursive=args.recursive)
    if not tdms_files and not args.skip_stage1:
        print("Error: No TDMS files found.")
        sys.exit(1)

    # Stage 1
    analysis_jsons = []
    if not args.skip_stage1:
        analysis_jsons = run_stage1(tdms_files, args)
    else:
        # If stage1 is skipped, attempt to find existing analysis files
        for base in args.inputs:
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.endswith("_analysis.json"):
                            analysis_jsons.append(os.path.join(root, f))
            elif os.path.isfile(base) and base.endswith("_analysis.json"):
                analysis_jsons.append(base)

    # Stage 2
    stats_jsons = []
    if not args.skip_stage2:
        if not analysis_jsons:
            analysis_jsons = analysis_output_paths(tdms_files, args.stage1_output)
        stats_jsons = run_stage2(analysis_jsons, args)

    # Stage 3
    if not args.skip_stage3:
        stats_dirs = find_statistics_dirs(stats_jsons, args.stage2_output, args.inputs)
        run_stage3(stats_dirs, args)


if __name__ == "__main__":
    main()
