import json
import argparse
import numpy as np
from scipy import stats
import os
from datetime import datetime
import matplotlib.pyplot as plt
# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot histogram of pulse_fall_range_ptp values')
parser.add_argument('in_filenames', nargs="+", help='input filenames')
args = parser.parse_args()

def analysis(events, threshold=0.02):
    """Analyze a single file"""
    PulseRanges = [event.get('pulse_fall_range_ptp') for event in events if 'pulse_fall_range_ptp' in event]
    PulseRanges_filtered = [val for val in PulseRanges if val >= threshold]
    average_ptp = sum(PulseRanges_filtered) / len(PulseRanges_filtered) if PulseRanges_filtered else 0.0
    if not PulseRanges:
        return 0.0    
    detected_events = sum(1 for val in PulseRanges_filtered) 
    efficiency = detected_events / len(PulseRanges) if PulseRanges else 0.0    
    return PulseRanges, PulseRanges_filtered, efficiency, average_ptp

def plot_events(values, rangemin=0.0, rangemax=0.1, metadata_str="", plot_dir="plots"):
    """Plot events as a scatter plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(values)), values, color='skyblue', edgecolor='black', s=100)
    plt.xlabel('Event Index')
    plt.ylabel('pulse_fall_range_ptp')
    plt.title(f'Pulse Fall Range PTP - {metadata_str}')
    plt.xlim(0, len(values))
    plt.ylim(rangemin, rangemax)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plot_dir}/pulse_fall_range_ptp_{metadata_str.replace(" ", "_").replace(",", "").replace(":", "")}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_histogram(values, binnumber=40, rangemin=0.0, rangemax=0.1, metadata_str="", plot_dir="plots"):
    """Plot histogram for a single file"""
    plt.figure()
    bincontent, binedges, _ = plt.hist(values, bins=binnumber, range=(rangemin, rangemax), color='skyblue', edgecolor='black')
    plt.xlabel('pulse_fall_range_ptp')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.title(f'Histogram of pulse_fall_range_ptp - {metadata_str}')   
    plt.savefig(f'{plot_dir}/histogram_{metadata_str.replace(" ", "_").replace(",", "").replace(":", "")}.png', dpi=300, bbox_inches='tight')
    return bincontent, binedges
    # plt.show()

def read_file(filename,attenuator):
    events = read_events(filename)
    if not events:
        print(f"No events found in {filename}")
        return
    metadata = read_metadata(filename,attenuator)
    if not metadata:
        print(f"No metadata found for {filename}")
        return
    print(f"Analyzing {filename} with metadata: {metadata['str']}")
    return events, metadata

def read_events(filename):
    with open(filename) as f:
        data = json.load(f)    
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                events = v
                break
    else:
        events = data
    return events
    
def read_metadata(filename,attenuator):
    meta_filename = filename.replace('_analysis.json', '_meta.json')
    try:
        with open(meta_filename) as f:
            metadata = {}
            for line in f:
                entry = json.loads(line)
                if entry.get('metaKey') == 'Polarization':
                    metadata['Polarization'] = int(entry.get('metaValue'))
                    metadata['Power'] = attenuator.get(metadata['Polarization'])
                elif entry.get('metaKey') == 'Bias Voltage (mV)':
                    metadata['Bias_Voltage'] = entry.get('metaValue')
            metadata['str'] = f"Polarization: {metadata["Polarization"]}, Bias Voltage: {metadata["Bias_Voltage"]}"
            return metadata if metadata else None
    except FileNotFoundError:
        print(f'Metadata file not found: {meta_filename}')
        return None
        
def read_attenuator_file():
    """Read the attenuator file and return a dictionary of degree to attenuation values"""
    attenuator = {}
    with open('python/config/attenuator.txt') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                degree = int(parts[0])
                attenuation = float(parts[1])
                attenuator[degree] = attenuation
    return attenuator

def main():
    thd = 0.03  # Threshold for pulse_fall_range_ptp
    attenuator = read_attenuator_file()
    for filename in args.in_filenames:
        print(filename)
        plot_dir = os.path.dirname(filename)
        events, metadata = read_file(filename,attenuator)
        PulseRanges, PulseRanges_filtered, efficiency, average_ptp = analysis(events, thd)
        plot_events(PulseRanges, rangemin=0.0, rangemax=0.1, metadata_str=metadata['str'], plot_dir=plot_dir)
        bin_content, bin_edges = plot_histogram(PulseRanges, metadata_str=metadata['str'], plot_dir=plot_dir)
        print(f"Efficiency: {efficiency:.6f}, Average PTP: {average_ptp:.6f}")
        # Create data structure to dump
        output_data = {
            'filename': filename,
            'metadata': metadata,
            'analysis_results': {
            'threshold': thd,
            'efficiency': efficiency,
            'average_ptp': average_ptp,
            'total_events': len(PulseRanges),
            'filtered_events': len(PulseRanges_filtered)
            },
            'histogram_data': {
            'bin_content': bin_content.tolist(),
            'bin_edges': bin_edges.tolist()
            }
        }
        # Generate output filename based on input filename
        output_filename = filename.replace('_analysis.json', '_results.json')                
        # Write output data to file
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)                
        print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()