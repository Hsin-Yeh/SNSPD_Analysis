import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot histogram of pulse_fall_range_ptp values')
parser.add_argument('in_filenames', nargs="+", help='input filenames')
parser.add_argument('--output_dir','-d', default='.', help='Output directory for plots')
args = parser.parse_args()

def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    
    # If data is a string, parse it again
    if isinstance(data, str):
        data = json.loads(data)
    
    # If data is a dict with events, extract the events list
    if isinstance(data, dict):
        if 'event_by_event_data' in data:
            data = data['event_by_event_data']
        elif 'events' in data:
            data = data['events']
        elif 'data' in data:
            data = data['data']
    
    return data

def plot_variable_vs_event(data, variable_name, filename):
    """Plot a variable vs event number"""
    event_numbers = [event['event_number'] for event in data]
    values = [event[variable_name] for event in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(event_numbers, values, marker='o', linestyle='', markersize=3)
    plt.xlabel('Event Number')
    plt.ylabel(variable_name)
    
    # Zoom in for trigger_check around 180-210
    if variable_name == 'trigger_check':
        plt.ylim(180, 210)
    
    plt.title(f'{variable_name} vs Event Number\n{filename}')
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def plot_variable_histogram(data, variable_name, filename, bins=50):
    """Plot histogram of a variable"""
    values = [event[variable_name] for event in data]
    
    # Use smaller bin width for trigger_check
    if variable_name == 'trigger_check':
        bins = 300
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(variable_name)
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.title(f'{variable_name} Histogram\n{filename}')
    plt.grid(True, alpha=0.3)
    
    # Zoom in for trigger_check around 200 and calculate mean in that range
    if variable_name == 'trigger_check':
        plt.xlim(185, 205)
        # Calculate mean for values around 200 (between 185 and 205)
        values_around_200 = [v for v in values if 180 <= v <= 210]
        if values_around_200:
            mean_around_200 = np.mean(values_around_200)
            std_around_200 = np.std(values_around_200)
            plt.axvline(mean_around_200, color='purple', linestyle='--', linewidth=2, 
                       label=f'Mean (185-205): {mean_around_200:.4f}')
            print(f"  Mean of trigger_check (185-205): {mean_around_200:.4f} ± {std_around_200:.4f}")
            print(f"  Count in range (185-205): {len(values_around_200)} / {len(values)}")
    
    # Add statistics text
    mean_val = np.mean(values)
    std_val = np.std(values)
    plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean (all): {mean_val:.4f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'±1σ: {std_val:.4f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_2d_correlation(data, var_x, var_y, filename, bins=50):
    """Plot 2D correlation between two variables"""
    values_x = [event[var_x] for event in data]
    values_y = [event[var_y] for event in data]
    
    plt.figure(figsize=(10, 8))
    plt.hist2d(values_x, values_y, bins=bins, cmap='viridis', cmin=1)
    plt.colorbar(label='Counts')
    
    # Mark mean value of pulse_fall_range_ptp if it's one of the variables
    if var_y == 'pulse_fall_range_ptp':
        mean_y = np.mean(values_y)
        plt.axhline(mean_y, color='red', linestyle='--', linewidth=2, label=f'Mean {var_y}: {mean_y:.4f}')
        plt.legend()
    elif var_x == 'pulse_fall_range_ptp':
        mean_x = np.mean(values_x)
        plt.axvline(mean_x, color='red', linestyle='--', linewidth=2, label=f'Mean {var_x}: {mean_x:.4f}')
        plt.legend()
    
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    plt.title(f'2D Correlation: {var_x} vs {var_y}\n{filename}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def analyze_cuts(data, filename):
    """Analyze signal and dark counts with cuts"""
    
    # Apply cuts: trigger_check between 22-24
    time_cut_min = 22
    time_cut_max = 24
    
    # Total events
    total_events = len(data)
    
    # Events passing trigger_check cut (signal events)
    signal_events = [event for event in data if time_cut_min <= event.get('trigger_check', -999) <= time_cut_max]
    signal_count = len(signal_events)
    
    # Events failing trigger_check cut (dark counts)
    dark_events = [event for event in data if not (time_cut_min <= event.get('trigger_check', -999) <= time_cut_max)]
    dark_count = len(dark_events)
    
    # Calculate efficiency (assuming signal events are the ones passing the cut)
    efficiency = (signal_count / total_events * 100) if total_events > 0 else 0
    
    # Calculate rates (events per measurement - would need time info for real rate)
    signal_rate = signal_count
    dark_rate = dark_count
    
    print(f"\n{'='*60}")
    print(f"Cut Analysis for: {filename}")
    print(f"{'='*60}")
    print(f"Trigger Check Cut: {time_cut_min} <= trigger_check <= {time_cut_max}")
    print(f"Total Events: {total_events}")
    print(f"Signal Events (passed cut): {signal_count}")
    print(f"Dark Events (failed cut): {dark_count}")
    print(f"Efficiency: {efficiency:.2f}%")
    print(f"Signal Count Rate: {signal_rate} events")
    print(f"Dark Count Rate: {dark_rate} events")
    print(f"Signal/Dark Ratio: {signal_count/dark_count:.2f}" if dark_count > 0 else "Signal/Dark Ratio: inf")
    print(f"{'='*60}\n")
    
    return {
        'total_events': total_events,
        'signal_count': signal_count,
        'dark_count': dark_count,
        'efficiency': efficiency,
        'signal_rate': signal_rate,
        'dark_rate': dark_rate
    }

def main():
    variables_to_plot = [
        'pre_mean', 'pulse_max', 'pulse_min', 
        'pulse_time', 'pulse_time_interval',
        'pulse_rise_range_ptb', 'pulse_fall_range_ptp',
        'trigger_check'
    ]
    
    for filename in args.in_filenames:
        print(f"Processing {filename}")
        data = read_file(filename)
        
        for var in variables_to_plot:
            try:
                # Plot variable vs event
                fig = plot_variable_vs_event(data, var, filename)
                output_path = f"{args.output_dir}/{var}_vs_event_{filename.split('/')[-1].replace('.json', '.png')}"
                fig.savefig(output_path)
                plt.close(fig)
                print(f"  Saved {output_path}")
                
                # Plot histogram
                fig = plot_variable_histogram(data, var, filename)
                output_path = f"{args.output_dir}/{var}_histogram_{filename.split('/')[-1].replace('.json', '.png')}"
                fig.savefig(output_path)
                plt.close(fig)
                print(f"  Saved {output_path}")
            except KeyError:
                print(f"  Warning: Variable '{var}' not found in data")
        
        # Plot 2D correlation between trigger_check and pulse_time_interval
        try:
            fig = plot_2d_correlation(data, 'trigger_check', 'pulse_time_interval', filename)
            output_path = f"{args.output_dir}/correlation_trigger_check_vs_pulse_time_interval_{filename.split('/')[-1].replace('.json', '.png')}"
            fig.savefig(output_path)
            plt.close(fig)
            print(f"  Saved {output_path}")
        except KeyError as e:
            print(f"  Warning: Could not create correlation plot - {e}")
        
        # Plot 2D correlation between trigger_check and pulse_fall_range_ptp
        try:
            fig = plot_2d_correlation(data, 'trigger_check', 'pulse_fall_range_ptp', filename)
            output_path = f"{args.output_dir}/correlation_trigger_check_vs_pulse_fall_range_ptp_{filename.split('/')[-1].replace('.json', '.png')}"
            fig.savefig(output_path)
            plt.close(fig)
            print(f"  Saved {output_path}")
        except KeyError as e:
            print(f"  Warning: Could not create correlation plot - {e}")
        
        # Plot 2D correlation between pulse_time_interval and pulse_fall_range_ptp
        try:
            fig = plot_2d_correlation(data, 'pulse_time_interval', 'pulse_fall_range_ptp', filename)
            output_path = f"{args.output_dir}/correlation_pulse_time_interval_vs_pulse_fall_range_ptp_{filename.split('/')[-1].replace('.json', '.png')}"
            fig.savefig(output_path)
            plt.close(fig)
            print(f"  Saved {output_path}")
        except KeyError as e:
            print(f"  Warning: Could not create correlation plot - {e}")
        
        # Analyze cuts and calculate efficiency
        cut_results = analyze_cuts(data, filename)

if __name__ == "__main__":
    main()