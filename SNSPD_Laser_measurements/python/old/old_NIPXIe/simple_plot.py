import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot histogram of pulse_fall_range_ptp values')
parser.add_argument('in_filenames', nargs="+", help='input filenames')
parser.add_argument('--output_dir','-d', default='.', help='Output directory for plots')
args = parser.parse_args()

def plot_graph(xvalues, yvalues):
    plt.figure(figsize=(10, 6))
    plt.scatter(xvalues, yvalues, color='lightcoral', edgecolor='black', s=100)
    plt.xlabel('Intensity')
    plt.ylabel('Efficiency')
    #plt.ylim(0, 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add efficiency values next to points
    for intensity, eff in zip(xvalues, yvalues):
        plt.annotate(f'{eff:.6f}', (intensity, eff), xytext=(5, 5), 
                    textcoords='offset points', ha='left', va='bottom')
    
    plt.show()

def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    
    # Extract power and efficiency values
    power = float(data['metadata']['Power'])
    bias = int(data['metadata']['Bias_Voltage'])
    efficiency = float(data['analysis_results']['efficiency'])
    filtered_events = int(data['analysis_results']['filtered_events'])
    event = int(data['analysis_results']['total_events'])
    return power, bias, efficiency, filtered_events, event

def plot_by_bias(powers, efficiencies, biases):
    unique_biases = sorted(set(biases))
    plt.figure(figsize=(10, 6))
    
    for bias in unique_biases:
        bias_powers = [p for p, b in zip(powers, biases) if b == bias]
        bias_effs = [e for e, b in zip(efficiencies, biases) if b == bias]
        
        # Filter out zeros for log-log plot
        filtered_powers = []
        filtered_effs = []
        for p, e in zip(bias_powers, bias_effs):
            if p > 0 and e > 0:
                filtered_powers.append(p)
                filtered_effs.append(e)
        
        if filtered_powers:  # Only plot if we have valid data
            plt.scatter(filtered_powers, filtered_effs, label=f'Bias: {bias}V', s=100)
            
            # Fit log-log line for this bias
            if len(filtered_powers) > 1:
                log_powers = np.log10(filtered_powers)
                log_effs = np.log10(filtered_effs)
                coeffs = np.polyfit(log_powers, log_effs, 1)
                fit_powers = np.logspace(np.log10(min(filtered_powers)), np.log10(max(filtered_powers)), 100)
                fit_effs = 10**(coeffs[0] * np.log10(fit_powers) + coeffs[1])
                plt.plot(fit_powers, fit_effs, '--', alpha=0.7, label=f'Bias {bias}V fit (slope={coeffs[0]:.2f})')
                # Extrapolate to 89.6V and 85.7V
                extrapolate_powers = [89.6, 85.7, 0.1]
                for extrap_power in extrapolate_powers:
                    extrap_eff = 10**(coeffs[0] * np.log10(extrap_power) + coeffs[1])
                    print(f"Extrapolated efficiency at {extrap_power} power for bias {bias}V: {extrap_eff}")

        plt.xlabel('Power')
    plt.ylabel('Efficiency')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    for power, eff in zip(powers, efficiencies):
        plt.annotate(f'{eff:.6f}', (power, eff), xytext=(5, 5), 
                     textcoords='offset points', ha='left', va='bottom')    
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{args.output_dir}/effvspow.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_by_power(powers, efficiencies, biases):
    unique_powers = sorted(set(powers))
    plt.figure(figsize=(10, 6))
    
    for power in unique_powers:
        power_biases = [b for p, b in zip(powers, biases) if p == power]
        power_effs = [e for p, e in zip(powers, efficiencies) if p == power]
        plt.scatter(power_biases, power_effs, label=f'Power: {power}', s=100)
    
    plt.xlabel('Bias Voltage')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    powers = []
    efficiencies = []
    counts = []
    biases = []
    events = []
    for filename in args.in_filenames:
        print(filename)
        # Read the file and extract data
        power, bias, efficiency, count, event = read_file(filename)
        powers.append(power)
        efficiencies.append(efficiency)
        counts.append(count)
        biases.append(bias)
    # Plot the results  
    plot_by_power(powers, efficiencies, biases)
    plot_by_bias(powers, efficiencies, biases)


if __name__ == "__main__":
    main()