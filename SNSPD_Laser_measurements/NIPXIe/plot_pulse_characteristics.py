import numpy as np
import matplotlib.pyplot as plt

def plot_pulse_ptp_vs_bias(power_groups, output_dir):
    """Plot pulse peak-to-peak vs bias voltage for each power group."""
    for power, pdata in power_groups.items():
        bias_voltages = [d['bias_voltage'] for d in pdata]
        ptp_values = [d.get('pulse_ptp', np.nan) for d in pdata]
        plt.figure(figsize=(8, 5))
        plt.plot(bias_voltages, ptp_values, marker='o', linestyle='-', label=f'{power} nW')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('Pulse Peak-to-Peak (V)')
        plt.title(f'Pulse PTP vs Bias Voltage ({power} nW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = f'{output_dir}/pulse_ptp_vs_bias_{power}nW.png'
        plt.savefig(fname, dpi=150)
        plt.close()

def plot_pulse_ptp_vs_power(bias_groups, output_dir):
    """Plot pulse peak-to-peak vs power for each bias group."""
    for bias, bdata in bias_groups.items():
        powers = [d['power'] for d in bdata if d['power'] is not None]
        ptp_values = [d.get('pulse_ptp', np.nan) for d in bdata if d['power'] is not None]
        plt.figure(figsize=(8, 5))
        plt.plot(powers, ptp_values, marker='o', linestyle='-', label=f'{bias} mV')
        plt.xlabel('Power (nW)')
        plt.ylabel('Pulse Peak-to-Peak (V)')
        plt.title(f'Pulse PTP vs Power ({bias} mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = f'{output_dir}/pulse_ptp_vs_power_{bias}mV.png'
        plt.savefig(fname, dpi=150)
        plt.close()
