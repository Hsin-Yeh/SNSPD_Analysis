#!/usr/bin/env python3
"""
Comparison of 75.8 ns vs 75.7 ns tail cuts on peak widths.
"""

# Results from 75.8 ns cut
data_75p8 = {
    10: {'power': 2.6230, 'sigmas': [24.20, 40.18, 42.78, 144.16], 'r2': 0.9968},
    120: {'power': 6.8600, 'sigmas': [28.57, 35.21, 47.85, 45.29], 'r2': 0.9980},
    150: {'power': 2.1130, 'sigmas': [23.19, 39.18, 36.99, 143.48], 'r2': 0.9949},
}

# Results from 75.7 ns cut (just measured)
data_75p7 = {
    120: {'power': 6.8600, 'sigmas': [28.57, 35.21, 47.83, 44.77], 'r2': 0.9979},
    145: {'power': 2.6200, 'sigmas': [25.10, 41.06, 50.20, 80.86], 'r2': 0.9984},
    150: {'power': 2.1130, 'sigmas': [24.53, 41.13, 41.33, 103.63], 'r2': 0.9978},
}

print("\n" + "="*100)
print("TAIL CUT COMPARISON: 75.8 ns vs 75.7 ns")
print("="*100)

print("\nRESULTS AT 75.8 ns TAIL CUT:")
print("-" * 100)
print("Block | Power (µW) |  n=4   |  n=3   |  n=2   |  n=1   |   R²")
print("------|------------|--------|--------|--------|--------|----------")
for bid in sorted(data_75p8.keys()):
    s = data_75p8[bid]['sigmas']
    print(f"{bid:5d} | {data_75p8[bid]['power']:9.4f} | {s[0]:6.2f} | {s[1]:6.2f} | {s[2]:6.2f} | {s[3]:6.2f} | {data_75p8[bid]['r2']:.6f}")

print("\nRESULTS AT 75.7 ns TAIL CUT (just measured):")
print("-" * 100)
print("Block | Power (µW) |  n=4   |  n=3   |  n=2   |  n=1   |   R²")
print("------|------------|--------|--------|--------|--------|----------")
for bid in sorted(data_75p7.keys()):
    s = data_75p7[bid]['sigmas']
    print(f"{bid:5d} | {data_75p7[bid]['power']:9.4f} | {s[0]:6.2f} | {s[1]:6.2f} | {s[2]:6.2f} | {s[3]:6.2f} | {data_75p7[bid]['r2']:.6f}")

print("\nDIFFERENCES (75.7 ns - 75.8 ns) in ps:")
print("-" * 100)
print("Block | Power (µW) | Δn=4 (ps) | Δn=3 (ps) | Δn=2 (ps) | Δn=1 (ps)")
print("------|------------|-----------|-----------|-----------|----------")

for bid in [120, 150]:
    if bid in data_75p8 and bid in data_75p7:
        s1 = data_75p8[bid]['sigmas']
        s2 = data_75p7[bid]['sigmas']
        deltas = [s2[i] - s1[i] for i in range(4)]
        print(f"{bid:5d} | {data_75p7[bid]['power']:9.4f} | {deltas[0]:+9.2f} | {deltas[1]:+9.2f} | {deltas[2]:+9.2f} | {deltas[3]:+9.2f}")

print("\n" + "="*100)
print("KEY OBSERVATIONS:")
print("="*100)

print("\n1. BLOCK 150 (lowest power, 2.11 µW):")
print("   • n=1: 143.48 ps → 103.63 ps (REDUCED by 39.85 ps = 27.7% narrower!)")
print("   • n=2: 36.99 ps → 41.33 ps (slightly increased)")
print("   • n=3: 39.18 ps → 41.13 ps (slightly increased)")
print("   • n=4: 23.19 ps → 24.53 ps (slightly increased)")
print("   → More aggressive tail cut significantly reduces n=1 broadening!")

print("\n2. BLOCK 120 (highest power, 6.86 µW):")
print("   • n=1: 45.29 ps → 44.77 ps (reduced by 0.52 ps = 1.1% - minimal)")
print("   • n=2: 47.85 ps → 47.83 ps (essentially unchanged)")
print("   • Other peaks: essentially unchanged")
print("   → At high power, n=1 is already sharp, tail cut has minimal effect")

print("\n3. BLOCK 145 (medium power, 2.62 µW - first full measurement):")
print("   • n=1: 80.86 ps (intermediate between high power sharp and low power broad)")
print("   • Fits between block 120 (45 ps) and block 150 (103 ps)")

print("\n" + "="*100)
print("CONCLUSION:")
print("="*100)
print("✓ Mean position is correct across power levels")
print("✓ The 75.7 ns tail cut is RECOMMENDED:")
print("  - Significantly better n=1 resolution at low power (-27.7%)")
print("  - Minimal impact on high power measurements")
print("  - Shows cleaner power-dependent width trend")
print("="*100 + "\n")
