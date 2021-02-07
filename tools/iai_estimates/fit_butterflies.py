import sys
from matplotlib import pyplot as plt
import numpy as np
from read_cachegrind_benches import  read_cachegrind, get_cycles

showplots = False
if len(sys.argv)>1:
    showplots = bool(sys.argv[1])
calibfile = "target/iai/cachegrind.out.iai_calibration"
calibration = read_cachegrind(calibfile)
length = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32]
multi_nbr = np.arange(2,11)
plot_nbr = np.arange(0,11)
all_cycles = []
k_v = []
m_v = []
for fftlen in length:
    estimated_cycles = []
    for nbr in multi_nbr:
        fname = f"target/iai/cachegrind.out.bench_planned_multi_{fftlen}_{nbr}"
        fname_noop = f"target/iai/cachegrind.out.bench_planned_multi_setup_{fftlen}_{nbr}"
        results = read_cachegrind(fname)
        cycles = get_cycles(results, calibration)
        results_noop = read_cachegrind(fname_noop)
        cycles_noop = get_cycles(results_noop, calibration)
        fft_cycles = (cycles-cycles_noop)
        #print(f"{fftlen} {fft_cycles}")
        estimated_cycles.append(fft_cycles)
    # y = k*x + m
    k, m = np.polyfit(multi_nbr, estimated_cycles, 1)
    k_v.append(k)
    m_v.append(m)
    
    all_cycles.append(estimated_cycles)
    if showplots:
        plt.figure(fftlen)
        plt.plot(multi_nbr, estimated_cycles, '*')
        plt.plot(plot_nbr, k*plot_nbr + m)
print("--- Paste in scalar_planner_estimates.rs ---")
print("// --- Begin code generated by tools/fit_butterflies.py --- \n")
for fftlen, k, m in zip(length, k_v, m_v):
    print(f"const BUTTERFLY_SLOPE_{fftlen}: f32 = {k:.5f};")
    print(f"const BUTTERFLY_CONST_{fftlen}: f32 = {m:.5f};")
    print("")
    print(f"pub fn estimate_butterfly_cost_{fftlen}(repeats: usize) -> f32 {{")
    print(f"    BUTTERFLY_SLOPE_{fftlen} * repeats as f32 + BUTTERFLY_CONST_{fftlen}")
    print("}")
    print("")
print("// --- End code generated by tools/fit_butterflies.py --- \n")

print("\n--- Paste in fit_mixedradixes.py ---")
for fftlen, k, m in zip(length, k_v, m_v):
    print(f'    {fftlen}: {{ "slope": {k}, "const": {m}}},')
if showplots:
    plt.show()

