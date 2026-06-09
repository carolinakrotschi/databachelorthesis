import numpy as np
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR / "rawdata"
PLOT_DIR = SCRIPT_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

measurements = [
    ("uniphase1507p", "uniphase1507p_2.txt"),
    ("thorlabsCPS780S", "thorlabsCPS780S.txt"),
    ("uniphase023p", "uniphase023p.txt"),
    ("uniphase1103p1180380", "uniphase1103p1180380_2.txt"),
    ("uniphase1103p1177761", "uniphase1103p11177761_2.txt"),
    ("uniphase1122p", "uniphase1122p_2.txt")
]

results = []

plt.figure(figsize=(12, 6))

for name, filename in measurements:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    time = []
    signal = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    s = float(parts[1])
                    time.append(t)
                    signal.append(s)
                except ValueError:
                    pass

    time = np.array(time)
    signal = np.array(signal)

    if len(time) == 0 or len(signal) == 0:
        print(f"No measurement data found: {file_path}")
        continue

    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    snr = mean_signal / std_signal if std_signal != 0 else np.inf
    snr_db = 20 * np.log10(snr) if snr > 0 else np.nan

    results.append([name, filename, mean_signal, std_signal, snr, snr_db])

    # Individual plot
    plt_individual = plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (W)")
    plt.title(name)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / f"{name}.png", dpi=300)
    plt.close(plt_individual)

    # Add to comparison plot
    plt.figure(1)
    plt.plot(time, signal, label=name)

# Comparison plot
plt.figure(1)
plt.xlabel("Time (s)")
plt.ylabel("Signal (W)")
plt.title("Comparison of All Measurements")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "comparison_all_measurements.png", dpi=300)
plt.close()

# Write SNR results to text file
with open(PLOT_DIR / "snr_results.txt", "w") as f:
    f.write("Measurement\tFile\tMean_Signal_W\tStd_Dev_W\tSNR\tSNR_dB\n")

    for r in results:
        f.write(
            f"{r[0]}\t"
            f"{r[1]}\t"
            f"{r[2]:.6e}\t"
            f"{r[3]:.6e}\t"
            f"{r[4]:.6f}\t"
            f"{r[5]:.6f}\n"
        )

print("Done.")
print(f"Plots and SNR results saved in: {PLOT_DIR.resolve()}")
