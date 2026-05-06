import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("rawdata")
PLOT_DIR = Path("Plots")
PLOT_DIR.mkdir(exist_ok=True)

files = [
    "uniphase1502p.txt",
    "thorlabslds5.txt",
    "uniphase023p.txt",
    "uniphase1103p1108380.txt",
    "uniphase1103p1177761.txt",
    "uniphase1122p.txt"
]

results = []

plt.figure(figsize=(12, 6))

for filename in files:
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

    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    snr = mean_signal / std_signal
    snr_db = 20 * np.log10(snr)

    results.append([filename, mean_signal, std_signal, snr, snr_db])

    name = Path(filename).stem

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
    f.write("File\tMean_Signal_W\tStd_Dev_W\tSNR\tSNR_dB\n")

    for r in results:
        f.write(
            f"{r[0]}\t"
            f"{r[1]:.6e}\t"
            f"{r[2]:.6e}\t"
            f"{r[3]:.6f}\t"
            f"{r[4]:.6f}\n"
        )

print("Done.")
print(f"Plots and SNR results saved in: {PLOT_DIR.resolve()}")