import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = [
    "uniphase1502p.txt",
    "thorlabslds5.txt",
    "uniphase023p.txt",
    "uniphase1103p1108380.txt",
    "uniphase1103p1177761.txt",
    "uniphase1122p.txt"
]

plot_dir = Path("Plots")
plot_dir.mkdir(exist_ok=True)

results = []

plt.figure(figsize=(12, 6))

for filename in files:
    time = []
    signal = []

    with open(filename, "r") as f:
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

    # Individual plot
    plt_individual = plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (W)")
    plt.title(filename)
    plt.grid(True)
    plt.tight_layout()

    save_name = Path(filename).stem + ".png"
    plt.savefig(plot_dir / save_name, dpi=300)
    plt.close(plt_individual)

    # Add to comparison plot
    plt.figure(1)
    plt.plot(time, signal, label=Path(filename).stem)

# Comparison plot
plt.figure(1)
plt.xlabel("Time (s)")
plt.ylabel("Signal (W)")
plt.title("Comparison of All Measurements")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "comparison_all_measurements.png", dpi=300)
plt.close()

# Write SNR results to text file
with open(plot_dir / "snr_results.txt", "w") as f:
    f.write("SNR RESULTS\n")
    f.write("=" * 50 + "\n\n")

    for r in results:
        f.write(f"File: {r[0]}\n")
        f.write(f"Mean Signal : {r[1]:.6e} W\n")
        f.write(f"Std Dev     : {r[2]:.6e} W\n")
        f.write(f"SNR         : {r[3]:.2f}\n")
        f.write("-" * 50 + "\n")

print("Done.")
print(f"Plots and SNR results saved in: {plot_dir.resolve()}")