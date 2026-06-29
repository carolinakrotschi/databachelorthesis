import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR / "rawdata"
PLOT_DIR = SCRIPT_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)


def read_measurement(file_path: Path):
    time = []
    signal = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            try:
                t = float(parts[0])
                s = float(parts[1])
            except ValueError:
                continue

            time.append(t)
            signal.append(s)

    return np.array(time), np.array(signal)


measurements = [
    ("thorlabsCPS780S", "thorlabsCPS780S.txt"),
    ("thorlabsCPS780S_2", "thorlabsCPS780S_2.txt"),
]

results = []

plt.figure(figsize=(12, 6))

for label, filename in measurements:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    time, signal = read_measurement(file_path)

    if len(time) == 0 or len(signal) == 0:
        print(f"No measurement data found: {file_path}")
        continue

    mean_signal = float(np.mean(signal))
    std_signal = float(np.std(signal))
    snr = mean_signal / std_signal if std_signal != 0 else np.inf
    results.append((label, mean_signal, std_signal, snr))

    plot_label = f"{label} (SNR={snr:.2f})"
    plt.plot(time, signal, label=plot_label)

plt.xlabel("Time (s)")
plt.ylabel("Signal (W)")
plt.title("Thorlabs CPS780S Comparison")
plt.grid(True)
plt.legend()

if results:
    info_lines = [
        f"{label}: SNR={snr:.2f}"
        for label, _, _, snr in results
    ]
    plt.gcf().text(
        0.02,
        0.02,
        "\n".join(info_lines),
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

plt.tight_layout()
plt.savefig(PLOT_DIR / "thorlabsCPS780S_comparison.png", dpi=300)
plt.close()

print("Done.")
for label, mean_signal, std_signal, snr in results:
    print(f"{label}: mean={mean_signal:.6e} W, std={std_signal:.6e} W, SNR={snr:.2f}")
print(f"Comparison plot saved in: {PLOT_DIR.resolve()}")
