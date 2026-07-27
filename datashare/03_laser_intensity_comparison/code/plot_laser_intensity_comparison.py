import numpy as np
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = SCRIPT_DIR.parent / "rawdata"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THESIS_LABEL_SIZE = 18
THESIS_TICK_SIZE = 18
THESIS_LEGEND_SIZE = 18

# (display name, raw data file) for each laser's power-vs-time trace
measurements = [
    ("uniphase1507p", "uniphase_1507p.txt"),
    ("thorlabsCPS780S", "thorlabs_cps780s.txt"),
    ("uniphase023p", "uniphase_023p.txt"),
    ("uniphase1103p1108380", "uniphase_1103p_1108380.txt"),
    ("uniphase1103p1177761", "uniphase_1103p_1177761.txt"),
    ("uniphase1122p", "uniphase_1122p.txt")
]

plt.figure(figsize=(12, 6))

# Load and plot each laser's power-vs-time trace on one shared axis
for name, filename in measurements:
    file_path = DATA_DIR / filename
    if not file_path.exists():
        continue

    time = []
    signal = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    time.append(float(parts[0]))
                    signal.append(float(parts[1]))
                except ValueError:
                    pass

    time = np.array(time)
    signal = np.array(signal) * 1e3  # power meter reports W, plot in mW

    if len(time) > 0 and len(signal) > 0:
        plt.plot(time, signal, label=name)

plt.xlabel("Time [s]", fontsize=THESIS_LABEL_SIZE)
plt.ylabel("Signal [mW]", fontsize=THESIS_LABEL_SIZE)
plt.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
plt.grid(True)
plt.legend(
    fontsize=THESIS_LEGEND_SIZE,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "laser_intensity_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Results saved in: {RESULTS_DIR.resolve()}")
