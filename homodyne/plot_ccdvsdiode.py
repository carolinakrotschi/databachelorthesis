from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data
ccd = [41, 37, 31, 28, 40, 36]
single_photodiode = [41, 51, 49, 52, 39, 37]
homodyne_forward = [20.110, 26.477, 29.202, 21.775, 26.907, 26.296]
homodyne_backward = [16.601, 10.924, 23.276, 9.982, 19.935, 13.998]


# Create plot
plt.figure(figsize=(8, 6))

# Draw points
plt.scatter([0] * len(ccd), ccd, label="CCD", s=70)
plt.scatter([1] * len(single_photodiode), single_photodiode, label="Single Photodiode", s=70)
plt.scatter([2] * len(homodyne_forward), homodyne_forward, label="Homodyne Forward", s=70)
plt.scatter([3] * len(homodyne_backward), homodyne_backward, label="Homodyne Backward", s=70)

# Axis labels
plt.xticks([0, 1, 2, 3], ["CCD", "Single Photodiode", "Homodyne Forward", "Homodyne Backward"])
plt.ylabel("Number of Fringes")

# Optional: grid
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Axis limits
plt.xlim(-0.5, 3.5)

# Title
plt.title("Comparison of Number of Fringes")

# Add formula and calculations below the plot
info_text = (
    r"$N = \frac{2d}{\lambda}$" + "\n" +
    r"$d = 0.01\,\mathrm{mm} = 10\,\mathrm{\mu m},\ \lambda = 780\,\mathrm{nm} = 0.780\,\mathrm{\mu m}$" + "\n" +
    r"$N = \frac{2 \cdot 10}{0.780} \approx 25.6 \rightarrow$ should see 26 fringes"
)

# Add text box
plt.figtext(
    0.5, 0.02, info_text, ha="center", fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray", alpha=0.9)
)

# Make room for the text box at the bottom
plt.subplots_adjust(bottom=0.25)

base_dir = Path(__file__).resolve().parent
results_dir = base_dir / 'results'
results_dir.mkdir(exist_ok=True)

output_path = results_dir / 'ccd_vs_single_diode.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to {output_path}")
