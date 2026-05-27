import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO


def read_opm_file(filepath: str, sep: str = ",") -> tuple[pd.DataFrame, dict[str, str]]:
    """Liest eine OPM-Exportdatei ein und gibt Graph-Daten und Statistik zurück."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    stats = {}
    graph_start = None
    stats_start = None

    for idx, line in enumerate(lines):
        if line.strip().startswith("--Statistics--"):
            stats_start = idx + 1
        elif line.strip().startswith("--Graph Data--"):
            graph_start = idx + 1
            break

    if graph_start is None:
        raise ValueError(f"Datei '{filepath}' enthält keinen Graph Data Abschnitt.")

    if stats_start is not None and stats_start < graph_start:
        for line in lines[stats_start:graph_start - 1]:
            if not line.strip():
                continue
            if "," not in line:
                continue
            key, value = line.split(",", 1)
            stats[key.strip()] = value.strip()

    data_text = "\n".join(lines[graph_start:]).strip()
    if not data_text:
        raise ValueError(f"Datei '{filepath}' enthält keine Graph-Daten.")

    df = pd.read_csv(StringIO(data_text), sep=sep)
    return df, stats


def format_stats(title: str, stats: dict[str, str]) -> list[str]:
    lines = [title]
    lines.extend(f"{key}: {value}" for key, value in stats.items())
    return lines


def format_stats_side_by_side(title_left: str, stats_left: dict[str, str], title_right: str, stats_right: dict[str, str]) -> str:
    left_lines = format_stats(title_left, stats_left)
    right_lines = format_stats(title_right, stats_right)
    max_left = max(len(line) for line in left_lines)
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_lines - len(left_lines))
    right_lines += [""] * (max_lines - len(right_lines))
    lines = []
    for left, right in zip(left_lines, right_lines):
        lines.append(f"{left.ljust(max_left + 4)}{right}")
    return "\n".join(lines)


file_without = "withoutmovestag2e.csv"
file_with = "withmovestage2.csv"

# OPM-Dateien einlesen
# Die Dateien müssen im gleichen Ordner wie plot.py liegen.
# Wenn der Separator in deiner Datei nicht "," ist, kannst du sep=";" verwenden.
df_without, stats_without = read_opm_file(file_without, sep=",")
df_with, stats_with = read_opm_file(file_with, sep=",")

print("WITHOUT MOVE STAGE:")
print(df_without.columns)
print(stats_without)

print("\nWITH MOVE STAGE:")
print(df_with.columns)
print(stats_with)

# Spalten automatisch auswählen
try:
    time_col = next(c for c in df_without.columns if "time" in c.lower())
    signal_col = next(c for c in df_without.columns if any(x in c.lower() for x in ["power", "intensity", "signal", "voltage"]))
except StopIteration:
    time_col = df_without.columns[0]
    signal_col = df_without.columns[1]

print(f"Verwende time_col={time_col}, signal_col={signal_col}")

# Daten extrahieren
# Falls die Zeit in ms vorliegt, kann man sie bei Bedarf umrechnen.
t_without = df_without[time_col]
y_without = df_without[signal_col]

t_with = df_with[time_col]
y_with = df_with[signal_col]

# Normalisieren
# Damit beide besser vergleichbar sind

y_without_norm = (y_without - np.mean(y_without)) / np.std(y_without)
y_with_norm = (y_with - np.mean(y_with)) / np.std(y_with)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t_without, y_without_norm, label="Without moving stage", alpha=0.8)
ax.plot(t_with, y_with_norm, label="With moving stage", alpha=0.8)

ax.set_title("Interferometer Signal Comparison")
ax.set_xlabel(time_col)
ax.set_ylabel(signal_col + " (normiert)")
ax.legend()
ax.grid(True)

stats_text = format_stats_side_by_side(
    "WITHOUT MOVE STAGE",
    stats_without,
    "WITH MOVE STAGE",
    stats_with,
)

fig.text(
    0.01,
    0.01,
    stats_text,
    fontsize=9,
    ha="left",
    va="bottom",
    family="monospace",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.3")
)

plt.tight_layout(rect=[0, 0.12, 1, 1])

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "comparison_plot.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot gespeichert als: {output_path}")

plt.show()
