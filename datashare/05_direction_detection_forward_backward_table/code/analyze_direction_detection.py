#!/usr/bin/env python3
import csv
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "rawdata"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FILES = [
    ("Measurement direction backward #1", "homodyne_backward_01.csv"),
    ("Measurement direction backward #2", "homodyne_backward_02.csv"),
    ("Measurement direction backward #3", "homodyne_backward_03.csv"),
    ("Measurement direction forward #1", "homodyne_forward_01.csv"),
    ("Measurement direction forward #2", "homodyne_forward_02.csv"),
    ("Measurement direction forward #3", "homodyne_forward_03.csv"),
]


# Determine motion direction from the quadrature (S1/S2) phase evolution:
# a monotonically increasing/decreasing unwrapped phase means forward/backward motion.
def analyze_direction(csv_path: Path):
    s1, s2 = [], []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s1.append(float(row["Raw_S1_V"]))
            s2.append(float(row["Raw_S2_V"]))

    # Remove DC offset so the signals trace out a circle centered on the origin
    s1 = np.array(s1) - np.mean(s1)
    s2 = np.array(s2) - np.mean(s2)

    phase = np.unwrap(np.arctan2(s2, s1))
    dphase = np.diff(phase)

    # Classify each phase step as forward/backward motion or noise ("still")
    threshold = 0.05
    forward_count = np.sum(dphase > threshold)
    backward_count = np.sum(dphase < -threshold)
    still_count = np.sum(np.abs(dphase) <= threshold)
    total = len(dphase)

    return (
        forward_count / total * 100,
        backward_count / total * 100,
        still_count / total * 100,
    )


def main():
    table_rows = ["| File                  | forward   | backward  | still |", ""]
    for label, filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        fwd, bwd, stl = analyze_direction(path)
        table_rows.append(f"| {label:<37} | {fwd:6.2f}% | {bwd:6.2f}% | {stl:6.2f}% |")

    output_path = RESULTS_DIR / "direction_detection_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_rows) + "\n")

    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
