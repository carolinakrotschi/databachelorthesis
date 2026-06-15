from __future__ import annotations

import csv
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RAWDATA_DIR = SCRIPT_DIR / "rawdata"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FILE_PAIRS: list[tuple[str, str]] = [
    (
        "thorlabsCPS780S_2_NQ51B22081__0__11-36-36-140.txt",
        "backthorlabsCPS780S_2_NQ51B22081__0__11-35-54-129-1.txt",
    ),
    (
        "uniphase1023p_USB4C031521__0__12-07-47-856.txt",
        "backuniphase1023p_USB4C031521__3__12-07-31-183.txt",
    ),
    (
        "uniphase1103p11777_USB4C031521__0__12-13-38-733.txt",
        "backuniphase1103p11777_USB4C031521__0__12-12-07-080.txt",
    ),
    (
        "uniphase1103p110838_USB4C031521__0__16-10-06-126.txt",
        "backuniphase1103p110838_USB4C031521__0__16-09-42-795.txt",
    ),
    (
        "uniphase1122p_USB4C031521__0__16-13-14-685.txt",
        "backuniphase1122p_USB4C031521__0__16-12-54-022.txt",
    ),
    (
        "uniphase1507p0_USB4C031521__0__16-18-35-607.txt",
        "backuniphase1507p0_USB4C031521__0__16-18-20-348.txt",
    ),
]


def gaussian(x: float, A: float, mu: float, sigma: float, offset: float) -> float:
    return A * math.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def load_spectrum(file_path: Path) -> tuple[list[float], list[float]]:
    wavelengths: list[float] = []
    intensities: list[float] = []
    reading_data = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if "Begin Spectral Data" in line:
                reading_data = True
                continue

            if not reading_data:
                continue

            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue

            try:
                wavelengths.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                continue

    return wavelengths, intensities


def fit_fwhm_from_half_max(wavelengths: list[float], intensities: list[float]) -> tuple[float, float]:
    if not wavelengths or not intensities or len(wavelengths) != len(intensities):
        raise ValueError("Empty spectrum or mismatched input lengths.")

    peak_idx = max(range(len(intensities)), key=intensities.__getitem__)
    peak_wl = wavelengths[peak_idx]
    peak_intensity = intensities[peak_idx]
    baseline = min(intensities)
    half_max = baseline + 0.5 * (peak_intensity - baseline)

    left = peak_idx
    while left > 0 and intensities[left] > half_max:
        left -= 1

    right = peak_idx
    last_index = len(intensities) - 1
    while right < last_index and intensities[right] > half_max:
        right += 1

    def interpolate(i0: int, i1: int) -> float:
        x0, y0 = wavelengths[i0], intensities[i0]
        x1, y1 = wavelengths[i1], intensities[i1]
        if y1 == y0:
            return (x0 + x1) / 2.0
        return x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    if left == 0:
        left_cross = wavelengths[0]
    else:
        left_cross = interpolate(left, left + 1)

    if right == last_index:
        right_cross = wavelengths[last_index]
    else:
        right_cross = interpolate(right - 1, right)

    fwhm = max(right_cross - left_cross, 0.0)
    return peak_wl, fwhm


def estimate_coherence_length_nm(wavelength_nm: float, fwhm_nm: float) -> float:
    if fwhm_nm <= 0:
        return float("inf")
    return wavelength_nm**2 / fwhm_nm


def normalize_label(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.startswith("back"):
        stem = stem[4:]
    return stem


def find_signal_background_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for signal_name, background_name in FILE_PAIRS:
        signal_path = RAWDATA_DIR / signal_name
        background_path = RAWDATA_DIR / background_name

        if not signal_path.exists():
            print(f"Signal fehlt: {signal_path.name}")
            continue

        if not background_path.exists():
            print(f"Hintergrund fehlt: {background_path.name}")
            continue

        pairs.append((signal_path, background_path))
    return pairs


def maybe_make_plot(results: list[dict[str, float | str]]) -> None:
    try:
        mpl_cache_dir = RESULTS_DIR / ".matplotlib-cache"
        mpl_cache_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib nicht verfügbar, Plot wird übersprungen.")
        return

    if not results:
        return

    labels = [str(row["Laser"]) for row in results]
    wavelengths = [float(row["Wavelength_nm"]) for row in results]
    coherence_lengths = [float(row["CoherenceLength_mm"]) for row in results]
    fwhm_values = [float(row["FWHM_nm"]) for row in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(labels))

    ax1.bar(x, wavelengths, color="#2c7fb8", alpha=0.85, label="Peak wavelength")
    ax1.set_ylabel("Wavelength / nm")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, coherence_lengths, color="#d95f0e", marker="o", label="Coherence length")
    ax2.plot(x, fwhm_values, color="#31a354", marker="s", label="FWHM")
    ax2.set_ylabel("mm / nm")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "coherence_summary.png", dpi=300)
    plt.close(fig)


def main() -> int:
    if not RAWDATA_DIR.exists():
        print(f"Rawdata-Ordner nicht gefunden: {RAWDATA_DIR}")
        return 1

    pairs = find_signal_background_pairs()
    if not pairs:
        print("Keine Signal/Hintergrund-Paare gefunden.")
        return 1

    results: list[dict[str, float | str]] = []

    for signal_file, background_file in pairs:
        try:
            wavelengths, signal = load_spectrum(signal_file)
            wavelengths_b, background = load_spectrum(background_file)
        except FileNotFoundError as exc:
            print(f"{signal_file.name}: {exc}")
            continue

        if len(wavelengths) != len(background) or len(signal) != len(background):
            print(f"Längen passen nicht: {signal_file.name}")
            continue

        if wavelengths != wavelengths_b:
            print(f"Wellenlängen-Achsen passen nicht: {signal_file.name}")
            continue

        corrected = [max(s - b, 0.0) for s, b in zip(signal, background)]
        try:
            peak_wl, fwhm = fit_fwhm_from_half_max(wavelengths, corrected)
        except ValueError as exc:
            print(f"{signal_file.name}: {exc}")
            continue

        coherence_length_nm = estimate_coherence_length_nm(peak_wl, fwhm)
        coherence_length_mm = coherence_length_nm * 1e-6

        results.append(
            {
                "Laser": normalize_label(signal_file.name),
                "Wavelength_nm": peak_wl,
                "FWHM_nm": fwhm,
                "CoherenceLength_mm": coherence_length_mm,
            }
        )

        print(
            f"{signal_file.name}: "
            f"λ={peak_wl:.2f} nm, "
            f"FWHM={fwhm:.4f} nm, "
            f"Lc={coherence_length_mm:.4f} mm"
        )

    csv_path = RESULTS_DIR / "coherence_lengths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Laser", "Wavelength_nm", "FWHM_nm", "CoherenceLength_mm"],
        )
        writer.writeheader()
        writer.writerows(results)

    maybe_make_plot(results)

    print("\nErgebnisse gespeichert:")
    print(csv_path)
    if results:
        print(Path(csv_path).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
