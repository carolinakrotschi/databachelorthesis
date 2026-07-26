#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
WAVELENGTH_NM = 787.324
DISTANCE_PER_FRINGE_MM = WAVELENGTH_NM * 1e-6 / 2.0


def load_csv(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "Relative_Time_s": [],
        "Raw_Voltage_V": [],
        "Intensity": [],
        "Fringe_Count": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["Relative_Time_s"].append(float(row["Relative_Time_s"]))
            data["Raw_Voltage_V"].append(float(row["Raw_Voltage_V"]))
            if "Intensity" in row and row["Intensity"] != "":
                data["Intensity"].append(float(row["Intensity"]))
            data["Fringe_Count"].append(float(row["Fringe_Count"]))

    return data


def load_camera_csv(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "Relative_Time_s": [],
        "Intensity": [],
        "Fringe_Count": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["Relative_Time_s"].append(float(row["Relative_Time_s"]))
            data["Intensity"].append(float(row["Intensity"]))
            data["Fringe_Count"].append(float(row["Fringe_Count"]))

    return data


def load_homodyneforward_csv(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "Relative_Time_s": [],
        "Raw_S1_V": [],
        "Raw_S2_V": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["Relative_Time_s"].append(float(row["Relative_Time_s"]))
            data["Raw_S1_V"].append(float(row["Raw_S1_V"]))
            data["Raw_S2_V"].append(float(row["Raw_S2_V"]))

    return data


def plot_raw_individual(csv_path: Path, data: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["Relative_Time_s"], data["Raw_Voltage_V"], linewidth=1.2)
    ax.set_title(f"Raw Voltage vs Time - {csv_path.stem}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Raw Voltage [V]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{csv_path.stem}_raw_voltage.png", dpi=200)
    plt.close(fig)


def plot_raw_combined(series: list[tuple[str, dict[str, list[float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for idx, (label, data) in enumerate(series):
        ax.plot(data["Relative_Time_s"], data["Raw_Voltage_V"], linewidth=1.1, label=label)
    ax.set_title("Raw Voltage vs Time - All Files")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Raw Voltage [V]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "all_raw_voltage.png", dpi=200)
    plt.close(fig)


def plot_fringe_summary(series: list[tuple[str, dict[str, list[float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    max_fringes = [max(data["Fringe_Count"]) for _, data in series]
    x = 1.0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, ((label, _), fringe_count) in enumerate(zip(series, max_fringes, strict=True)):
        color = colors[idx % len(colors)]
        ax.scatter([x], [fringe_count], s=70, color=color, zorder=3)
        ax.annotate(
            label,
            (x, fringe_count),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            fontsize=10,
        )

    ax.set_xticks([x], ["Photodiode"])
    ax.set_xlim(0.7, 1.3)

    y_min = min(max_fringes) - 1
    y_max = max(max_fringes) + 1
    ax.set_ylim(y_min, y_max)
    ax.set_title("Maximum Fringe Count per Photodiode")
    ax.set_xlabel("Photodiode")
    ax.set_ylabel("Max Fringe Count")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fringe_count_by_diode.png", dpi=200)
    plt.close(fig)


def plot_camera_intensity(series: list[tuple[str, dict[str, list[float]]]]) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 11), sharex=True)
    if len(series) == 1:
        axes = [axes]

    thesis_label_size = 18
    thesis_tick_size = 16
    thesis_title_size = 18
    x_end = 1200.0

    display_names = {
        "kamera2": "measurement 1",
        "kamera3": "measurement 2",
        "kamera4": "measurement 3",
    }

    for ax, (label, data) in zip(axes, series, strict=True):
        time = data["Relative_Time_s"]
        values = data["Intensity"]
        if not time or not values:
            continue

        t0 = time[0]
        shifted_time = [t - t0 for t in time]
        clipped = [(t, v) for t, v in zip(shifted_time, values, strict=True) if t <= x_end]
        if not clipped:
            continue

        plot_time = [t for t, _ in clipped]
        plot_values = [v for _, v in clipped]

        y_min = min(plot_values)
        y_max = max(plot_values)
        y_span = y_max - y_min
        padding = max(y_span * 8.5, 40.0)

        ax.plot(plot_time, plot_values, linewidth=1.1)
        ax.set_ylabel("Intensity [arb. unit]", fontsize=thesis_label_size)
        ax.set_title(display_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_end)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)

    axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "camera_intensity_over_time.png", dpi=200)
    plt.close(fig)


def plot_diode_intensity(series: list[tuple[str, dict[str, list[float]]]]) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 11), sharex=True)
    if len(series) == 1:
        axes = [axes]

    thesis_label_size = 20
    thesis_tick_size = 16
    thesis_title_size = 18
    x_end = 1250.0
    line_color = "#1f77b4"

    display_names = {
        "nullkommafuenfdiode2_1_neu": "measurement 1",
        "nullkommafuenfdiode2_2_neu": "measurement 2",
        "nullkommafuenfdiode2_3_neu": "measurement 3",
    }

    for ax, (label, data) in zip(axes, series, strict=True):
        time = data["Relative_Time_s"]
        values = data["Raw_Voltage_V"]
        if not time or not values:
            continue

        t0 = time[0]
        shifted_time = [t - t0 for t in time]
        if not shifted_time:
            continue

        max_time = max(shifted_time)
        if max_time <= 0:
            continue

        plot_time = [t / max_time * x_end for t in shifted_time]
        plot_values = values

        y_min = min(plot_values)
        y_max = max(plot_values)
        y_span = y_max - y_min
        padding = max(y_span * 0.02, 0.02)

        ax.plot(plot_time, plot_values, linewidth=1.1, color=line_color)
        ax.set_ylabel("Raw Voltage [V]", fontsize=thesis_label_size, labelpad=8)
        ax.set_title(display_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_end)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)

    axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)
    fig.tight_layout()
    fig.subplots_adjust(left=0.13)
    fig.savefig(RESULTS_DIR / "diode_intensity_over_time.png", dpi=200)
    plt.close(fig)


def plot_camera_diode_side_by_side(
    camera_series: list[tuple[str, dict[str, list[float]]]],
    diode_series: list[tuple[str, dict[str, list[float]]]],
) -> None:
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.1)

    camera_grid = outer[0].subgridspec(len(camera_series), 1, hspace=0.18)
    diode_grid = outer[1].subgridspec(len(diode_series), 1, hspace=0.18)

    thesis_label_size = 15
    thesis_tick_size = 11
    thesis_title_size = 13
    x_end = 1200.0
    line_color = "#1f77b4"

    camera_names = {
        "kamera2": "measurement 1",
        "kamera3": "measurement 2",
        "kamera4": "measurement 3",
    }
    diode_names = {
        "nullkommafuenfdiode2_1_neu": "measurement 1",
        "nullkommafuenfdiode2_2_neu": "measurement 2",
        "nullkommafuenfdiode2_3_neu": "measurement 3",
    }

    fig.text(0.25, 1.015, "Camera", ha="center", va="top", fontsize=17)
    fig.text(0.75, 1.015, "Photodiode", ha="center", va="top", fontsize=17)

    camera_axes = []
    for idx, (label, data) in enumerate(camera_series):
        ax = fig.add_subplot(camera_grid[idx, 0])
        camera_axes.append(ax)
        time = data["Relative_Time_s"]
        values = data["Intensity"]
        if time and values:
            t0 = time[0]
            shifted_time = [t - t0 for t in time]
            max_time = max(shifted_time) if shifted_time else 0
            if max_time > 0:
                plot_time = [t / max_time * x_end for t in shifted_time]
                plot_values = values
                y_max = max(plot_values)
                ax.plot(plot_time, plot_values, linewidth=1.1, color=line_color)
                ax.set_xlim(0, x_end)
                ax.set_ylim(0, y_max + 6400)
        ax.set_ylabel("Intensity [arb. unit]", fontsize=thesis_label_size)
        ax.set_title(camera_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)
        ax.grid(True, alpha=0.3)

    if camera_axes:
        camera_axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)

    diode_axes = []
    for idx, (label, data) in enumerate(diode_series):
        ax = fig.add_subplot(diode_grid[idx, 0])
        diode_axes.append(ax)
        time = data["Relative_Time_s"]
        values = data["Raw_Voltage_V"]
        if time and values:
            t0 = time[0]
            shifted_time = [t - t0 for t in time]
            max_time = max(shifted_time) if shifted_time else 0
            if max_time > 0:
                plot_time = [t / max_time * x_end for t in shifted_time]
                plot_values = values
                y_min = min(plot_values)
                y_max = max(plot_values)
                y_span = y_max - y_min
                padding = max(y_span * 0.02, 0.02)
                ax.plot(plot_time, plot_values, linewidth=1.1, color=line_color)
                ax.set_xlim(0, x_end)
                ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_ylabel("Raw Voltage [V]", fontsize=thesis_label_size)
        ax.set_title(diode_names.get(label, label), loc="left", fontsize=thesis_title_size)
        ax.tick_params(axis="both", labelsize=thesis_tick_size)
        ax.grid(True, alpha=0.3)

    if diode_axes:
        diode_axes[-1].set_xlabel("Time [s]", fontsize=thesis_label_size)

    fig.savefig(RESULTS_DIR / "camera_diode_side_by_side.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_camera_fringe_delta(series: list[tuple[str, dict[str, list[float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    x = 1.0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, data) in enumerate(series):
        fringe_count = data["Fringe_Count"]
        if not fringe_count:
            continue
        delta = fringe_count[-1] - fringe_count[0]
        color = colors[idx % len(colors)]
        ax.scatter([x], [delta], s=90, color=color, zorder=3)
        ax.annotate(
            label,
            (x, delta),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            fontsize=10,
        )

    ax.set_xticks([x], ["Camera"])
    ax.set_xlim(0.7, 1.3)
    ax.set_xlabel("Camera")
    ax.set_ylabel("Fringe Count Delta")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "camera_fringe_delta.png", dpi=200)
    plt.close(fig)


def plot_fringe_summary_combined(
    camera_series: list[tuple[str, dict[str, list[float]]]],
    diode_series: list[tuple[str, dict[str, list[float]]]],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax_right = ax.twinx()

    thesis_label_size = 26
    thesis_tick_size = 22
    point_color = "#1f77b4"
    x_camera = 1.0
    x_diode = 2.0

    camera_values = []
    for _, data in camera_series:
        fringe_count = data["Fringe_Count"]
        if not fringe_count:
            continue
        delta = fringe_count[-1] - fringe_count[0]
        camera_values.append(delta)
        ax.scatter([x_camera], [delta], s=220, color=point_color, zorder=3)

    diode_values = [max(data["Fringe_Count"]) for _, data in diode_series]
    for fringe_count in diode_values:
        ax.scatter([x_diode], [fringe_count], s=220, color=point_color, zorder=3)

    ax.set_xticks([x_camera, x_diode], ["Camera", "Photodiode"])
    ax.set_xlim(0.6, 2.4)
    ax.set_ylabel("Fringe Count", fontsize=thesis_label_size)
    ax.tick_params(axis="both", labelsize=thesis_tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(1050, 1260)

    if camera_values:
        camera_delta = max(camera_values) - min(camera_values)
        ax.text(
            0.02,
            0.98,
            f"Camera Δ = {camera_delta:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=thesis_label_size,
        )
    if diode_values:
        diode_delta = max(diode_values) - min(diode_values)
        ax.text(
            0.98,
            0.98,
            f"Photodiode Δ = {diode_delta:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=thesis_label_size,
        )

    left_ylim = ax.get_ylim()
    right_ylim = tuple(value * DISTANCE_PER_FRINGE_MM for value in left_ylim)
    ax_right.set_ylim(right_ylim)
    ax_right.set_ylabel("Distance [mm]", fontsize=thesis_label_size)
    ax_right.tick_params(axis="y", labelsize=thesis_tick_size, right=True, labelright=True)
    ax_right.spines["right"].set_visible(True)
    ax_right.yaxis.set_ticks_position("right")

    left_ticks = ax.get_yticks()
    ax_right.set_yticks([tick * DISTANCE_PER_FRINGE_MM for tick in left_ticks])

    fig.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.12)
    fig.savefig(RESULTS_DIR / "fringe_summary_combined.png", dpi=200)
    plt.close(fig)


def plot_measurement_2_zoom(
    camera_data: dict[str, list[float]],
    diode_data: dict[str, list[float]],
) -> None:
    fig, (ax_camera, ax_diode) = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    thesis_label_size = 18
    thesis_tick_size = 14
    zoom_center_s = 811.0
    zoom_half_width_s = 5.0

    camera_time = camera_data["Relative_Time_s"]
    camera_values = camera_data["Intensity"]
    if camera_time and camera_values:
        t0 = camera_time[0]
        shifted_time = [t - t0 for t in camera_time]
        camera_plot = [
            (t, v)
            for t, v in zip(shifted_time, camera_values, strict=True)
            if zoom_center_s - zoom_half_width_s <= t <= zoom_center_s + zoom_half_width_s
        ]
        if camera_plot:
            plot_time = [t for t, _ in camera_plot]
            plot_values = [v for _, v in camera_plot]
            ax_camera.plot(plot_time, plot_values, linewidth=1.3, color="#1f77b4")
            ax_camera.set_xlim(zoom_center_s - zoom_half_width_s, zoom_center_s + zoom_half_width_s)
            y_min = min(plot_values)
            y_max = max(plot_values)
            y_span = y_max - y_min
            padding = max(y_span * 0.15, 40.0)
            ax_camera.set_ylim(y_min - padding, y_max + padding)
    ax_camera.set_title("Camera - measurement 2", fontsize=thesis_label_size)
    ax_camera.set_xlabel("Time [s]", fontsize=thesis_label_size)
    ax_camera.set_ylabel("Intensity [arb. unit]", fontsize=thesis_label_size)
    ax_camera.tick_params(axis="both", labelsize=thesis_tick_size)
    ax_camera.grid(True, alpha=0.3)

    diode_time = diode_data["Relative_Time_s"]
    diode_values = diode_data["Raw_Voltage_V"]
    if diode_time and diode_values:
        t0 = diode_time[0]
        shifted_time = [t - t0 for t in diode_time]
        diode_plot = [
            (t, v)
            for t, v in zip(shifted_time, diode_values, strict=True)
            if zoom_center_s - zoom_half_width_s <= t <= zoom_center_s + zoom_half_width_s
        ]
        if diode_plot:
            plot_time = [t for t, _ in diode_plot]
            plot_values = [v for _, v in diode_plot]
            ax_diode.plot(plot_time, plot_values, linewidth=1.3, color="#1f77b4")
            ax_diode.set_xlim(zoom_center_s - zoom_half_width_s, zoom_center_s + zoom_half_width_s)
            ax_diode.set_ylim(2.67, 2.83)
    ax_diode.set_title("Photodiode - measurement 2", fontsize=thesis_label_size)
    ax_diode.set_xlabel("Time [s]", fontsize=thesis_label_size)
    ax_diode.set_ylabel("Raw Voltage [V]", fontsize=thesis_label_size)
    ax_diode.tick_params(axis="both", labelsize=thesis_tick_size)
    ax_diode.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "measurement2_camera_photodiode_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_homodyneforward1_zoom(path: Path) -> None:
    data = load_homodyneforward_csv(path)

    time = data["Relative_Time_s"]
    s1 = data["Raw_S1_V"]
    s2 = data["Raw_S2_V"]
    if not time or not s1 or not s2:
        return

    t_min = min(time)
    t_max = max(time)
    center_s = 388.0
    half_width_s = 5.0
    left_s = center_s - half_width_s
    right_s = center_s + half_width_s

    window_s1 = [(t, v) for t, v in zip(time, s1, strict=True) if left_s <= t <= right_s]
    window_s2 = [(t, v) for t, v in zip(time, s2, strict=True) if left_s <= t <= right_s]
    if not window_s1 or not window_s2:
        return

    fig, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    t1 = [t for t, _ in window_s1]
    v1 = [v for _, v in window_s1]
    ax_s1.plot(t1, v1, linewidth=1.2, color="#1f77b4")
    ax_s1.set_ylabel("Raw S1 [V]", fontsize=18)
    ax_s1.set_title("Homodyne forward 1 - zoom", fontsize=18)
    ax_s1.grid(True, alpha=0.3)

    t2 = [t for t, _ in window_s2]
    v2 = [v for _, v in window_s2]
    ax_s2.plot(t2, v2, linewidth=1.2, color="#ff7f0e")
    ax_s2.set_ylabel("Raw S2 [V]", fontsize=18)
    ax_s2.set_xlabel("Time [s]", fontsize=18)
    ax_s2.grid(True, alpha=0.3)

    ax_s1.set_xlim(left_s, right_s)
    ax_s1.tick_params(axis="both", labelsize=14)
    ax_s2.tick_params(axis="both", labelsize=14)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "homodyneforward1_zoom_middle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_files = sorted(BASE_DIR.glob("nullkommafuenfdiode2_*_neu.csv"))
    camera_files = [
        ("kamera2", BASE_DIR / "kamera2.csv"),
        ("kamera3", BASE_DIR / "kamera3.csv"),
        ("kamera4", BASE_DIR / "kamera4.csv"),
    ]
    if not csv_files and not any(path.exists() for _, path in camera_files):
        raise SystemExit("No matching measurement files found.")

    series: list[tuple[str, dict[str, list[float]]]] = []
    for csv_path in csv_files:
        data = load_csv(csv_path)
        series.append((csv_path.stem, data))
        plot_raw_individual(csv_path, data)

    plot_raw_combined(series)
    plot_fringe_summary(series)

    camera_series: list[tuple[str, dict[str, list[float]]]] = []
    for label, path in camera_files:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        camera_series.append((label, load_camera_csv(path)))

    if camera_series:
        plot_camera_intensity(camera_series)
        plot_camera_fringe_delta(camera_series)

    if camera_series and series:
        plot_fringe_summary_combined(camera_series, series)

    camera_measurement_2 = next((data for label, data in camera_series if label == "kamera3"), None)
    photodiode_measurement_2 = next((data for label, data in series if label == "nullkommafuenfdiode2_2_neu"), None)
    if camera_measurement_2 and photodiode_measurement_2:
        plot_measurement_2_zoom(camera_measurement_2, photodiode_measurement_2)

    homodyneforward1_path = BASE_DIR / "homodyneforward1.csv"
    if homodyneforward1_path.exists():
        plot_homodyneforward1_zoom(homodyneforward1_path)

    diode_series = []
    for path in [
        BASE_DIR / "nullkommafuenfdiode2_1_neu.csv",
        BASE_DIR / "nullkommafuenfdiode2_2_neu.csv",
        BASE_DIR / "nullkommafuenfdiode2_3_neu.csv",
    ]:
        if path.exists():
            diode_series.append((path.stem, load_csv(path)))

    if diode_series:
        plot_diode_intensity(diode_series)

    if camera_series and diode_series:
        plot_camera_diode_side_by_side(camera_series, diode_series)


if __name__ == "__main__":
    main()
