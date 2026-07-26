from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np

LASER_WAVELENGTH_NM = 787.3
FRINGE_DISTANCE_MM = (LASER_WAVELENGTH_NM / 2) / 1_000_000

base_dir = Path(__file__).resolve().parent
results_dir = base_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Find all CSV files in the directory
files = sorted(base_dir.glob('*.csv'))

def process_file(path, results_dir):
    if not path.exists():
        return None
    
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter=','))
    
    if not rows:
        return None

    data = []
    for r in rows:
        data.append({k: (float(v) if v not in ('', None) else float('nan')) for k, v in r.items()})
        
    positions = [row["Stage_Position_mm"] for row in data]
    times = [row["Relative_Time_s"] for row in data]
    
    # 1. Clean positions (forward-fill 0.0 glitches)
    valid_start_pos = [pos for pos in positions[:10] if pos > 0.0]
    x0 = sum(valid_start_pos) / len(valid_start_pos) if valid_start_pos else positions[0]
    
    cleaned_positions = []
    last_valid_pos = x0
    for pos in positions:
        if pos == 0.0 and x0 > 0.002:
            cleaned_positions.append(last_valid_pos)
        else:
            cleaned_positions.append(pos)
            if pos != 0.0:
                last_valid_pos = pos
                
    # 2. Find the peak index of the forward movement
    search_end = min(len(cleaned_positions), 1500)
    peak_val = -1e9
    peak_idx = 0
    for idx in range(search_end):
        if cleaned_positions[idx] > peak_val:
            peak_val = cleaned_positions[idx]
            peak_idx = idx
            
    # 3. Find start_idx
    start_idx = 0
    tolerance = 0.0005
    for idx in range(peak_idx, -1, -1):
        if abs(cleaned_positions[idx] - x0) <= tolerance:
            start_idx = idx
            break
            
    N = peak_idx - start_idx
    end_idx = min(len(data) - 1, peak_idx + N)
    
    sliced_data = data[start_idx:end_idx+1]
    filtered_rows = [row for row in sliced_data if row['Stage_Position_mm'] >= 0.01]
    
    peak_time = times[peak_idx]
    fwd_filtered = [row for row in filtered_rows if row['Relative_Time_s'] <= peak_time]
    bwd_filtered = [row for row in filtered_rows if row['Relative_Time_s'] > peak_time]
    
    # Equalize length
    min_len = min(len(fwd_filtered), len(bwd_filtered))
    if min_len == 0:
        print(f"Skipping {path.name}: insufficient data points for forward/backward split.")
        return None
    fwd_filtered = fwd_filtered[:min_len]
    bwd_filtered = bwd_filtered[:min_len]
    
    # Combined filtered rows
    combined_filtered = fwd_filtered + bwd_filtered
    
    # 4. Correct phase for backward segment to make sure it decreases relative to the peak
    phi = np.array([row['Unwrapped_Phase_rad'] for row in combined_filtered])
    peak_in_filtered = min_len
    phi_peak = phi[peak_in_filtered]
    phi_bwd = phi[peak_in_filtered:]
    if len(phi_bwd) > 0 and phi_bwd[-1] - phi_bwd[0] > 0:
        phi[peak_in_filtered:] = phi_peak - (phi_bwd - phi_peak)
        
    s1 = np.array([row['Norm_S1'] for row in combined_filtered])
    s2 = np.array([row['Norm_S2'] for row in combined_filtered])
    
    X = np.column_stack([np.cos(phi), np.sin(phi), np.ones_like(phi)])
    beta1 = np.linalg.lstsq(X, s1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X, s2, rcond=None)[0]
    
    s1_fit = X @ beta1
    s2_fit = X @ beta2
    
    s1_centered = s1 - beta1[2]
    s2_centered = s2 - beta2[2]
    
    # 5. Rotation direction (centered)
    angles = np.arctan2(s2_centered, s1_centered)
    delta_angles = np.diff(angles)
    delta_angles = (delta_angles + np.pi) % (2 * np.pi) - np.pi
    
    sum_da_fwd = np.sum(delta_angles[:min_len])
    fwd_dir = "anticlockwise" if sum_da_fwd > 0 else "clockwise"
    
    # Physical requirement: Backward is always opposite to Forward
    bwd_dir = "clockwise" if fwd_dir == "anticlockwise" else "anticlockwise"
    
    # 6. Total and segment differences
    fwd_driven = fwd_filtered[-1]['Stage_Position_mm'] - fwd_filtered[0]['Stage_Position_mm']
    fwd_calc = fwd_filtered[-1]['Calculated_Distance_mm'] - fwd_filtered[0]['Calculated_Distance_mm']
    fwd_diff = fwd_driven - fwd_calc
    
    bwd_driven = bwd_filtered[0]['Stage_Position_mm'] - bwd_filtered[-1]['Stage_Position_mm']
    bwd_calc = bwd_filtered[0]['Calculated_Distance_mm'] - bwd_filtered[-1]['Calculated_Distance_mm']
    bwd_diff = bwd_driven - bwd_calc

    # Calculate fringes by summing absolute differences of Calculated_Distance_mm divided by FRINGE_DISTANCE_MM
    fwd_fringes = sum(
        abs(fwd_filtered[i]['Calculated_Distance_mm'] - fwd_filtered[i-1]['Calculated_Distance_mm']) / FRINGE_DISTANCE_MM
        for i in range(1, len(fwd_filtered))
    )
    bwd_fringes = sum(
        abs(bwd_filtered[i]['Calculated_Distance_mm'] - bwd_filtered[i-1]['Calculated_Distance_mm']) / FRINGE_DISTANCE_MM
        for i in range(1, len(bwd_filtered))
    )

    stage_start = combined_filtered[0]['Stage_Position_mm']
    calc_start = combined_filtered[0]['Calculated_Distance_mm']
    total_driven = combined_filtered[-1]['Stage_Position_mm'] - stage_start
    total_calc = combined_filtered[-1]['Calculated_Distance_mm'] - calc_start
    total_diff = total_driven - total_calc
    
    # Plot two Lissajous plots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f'{path.name} - overview\n'
        f'Net Cycle -> Stage: {total_driven:+.6f} mm | Calc: {total_calc:+.6f} mm | Diff: {total_diff:+.6f} mm',
        fontsize=14, fontweight='bold'
    )
    
    # Common reference unit circle
    ref_theta = [t * 2 * math.pi / 100 for t in range(101)]
    ref_x = [math.cos(t) for t in ref_theta]
    ref_y = [math.sin(t) for t in ref_theta]
    
    # Helper to add small arrows to indicate direction of rotation
    def add_arrows(ax, x_coords, y_coords, color):
        n = len(x_coords)
        if n < 20:
            return
        indices = [int(i * n / 4) for i in range(1, 4)]
        for idx in indices:
            if idx >= n - 3:
                continue
            x_start = x_coords[idx]
            y_start = y_coords[idx]
            x_end = x_coords[idx + 2]
            y_end = y_coords[idx + 2]
            
            dx = x_end - x_start
            dy = y_end - y_start
            
            length = math.hypot(dx, dy)
            if length > 1e-5:
                dx_norm = dx / length
                dy_norm = dy / length
                ax.annotate('', xy=(x_start + dx_norm * 0.05, y_start + dy_norm * 0.05), 
                            xytext=(x_start, y_start),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=15))
                            
    # Forward Lissajous (Left Subplot)
    ax1.plot(ref_x, ref_y, color='gray', linestyle=':', alpha=0.5, label='Unit Circle')
    ax1.plot(s1[:min_len], s2[:min_len], color='green', alpha=0.4, label=f'Forward Data ({min_len} pts)', lw=1.2)
    ax1.plot(s1_fit[:min_len], s2_fit[:min_len], color='black', linestyle='--', label='Fit', lw=1.5)
    add_arrows(ax1, s1_fit[:min_len], s2_fit[:min_len], 'green')
    ax1.set_xlabel('Normalized S1')
    ax1.set_ylabel('Normalized S2')
    ax1.set_title(f'Forward Movement ({fwd_dir.upper()})')
    ax1.set_aspect('equal', 'box')
    
    # Distance comparison info box
    fwd_info = (
        f"Stage Driven: {fwd_driven:9.6f} mm ({fwd_driven*1e3:7.3f} µm)\n"
        f"Calculated:   {fwd_calc:9.6f} mm ({fwd_calc*1e3:7.3f} µm)\n"
        f"Difference:   {fwd_diff:9.6f} mm ({fwd_diff*1e3:7.3f} µm)"
    )
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.85)
    ax1.text(0.05, 0.95, fwd_info, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, fontfamily='monospace')
             
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Backward Lissajous (Right Subplot)
    ax2.plot(ref_x, ref_y, color='gray', linestyle=':', alpha=0.5, label='Unit Circle')
    ax2.plot(s1[min_len:], s2[min_len:], color='red', alpha=0.4, label=f'Backward Data ({min_len} pts)', lw=1.2)
    ax2.plot(s1_fit[min_len:], s2_fit[min_len:], color='black', linestyle='--', label='Fit', lw=1.5)
    add_arrows(ax2, s1_fit[min_len:], s2_fit[min_len:], 'red')
    ax2.set_xlabel('Normalized S1')
    ax2.set_ylabel('Normalized S2')
    ax2.set_title(f'Backward Movement ({bwd_dir.upper()})')
    ax2.set_aspect('equal', 'box')
    
    # Distance comparison info box
    bwd_info = (
        f"Stage Driven: {bwd_driven:9.6f} mm ({bwd_driven*1e3:7.3f} µm)\n"
        f"Calculated:   {bwd_calc:9.6f} mm ({bwd_calc*1e3:7.3f} µm)\n"
        f"Difference:   {bwd_diff:9.6f} mm ({bwd_diff*1e3:7.3f} µm)"
    )
    ax2.text(0.05, 0.95, bwd_info, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, fontfamily='monospace')
             
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = results_dir / f'{path.stem}.png'
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')
    return {
        'output_path': output_path,
        'fwd_driven': fwd_driven,
        'fwd_calc': fwd_calc,
        'fwd_diff': fwd_diff,
        'bwd_driven': bwd_driven,
        'bwd_calc': bwd_calc,
        'bwd_diff': bwd_diff,
        'fwd_fringes': fwd_fringes,
        'bwd_fringes': bwd_fringes
    }

saved_images = []
measurement_results = []

for path in files:
    print(f"Processing {path.name}...")
    try:
        res = process_file(path, results_dir)
        if res:
            saved_images.append(res['output_path'])
            measurement_results.append((path.name, res))
    except Exception as e:
        print(f"Error processing {path.name}: {e}")
        import traceback
        traceback.print_exc()

if saved_images:
    num_images = len(saved_images)
    cols = 2 if num_images > 1 else 1
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16 * cols, 8 * rows), squeeze=False)
    for idx, image_path in enumerate(saved_images):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.imshow(mpimg.imread(image_path))
        ax.axis('off')
        ax.set_title(image_path.name, fontsize=12)
    # Hide any unused subplots
    for idx in range(num_images, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis('off')
    plt.tight_layout()
    combined_path = results_dir / 'plots.jpg'
    # Adjust DPI dynamically so we don't end up with massive files for many subplots
    dpi_val = 100 if num_images > 4 else 200
    fig.savefig(combined_path, dpi=dpi_val, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined plots to {combined_path}')

# Output the summary text file
if measurement_results:
    txt_path = results_dir / 'measurement_summary.txt'
    with txt_path.open('w', encoding='utf-8') as f_out:
        for name, res in measurement_results:
            f_out.write(f"============================================================\n")
            f_out.write(f"Measurement: {name}\n")
            f_out.write(f"============================================================\n")
            f_out.write(f"{'Metric':<15} | {'Forward (mm)':<18} | {'Backward (mm)':<18}\n")
            f_out.write(f"{'-'*16}+{'-'*20}+{'-'*20}\n")
            f_out.write(f"{'Stage Driven':<15} | {res['fwd_driven']:<18.6f} | {res['bwd_driven']:<18.6f}\n")
            f_out.write(f"{'Calculated':<15} | {res['fwd_calc']:<18.6f} | {res['bwd_calc']:<18.6f}\n")
            f_out.write(f"{'Difference':<15} | {res['fwd_diff']:<18.6f} | {res['bwd_diff']:<18.6f}\n")
            f_out.write("\n")
    print(f"Saved summary text file to {txt_path}")

# Output the structured table text file
if measurement_results:
    table_path = results_dir / 'table.txt'
    laser_wavelength_nm = 787.3
    fringe_distance_mm = (laser_wavelength_nm / 2) / 1_000_000
    
    headers1 = f"| {'Measurement':<40} | {'Homodyne Forward':^68} | {'Homodyne Backward':^68} |"
    headers2 = f"| {'':<40} | {'actual moved (mm)':^18} | {'number of fringes':^18} | {'distance from fringes (mm)':^26} | {'actual moved (mm)':^18} | {'number of fringes':^18} | {'distance from fringes (mm)':^26} |"
    separator = f"|{'-'*42}|{'-'*20}|{'-'*20}|{'-'*28}|{'-'*20}|{'-'*20}|{'-'*28}|"
    
    with table_path.open('w', encoding='utf-8') as f_table:
        f_table.write(separator + "\n")
        f_table.write(headers1 + "\n")
        f_table.write(headers2 + "\n")
        f_table.write(separator + "\n")
        
        for name, res in measurement_results:
            fwd_fringes = res['fwd_fringes']
            bwd_fringes = res['bwd_fringes']
            fwd_dist_fringe = fwd_fringes * FRINGE_DISTANCE_MM
            bwd_dist_fringe = bwd_fringes * FRINGE_DISTANCE_MM
            
            row = (
                f"| {name:<40} | "
                f"{res['fwd_driven']:^18.6f} | "
                f"{fwd_fringes:^18.3f} | "
                f"{fwd_dist_fringe:^26.6f} | "
                f"{res['bwd_driven']:^18.6f} | "
                f"{bwd_fringes:^18.3f} | "
                f"{bwd_dist_fringe:^26.6f} |"
            )
            f_table.write(row + "\n")
            
        f_table.write(separator + "\n")
    print(f"Saved table text file to {table_path}")
    
    # Save a copy-paste friendly version for Excel (tab-separated, dot decimal, without Measurement column)
    excel_path = results_dir / 'table_excel.txt'
    with excel_path.open('w', encoding='utf-8') as f_excel:
        f_excel.write(f"Homodyne Forward\t\t\tHomodyne Backward\t\t\n")
        f_excel.write(f"actual moved (mm)\tnumber of fringes\tdistance from fringes (mm)\tactual moved (mm)\tnumber of fringes\tdistance from fringes (mm)\n")
        for name, res in measurement_results:
            fwd_fringes = res['fwd_fringes']
            bwd_fringes = res['bwd_fringes']
            fwd_dist_fringe = fwd_fringes * FRINGE_DISTANCE_MM
            bwd_dist_fringe = bwd_fringes * FRINGE_DISTANCE_MM
            f_excel.write(f"{res['fwd_driven']:.6f}\t{fwd_fringes:.3f}\t{fwd_dist_fringe:.6f}\t{res['bwd_driven']:.6f}\t{bwd_fringes:.3f}\t{bwd_dist_fringe:.6f}\n")
    print(f"Saved Excel copy-paste summary to {excel_path}")
