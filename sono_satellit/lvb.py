import numpy as np


def extract_lvb(mask_values, info, drop=0.18):
    """
    Identify the most likely position of the lumbar vertebral body (LVB).

    Scans from deep to shallow and stops at the first local peak that drops
    by more than `drop` (default 15%). Zero-sum rows in the lower 20% of the
    scan range are skipped before scanning begins.

    Returns a dict with the same keys as extract_linea_alba():
      mask_index, x, y, depth_cm
    """
    sums = [float(np.sum(v)) for v in mask_values]
    n = len(sums)
    global_max = max(sums)
    min_peak = global_max * 0.22

    # Crop everything at and below the topmost zero sum in the lower 20%
    # of the scan range (removes black border + on-image text below it).
    lower_20_start = int(n * 0.8)
    start_i = n - 1
    for i in range(lower_20_start, n):
        if sums[i] == 0:
            start_i = i - 1
            break

    local_peak_val = 0.0
    local_peak_i = start_i
    best_i = None

    for i in range(start_i, -1, -1):
        s = sums[i]
        if s > local_peak_val:
            local_peak_val = s
            local_peak_i = i
        elif local_peak_val > 0 and s < local_peak_val * (1.0 - drop):
            if local_peak_val >= min_peak:
                best_i = local_peak_i
                break
            # Peak too small — reset and keep scanning
            local_peak_val = s
            local_peak_i = i

    if best_i is None:
        return None

    x = info['window_center_x']
    y = info['region_y0'] + best_i * info['shift_step']
    depth_cm = round(best_i * info['shift_step'] * info['physical_delta_y'], 4)

    return {
        'mask_index': best_i + 1,
        'x': x,
        'y': y,
        'depth_cm': depth_cm,
    }
