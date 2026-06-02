import numpy as np


def extract_skin_layer(mask_values, info, drop=0.03, min_peak_fraction=0.20):
    """
    Detect the skin layer using a sliding window brightness scan.

    Scans from shallow to deep and stops at the first local peak that drops
    by more than `drop` (default 20%) and exceeds `min_peak_fraction`
    (default 20%) of the global maximum brightness sum.

    Parameters
    ----------
    mask_values : list of np.ndarray
        Per-window pixel value arrays as returned by generate_shifted_masks().
    info : dict
        Region dict as returned by generate_shifted_masks(), containing
        region_y0, shift_step, and physical_delta_y.
    drop : float
        Fractional drop from the local peak that triggers detection (0.20 = 20%).
    min_peak_fraction : float
        Minimum peak value as a fraction of the global max (0.20 = 20%).

    Returns
    -------
    int
        Absolute y pixel coordinate of the detected skin surface.

    Raises
    ------
    ValueError
        If no qualifying peak is found.
    """
    sums = [float(np.sum(v)) for v in mask_values]
    global_max = max(sums)
    min_peak = global_max * min_peak_fraction

    local_peak_val = 0.0
    local_peak_i = 0

    for i, s in enumerate(sums):
        if s > local_peak_val:
            local_peak_val = s
            local_peak_i = i
        elif local_peak_val > 0 and s < local_peak_val * (1.0 - drop):
            if local_peak_val >= min_peak:
                return info['region_y0'] + local_peak_i * info['shift_step']
            local_peak_val = s
            local_peak_i = i

    raise ValueError("No skin layer detected in sliding window scan")
