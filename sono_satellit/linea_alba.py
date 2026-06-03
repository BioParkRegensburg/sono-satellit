import numpy as np


def extract_linea_alba(mask_values, info, peak_min=0.65, drop=0.12,
                       skin_y=None, skin_drop=0.08):
    """
    Identify the most likely position of the linea alba.

    When `skin_y` is provided the scan uses a two-phase strategy:

    Phase 1 — starting from the skin mask index, wait until brightness drops
    by at least `skin_drop` (default 10 %) relative to the skin brightness.
    Peaks observed before this drop are ignored so the skin layer itself is
    never mistaken for the linea alba.

    Phase 2 — once the drop is confirmed, apply the two complementary
    conditions below to the remaining windows.

    If Phase 1 never completes (the 10 % drop is never seen), a warning is
    printed and the function falls back to scanning from the top of the image.

    When `skin_y` is None the scan starts from the top of the image directly
    with the two conditions below (original behaviour).

    Condition used in Phase 2 (and in the skin-y-less path):

    Peak detection: if a local brightness peak exceeds `peak_min` × global_max
    and the brightness subsequently drops by more than `drop` (relative to that
    peak), the peak position is returned as the linea alba. Points with a strong
    upslope but no pronounced downslope (saddle-like) are therefore rejected.

    Fallback: argmax of sums in the search range, if the condition never fires.

    Parameters
    ----------
    mask_values : list of np.ndarray
        Per-mask pixel-value arrays, ordered from shallowest to deepest.
    info : dict
        Region/scale dict from generate_shifted_masks().
    peak_min : float
        Minimum fraction of global_max a local peak must reach to qualify
        for early-exit detection (default 0.7 = 70 %).
    drop : float
        Relative drop from a qualifying local peak that triggers early exit
        (default 0.1 = 10 %).
    skin_y : int or None
        Absolute y pixel coordinate of the detected skin layer. When given,
        Phase 1 is activated.
    skin_drop : float
        Minimum fractional drop from skin brightness required before the
        linea alba search begins (default 0.08 = 8 %).

    Returns a dict with:
      mask_index   -- 1-based index of the selected mask
      x            -- pixel x coordinate (horizontal centre of the imaging region)
      y            -- pixel y coordinate (upper edge of the selected mask)
      depth_cm     -- physical depth of the upper edge from the top of the imaging region
    """
    sums = [float(np.sum(v)) for v in mask_values]
    global_max = max(sums)

    # Determine the index at which to start the linea alba search.
    start_i = 0
    if skin_y is not None:
        skin_i = (skin_y - info['region_y0']) // info['shift_step']
        skin_i = max(0, min(skin_i, len(sums) - 1))
        skin_brightness = sums[skin_i]
        drop_threshold = skin_brightness * (1.0 - skin_drop)

        # Phase 1: find where brightness drops >= skin_drop below skin level.
        phase1_done = False
        for i in range(skin_i, len(sums)):
            if sums[i] < drop_threshold:
                start_i = i
                phase1_done = True
                break

        if not phase1_done:
            print(
                "Warning: brightness never dropped 10 % below skin level; "
                "falling back to top-of-image linea alba scan."
            )
            start_i = 0

    # Phase 2 / main scan: apply peak-tracking and threshold detection.
    best_i = int(np.argmax(sums[start_i:])) + start_i  # fallback within search range

    local_peak_val = 0.0
    local_peak_i = start_i

    for i in range(start_i, len(sums)):
        s = sums[i]
        if s > local_peak_val:
            local_peak_val = s
            local_peak_i = i
        elif (local_peak_i > start_i
              and local_peak_val >= global_max * peak_min
              and s < local_peak_val * (1.0 - drop)):
            best_i = local_peak_i
            break

    # Posterior border: first window after the anterior peak where brightness
    # drops to (1 - drop) of the anterior peak brightness.
    anterior_peak_val = sums[best_i]
    posterior_i = len(sums) - 1  # fallback: last window
    for j in range(best_i + 1, len(sums)):
        if sums[j] < anterior_peak_val * (1.0 - drop):
            posterior_i = j
            break

    x = info['window_center_x']

    anterior_y = info['region_y0'] + best_i * info['shift_step']
    anterior_depth_cm = round(best_i * info['shift_step'] * info['physical_delta_y'], 4)

    posterior_y = info['region_y0'] + posterior_i * info['shift_step']
    posterior_depth_cm = round(posterior_i * info['shift_step'] * info['physical_delta_y'], 4)

    return {
        'anterior': {
            'mask_index': best_i + 1,
            'x': x,
            'y': anterior_y,
            'depth_cm': anterior_depth_cm,
        },
        'posterior': {
            'mask_index': posterior_i + 1,
            'x': x,
            'y': posterior_y,
            'depth_cm': posterior_depth_cm,
        },
    }
