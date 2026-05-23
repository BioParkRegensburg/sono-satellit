import cv2
import numpy as np


def _find_header_end(gray, x_left, x_right, fallback_y=0):
    """
    Detect where the burned-in machine header ends.

    Reads the pixel brightness at the top of the left and right edges
    (y=0) and scans downward until both columns drop to ≤ 50% of those
    initial values. That row is the first line of the true black background
    — i.e. where the scan area begins.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 pixel array (full image).
    x_left : int
        Column index for the left edge sample (typically region_x0).
    x_right : int
        Column index for the right edge sample (typically region_x1).
    fallback_y : int
        Row to return if no dark transition is found (e.g. region_y0).

    Returns
    -------
    int
        Row index of the first line below the header.
    """
    left_top = int(gray[0, x_left])
    right_top = int(gray[0, x_right])

    left_threshold = left_top * 0.5
    right_threshold = right_top * 0.5

    for y in range(gray.shape[0]):
        if gray[y, x_left] <= left_threshold and gray[y, x_right] <= right_threshold:
            return y

    return fallback_y


def extract_skin_layer(gray, region, min_threshold=100, max_threshold=200):
    """
    Detect the skin layer in a grayscale pixel array.

    First determines where the burned-in machine header ends by finding the
    first row where both the left and right edges of the imaging region drop
    to ≤ 50% of their top-row brightness. Skin edge detection then searches
    the middle 20% of the full image width, starting from that header-end
    row and spanning one quarter of the full image height downward.

    Finds the longest contiguous run of y-rows that contain Canny edges,
    then returns the lowest y coordinate (maximum y) of that run as the
    skin surface position.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 pixel array (full image, not cropped).
    region : dict
        Region dict as returned by read_dicom_region().
    min_threshold : int
        Lower hysteresis threshold for Canny edge detection.
    max_threshold : int
        Upper hysteresis threshold for Canny edge detection.

    Returns
    -------
    int
        Absolute y pixel coordinate of the lowest point of the main
        continuous skin edge.

    Raises
    ------
    ValueError
        If no edges are detected in the search area.
    """
    rx0 = region['region_x0']
    rx1 = region['region_x1']
    ry0 = region['region_y0']

    # Determine where the header ends
    header_end_y = _find_header_end(gray, rx0, rx1, fallback_y=ry0)

    # Middle 20% of the full image width
    h, w = gray.shape
    x_center = w // 2
    x_half = w // 10  # 10% each side = 20% total
    x_left = max(0, x_center - x_half)
    x_right = min(w, x_center + x_half)

    # Search one quarter of the full image height starting below the header
    y_start = header_end_y
    y_end = min(h, y_start + h // 4)

    crop = gray[y_start:y_end, x_left:x_right]

    if crop.dtype != np.uint8:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(crop, min_threshold, max_threshold)
    y_coords, _ = np.where(edges == 255)

    if len(y_coords) == 0:
        raise ValueError(
            "No skin layer edge detected in the search area "
            f"(y={y_start}–{y_end}, x={x_left}–{x_right})."
        )

    # Find the longest contiguous run of y-rows that contain edge pixels
    rows_with_edges = sorted(set(y_coords.tolist()))

    runs = []
    current_run = [rows_with_edges[0]]
    for r in rows_with_edges[1:]:
        if r == current_run[-1] + 1:
            current_run.append(r)
        else:
            runs.append(current_run)
            current_run = [r]
    runs.append(current_run)

    longest_run = max(runs, key=len)

    # Lowest point (max y) of the continuous edge, mapped to full-image coordinates
    return y_start + int(max(longest_run))
