#!/usr/bin/env python
# coding: utf-8

"""
Process sonographic DICOM images to extract subcutaneous adipose tissue measurements.
"""

import argparse
import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from sono_satellit.skin_detection import extract_skin_layer


def _extract_pixel_array(ds):
    """
    Extract a grayscale uint8 pixel array from a DICOM dataset.

    Handles multi-frame datasets (uses the first frame) and colour images
    (converts RGB → grayscale). Normalises non-uint8 arrays to 0-255.
    """
    pixel_array = ds.pixel_array

    n_frames = int(getattr(ds, 'NumberOfFrames', 1))
    if n_frames > 1:
        pixel_array = pixel_array[0]

    samples_per_pixel = int(getattr(ds, 'SamplesPerPixel', 1))
    if samples_per_pixel > 1:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

    if pixel_array.dtype != np.uint8:
        pixel_array = cv2.normalize(
            pixel_array, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return pixel_array


def read_dicom_region(ds):
    """
    Extract physical pixel spacing and imaging region bounds from a DICOM dataset.

    Reads (0018,6011) SequenceOfUltrasoundRegions and returns a dict with:
      region_x0, region_x1, region_y0, region_y1  -- pixel bounds of the imaging area
      physical_delta_y                             -- cm per pixel in the Y direction
    """
    try:
        regions = ds[0x0018, 0x6011].value
    except KeyError:
        raise ValueError(
            "DICOM file is missing SequenceOfUltrasoundRegions (0018,6011). "
            "Cannot determine physical scale."
        )

    region = regions[0]

    physical_units_y = int(region[0x0018, 0x6026].value)  # PhysicalUnitsYDirection
    physical_delta_y = float(region[0x0018, 0x602E].value)  # PhysicalDeltaY

    if physical_units_y == 4:  # mm -> convert to cm
        physical_delta_y /= 10.0
    elif physical_units_y != 3:
        raise ValueError(
            f"Unsupported Y-axis unit code {physical_units_y} in DICOM region "
            "(expected 3=cm or 4=mm)."
        )

    return {
        'region_x0': int(region[0x0018, 0x6018].value),  # RegionLocationMinX0
        'region_y0': int(region[0x0018, 0x601A].value),  # RegionLocationMinY0
        'region_x1': int(region[0x0018, 0x601C].value),  # RegionLocationMaxX1
        'region_y1': int(region[0x0018, 0x601E].value),  # RegionLocationMaxY1
        'physical_delta_y': physical_delta_y,
    }


def create_curve_mask(img_shape, x_min, x_max, y_top, thickness):
    """Create a filled rectangular mask strip."""
    h, w = img_shape

    rect = np.array([
        [x_min, y_top],
        [x_max, y_top],
        [x_max, y_top + thickness],
        [x_min, y_top + thickness],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect], 255)

    return mask, rect


def generate_shifted_masks(ds, thickness=20, shift_step=5):
    """
    Generate shifted rectangular masks sweeping the full imaging region.

    Returns:
        all_mask_values  -- list of 1-D pixel-value arrays, one per mask position
        info             -- dict with region bounds, scale, thickness, and shift_step
    """
    region = read_dicom_region(ds)

    pixel_array = _extract_pixel_array(ds)

    region_x0 = region['region_x0']
    region_x1 = region['region_x1']
    region_y0 = region['region_y0']
    region_y1 = region['region_y1']

    iterations = (region_y1 - region_y0) // shift_step

    all_mask_values = []

    for i in range(iterations):
        y_top = region_y0 + i * shift_step

        mask, _ = create_curve_mask(
            pixel_array.shape,
            region_x0,
            region_x1,
            y_top,
            thickness,
        )

        pixel_values = pixel_array[mask == 255]
        all_mask_values.append(pixel_values)

    return all_mask_values, {**region, 'thickness': thickness, 'shift_step': shift_step}


def extract_linea_alba(mask_values, info, threshold=0.9, peak_min=0.7, drop=0.1):
    """
    Identify the most likely position of the linea alba.

    Scans masks from shallow to deep using two complementary conditions:

    1. Early-exit peak detection: if a local brightness peak exceeds
       `peak_min` × global_max and the brightness subsequently drops by
       more than `drop` (relative to that peak), the peak position is
       returned as the linea alba. This handles cases where a sub-threshold
       spike clearly represents the linea alba before a deeper bright plateau.

    2. Main threshold: the first mask whose brightness reaches `threshold` ×
       global_max is returned. Fires on the upslope, so it is never confused
       by a post-peak downslope.

    Fallback: global argmax, if neither condition is triggered.

    Parameters
    ----------
    mask_values : list of np.ndarray
        Per-mask pixel-value arrays, ordered from shallowest to deepest.
    info : dict
        Region/scale dict from generate_shifted_masks().
    threshold : float
        Fraction of global_max required on the upslope (default 0.9 = 90 %).
    peak_min : float
        Minimum fraction of global_max a local peak must reach to qualify
        for early-exit detection (default 0.7 = 70 %).
    drop : float
        Relative drop from a qualifying local peak that triggers early exit
        (default 0.1 = 10 %).

    Returns a dict with:
      mask_index   -- 1-based index of the selected mask
      x            -- pixel x coordinate (horizontal centre of the imaging region)
      y            -- pixel y coordinate (upper edge of the selected mask)
      depth_cm     -- physical depth of the upper edge from the top of the imaging region
    """
    sums = [float(np.sum(v)) for v in mask_values]
    global_max = max(sums)

    local_peak_val = 0.0
    local_peak_i = 0
    best_i = int(np.argmax(sums))  # fallback

    for i, s in enumerate(sums):
        if s > local_peak_val:
            local_peak_val = s
            local_peak_i = i
        elif (local_peak_val >= global_max * peak_min
              and s < local_peak_val * (1.0 - drop)):
            best_i = local_peak_i
            break

        if s >= global_max * threshold:
            best_i = i
            break

    x = (info['region_x0'] + info['region_x1']) // 2
    y = info['region_y0'] + best_i * info['shift_step']
    depth_cm = round(best_i * info['shift_step'] * info['physical_delta_y'], 4)

    return {
        'mask_index': best_i + 1,
        'x': x,
        'y': y,
        'depth_cm': depth_cm,
    }


def save_debug_image(image_path, x, y, skin_y=None, debug_dir='./debug-images'):
    """
    Save an annotated copy of a DICOM image with landmarks marked.

    Draws a red circle + crosshair at (x, y) for the linea alba.
    Optionally draws a green circle + crosshair at (x, skin_y) for the skin layer.
    Writes the result to debug_dir/<original_filename>.png.
    """
    ds = pydicom.dcmread(image_path)
    gray = _extract_pixel_array(ds)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    radius = 12
    thickness = 2
    arm = radius + 8

    # Green crosshair + circle for skin layer
    if skin_y is not None:
        color = (0, 255, 0)
        cv2.circle(bgr, (x, skin_y), radius, color, thickness)
        cv2.line(bgr, (x - arm, skin_y), (x + arm, skin_y), color, thickness)
        cv2.line(bgr, (x, skin_y - arm), (x, skin_y + arm), color, thickness)

    # Red crosshair + circle for linea alba
    color = (0, 0, 255)
    cv2.circle(bgr, (x, y), radius, color, thickness)
    cv2.line(bgr, (x - arm, y), (x + arm, y), color, thickness)
    cv2.line(bgr, (x, y - arm), (x, y + arm), color, thickness)

    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(debug_dir, stem + '.png')
    cv2.imwrite(out_path, bgr)
    return out_path


def save_debug_curve(image_path, mask_values, info, linea_alba_y, debug_dir='./debug-images'):
    """
    Save a brightness-curve plot for the sliding-window scan.

    X axis   : y pixel position of the top edge of each mask window.
    Left axis  (blue)   : sum of pixel brightness per window.
    Right axis (yellow) : std of pixel brightness per window.
    A vertical dashed red line marks the detected linea alba y position.

    Saved to debug_dir/<stem>_graph.png.
    """
    shift_step = info['shift_step']
    region_y0 = info['region_y0']

    x_vals = [region_y0 + i * shift_step for i in range(len(mask_values))]
    sums = [float(np.sum(v)) for v in mask_values]
    stds = [float(np.std(v)) for v in mask_values]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(x_vals, sums, color='blue', linewidth=1.2, label='Brightness sum')
    ax1.set_xlabel('Window top edge y (px)')
    ax1.set_ylabel('Brightness sum', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(x_vals, stds, color='goldenrod', linewidth=1.2, label='Std')
    ax2.set_ylabel('Std', color='goldenrod')
    ax2.tick_params(axis='y', labelcolor='goldenrod')

    ax1.axvline(x=linea_alba_y, color='red', linestyle='--', linewidth=1.0,
                label=f'Linea alba y={linea_alba_y}')
    ax1.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    plt.title(stem)
    plt.tight_layout()

    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, stem + '_graph.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def process_image(image_path, linea_alba_threshold=0.9):
    """Process one DICOM image and return a single-row result."""
    filename = os.path.basename(image_path)

    ds = pydicom.dcmread(image_path)
    gray = _extract_pixel_array(ds)
    mask_values, info = generate_shifted_masks(ds)

    linea_alba = extract_linea_alba(mask_values, info, threshold=linea_alba_threshold)
    best_i = linea_alba['mask_index'] - 1  # convert back to 0-based index
    sum_val = float(np.sum(mask_values[best_i]))
    std_val = float(np.std(mask_values[best_i]))

    skin_y = extract_skin_layer(gray, info)
    distance_cm = round((linea_alba['y'] - skin_y) * info['physical_delta_y'], 4)

    rows = [(
        filename,
        linea_alba['x'],
        linea_alba['y'],
        linea_alba['depth_cm'],
        skin_y,
        distance_cm,
        sum_val,
        round(std_val, 4),
    )]
    debug = {'mask_values': mask_values, 'info': info}
    return rows, debug


def _is_dicom(path):
    """Return True if the file has the standard DICOM Part 10 preamble."""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except OSError:
        return False


def find_image_files(directory):
    """Return a sorted list of DICOM files found directly inside directory."""
    image_files = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return []

    for name in entries:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and _is_dicom(path):
            image_files.append(path)

    return sorted(image_files)


def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'linea_alba_x', 'linea_alba_y', 'linea_alba_depth_cm',
                         'skin_y', 'distance_cm', 'sum', 'std'])
        for row in results:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Process sonographic DICOM images to extract subcutaneous adipose tissue measurements.'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing DICOM files to process'
    )
    parser.add_argument(
        '--out-csv',
        default='./output.csv',
        help='Output CSV file path (default: ./output.csv)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save annotated debug images to ./debug-images/'
    )
    parser.add_argument(
        '--linea-alba-threshold',
        type=float,
        default=0.9,
        metavar='RATIO',
        help='Fraction of peak brightness required to detect linea alba (default: 0.9)'
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.out_csv

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = find_image_files(input_dir)

    if not image_files:
        print(f"Warning: No DICOM files found in: {input_dir}", file=sys.stderr)
        print("Creating empty CSV file.")
        write_csv([], output_file)
        return

    print(f"Found {len(image_files)} DICOM file(s) to process")

    all_results = []

    for image_path in image_files:
        try:
            print(f"Processing: {os.path.basename(image_path)}")
            results, debug_data = process_image(
                image_path,
                linea_alba_threshold=args.linea_alba_threshold,
            )
            all_results.extend(results)
            if args.debug:
                row = results[0]
                out_path = save_debug_image(image_path, x=row[1], y=row[2], skin_y=row[4])
                print(f"  Debug image saved: {out_path}")
                graph_path = save_debug_curve(
                    image_path,
                    debug_data['mask_values'],
                    debug_data['info'],
                    linea_alba_y=row[2],
                )
                print(f"  Graph saved: {graph_path}")
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}", file=sys.stderr)
            continue

    write_csv(all_results, output_file)
    print(f"Results written to: {output_file}")
    print(f"Total rows: {len(all_results)}")


if __name__ == '__main__':
    main()
