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

from sono_satellit.linea_alba import extract_linea_alba
from sono_satellit.lvb import extract_lvb
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


def generate_shifted_masks(ds, thickness=10, shift_step=5, width_fraction=0.10):
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

    region_width = region_x1 - region_x0
    center_x = region_x0 + region_width // 2
    half_width = int(region_width * width_fraction / 2)
    x_min = center_x - half_width
    x_max = center_x + half_width

    for i in range(iterations):
        y_top = region_y0 + i * shift_step

        mask, _ = create_curve_mask(
            pixel_array.shape,
            x_min,
            x_max,
            y_top,
            thickness,
        )

        pixel_values = pixel_array[mask == 255]
        all_mask_values.append(pixel_values)

    return all_mask_values, {
        **region,
        'thickness': thickness,
        'shift_step': shift_step,
        'window_center_x': center_x,
    }


def save_debug_image(image_path, x, y, skin_y=None, lvb_y=None,
                     la_posterior_y=None, debug_dir='./debug-images'):
    """
    Save an annotated copy of a DICOM image with landmarks marked.

    Draws a red circle + crosshair at (x, y) for the linea alba anterior border.
    Draws an orange circle + crosshair at (x, la_posterior_y) for the posterior border.
    Optionally draws a green circle + crosshair at (x, skin_y) for the skin layer.
    Optionally draws a blue circle + crosshair at (x, lvb_y) for the LVB.
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

    # Red crosshair + circle for linea alba anterior border
    color = (0, 0, 255)
    cv2.circle(bgr, (x, y), radius, color, thickness)
    cv2.line(bgr, (x - arm, y), (x + arm, y), color, thickness)
    cv2.line(bgr, (x, y - arm), (x, y + arm), color, thickness)

    # Orange crosshair + circle for linea alba posterior border
    if la_posterior_y is not None:
        color = (0, 165, 255)
        cv2.circle(bgr, (x, la_posterior_y), radius, color, thickness)
        cv2.line(bgr, (x - arm, la_posterior_y), (x + arm, la_posterior_y), color, thickness)
        cv2.line(bgr, (x, la_posterior_y - arm), (x, la_posterior_y + arm), color, thickness)

    # Blue crosshair + circle for LVB
    if lvb_y is not None:
        color = (255, 0, 0)
        cv2.circle(bgr, (x, lvb_y), radius, color, thickness)
        cv2.line(bgr, (x - arm, lvb_y), (x + arm, lvb_y), color, thickness)
        cv2.line(bgr, (x, lvb_y - arm), (x, lvb_y + arm), color, thickness)

    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(debug_dir, stem + '.png')
    cv2.imwrite(out_path, bgr)
    return out_path


def save_debug_curve(image_path, mask_values, info, linea_alba_y, lvb_y=None, skin_y=None,
                     la_posterior_y=None, debug_dir='./debug-images'):
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

    if skin_y is not None:
        ax1.axvline(x=skin_y, color='green', linestyle='--', linewidth=1.0,
                    label=f'Skin y={skin_y}')
    ax1.axvline(x=linea_alba_y, color='red', linestyle='--', linewidth=1.0,
                label=f'LA anterior y={linea_alba_y}')
    if la_posterior_y is not None:
        ax1.axvline(x=la_posterior_y, color='orange', linestyle='--', linewidth=1.0,
                    label=f'LA posterior y={la_posterior_y}')
    if lvb_y is not None:
        ax1.axvline(x=lvb_y, color='blue', linestyle='--', linewidth=1.0,
                    label=f'LVB y={lvb_y}')
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


def process_image(image_path, requested_type=None):
    """Process one DICOM image and return a single-row result.

    requested_type: 'linea-alba', 'lvb', or None for auto-detection.
    Auto-detection uses image depth: >=20 cm -> 'lvb', else -> 'linea-alba'.
    """
    filename = os.path.basename(image_path)

    ds = pydicom.dcmread(image_path)
    gray = _extract_pixel_array(ds)
    mask_values, info = generate_shifted_masks(ds)
    skin_mask_values, _ = generate_shifted_masks(ds, width_fraction=0.05)

    if requested_type:
        image_type = requested_type
    else:
        image_depth_cm = (info['region_y1'] - info['region_y0']) * info['physical_delta_y']
        image_type = 'lvb' if image_depth_cm >= 20 else 'linea-alba'
        print(f"  Auto-detected type: {image_type} (image depth {image_depth_cm:.1f} cm)")

    skin_y = extract_skin_layer(skin_mask_values, info)
    linea_alba = extract_linea_alba(mask_values, info, skin_y=skin_y)
    la_anterior = linea_alba['anterior']
    la_posterior = linea_alba['posterior']
    best_i = la_anterior['mask_index'] - 1
    sum_val = float(np.sum(mask_values[best_i]))
    std_val = float(np.std(mask_values[best_i]))
    distance_cm = round((la_anterior['y'] - skin_y) * info['physical_delta_y'], 4)

    if image_type == 'lvb':
        lvb = extract_lvb(mask_values, info)
        if lvb is not None:
            lvb_to_la_distance_cm = round(
                (lvb['y'] - la_posterior['y']) * info['physical_delta_y'], 4
            )
            lvb_y_out = lvb['y']
            lvb_depth_out = lvb['depth_cm']
        else:
            lvb_to_la_distance_cm = ''
            lvb_y_out = ''
            lvb_depth_out = ''
        rows = [(
            filename,
            'lvb',
            la_anterior['x'],
            la_anterior['y'],
            la_anterior['depth_cm'],
            la_posterior['y'],
            la_posterior['depth_cm'],
            skin_y,
            distance_cm,
            lvb_y_out,
            lvb_depth_out,
            lvb_to_la_distance_cm,
            sum_val,
            round(std_val, 4),
        )]
    else:
        rows = [(
            filename,
            'linea-alba',
            la_anterior['x'],
            la_anterior['y'],
            la_anterior['depth_cm'],
            la_posterior['y'],
            la_posterior['depth_cm'],
            skin_y,
            distance_cm,
            '',
            '',
            '',
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
        writer.writerow([
            'filename', 'measurement',
            'linea_alba_x',
            'linea_alba_anterior_y', 'linea_alba_anterior_depth_cm',
            'linea_alba_posterior_y', 'linea_alba_posterior_depth_cm',
            'skin_y', 'distance_cm',
            'lvb_y', 'lvb_depth_cm', 'lvb_to_la_distance_cm',
            'sum', 'std',
        ])
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
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        '--linea-alba',
        action='store_const',
        const='linea-alba',
        dest='image_type',
        help='Process all images as linea alba type (scans shallow to deep)'
    )
    type_group.add_argument(
        '--lvb',
        action='store_const',
        const='lvb',
        dest='image_type',
        help='Process all images as lumbar vertebral body type (scans deep to shallow)'
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
                requested_type=args.image_type,
            )
            all_results.extend(results)
            if args.debug:
                row = results[0]
                lvb_y_val = row[9] if row[9] != '' else None
                out_path = save_debug_image(
                    image_path, x=row[2], y=row[3], skin_y=row[7],
                    lvb_y=lvb_y_val, la_posterior_y=row[5],
                )
                print(f"  Debug image saved: {out_path}")
                graph_path = save_debug_curve(
                    image_path,
                    debug_data['mask_values'],
                    debug_data['info'],
                    linea_alba_y=row[3],
                    lvb_y=row[9] if row[9] != '' else None,
                    skin_y=row[7] if row[7] != '' else None,
                    la_posterior_y=row[5],
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
