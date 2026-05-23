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
import numpy as np
import pydicom


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

    pixel_array = ds.pixel_array

    # Multi-frame: use only the first frame
    n_frames = int(getattr(ds, 'NumberOfFrames', 1))
    if n_frames > 1:
        pixel_array = pixel_array[0]

    # Colour -> grayscale
    samples_per_pixel = int(getattr(ds, 'SamplesPerPixel', 1))
    if samples_per_pixel > 1:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

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


def extract_linea_alba(mask_values, info):
    """
    Identify the most likely position of the linea alba.

    Uses the mask with the highest total pixel brightness as the best estimate.
    The measurement point is the upper edge of that mask.

    Returns a dict with:
      mask_index   -- 1-based index of the brightest mask
      x            -- pixel x coordinate (horizontal centre of the imaging region)
      y            -- pixel y coordinate (upper edge of the brightest mask)
      depth_cm     -- physical depth of the upper edge from the top of the imaging region
    """
    sums = [float(np.sum(v)) for v in mask_values]
    best_i = int(np.argmax(sums))

    x = (info['region_x0'] + info['region_x1']) // 2
    y = info['region_y0'] + best_i * info['shift_step']
    depth_cm = round(best_i * info['shift_step'] * info['physical_delta_y'], 4)

    return {
        'mask_index': best_i + 1,
        'x': x,
        'y': y,
        'depth_cm': depth_cm,
    }


def process_image(image_path):
    """Process one DICOM image and return a single-row result for the linea alba."""
    filename = os.path.basename(image_path)

    ds = pydicom.dcmread(image_path)
    mask_values, info = generate_shifted_masks(ds)

    linea_alba = extract_linea_alba(mask_values, info)
    best_i = linea_alba['mask_index'] - 1  # convert back to 0-based index
    sum_val = float(np.sum(mask_values[best_i]))
    std_val = float(np.std(mask_values[best_i]))

    return [(
        filename,
        linea_alba['x'],
        linea_alba['y'],
        linea_alba['depth_cm'],
        sum_val,
        round(std_val, 4),
    )]


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
                         'sum', 'std'])
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
            results = process_image(image_path)
            all_results.extend(results)
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}", file=sys.stderr)
            continue

    write_csv(all_results, output_file)
    print(f"Results written to: {output_file}")
    print(f"Total rows: {len(all_results)}")


if __name__ == '__main__':
    main()
