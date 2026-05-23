#!/usr/bin/env python
# coding: utf-8

"""
Process sonographic images to extract subcutaneous adipose tissue measurements.
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np


def create_curve_mask(img_shape, top_curve, thickness, shift_y=0):
    h, w = img_shape

    x_min = int(w * 0.33)
    x_max = int(w * 0.66)

    y_top = np.min(top_curve[:, 1]) + shift_y

    rect = np.array([
        [x_min, y_top],
        [x_max, y_top],
        [x_max, y_top + thickness],
        [x_min, y_top + thickness]
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect], 255)

    return mask, rect


def generate_shifted_masks(image_path, thickness=20, iterations=60, shift_step=5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape

    top_curve = np.array([
        [180, 45],
        [230, 55],
        [280, 65],
        [340, 72],
        [400, 65],
        [460, 55],
        [520, 45]
    ], dtype=np.int32)

    all_mask_values = []

    for i in range(iterations):
        shift_y = i * shift_step

        mask, _ = create_curve_mask(
            img.shape,
            top_curve,
            thickness,
            shift_y
        )

        pixel_values = img[mask == 255]
        all_mask_values.append(pixel_values)

    return all_mask_values


def array_to_dict(array):
    mask_values = {}

    for i, value in enumerate(array):
        mask_values[i + 1] = [np.sum(value), np.std(value)]

    return mask_values


def process_image(image_path):
    filename = os.path.basename(image_path)
    results = []

    mask_values = generate_shifted_masks(image_path)
    stats_dict = array_to_dict(mask_values)

    for mask_index, (sum_val, std_val) in stats_dict.items():
        results.append((filename, mask_index, sum_val, std_val))

    return results


def find_image_files(directory):
    extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    image_files = []

    for ext in extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))

    return sorted(image_files)


def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'mask_index', 'sum', 'std'])

        for row in results:
            writer.writerow([row[0], row[1], row[2], row[3]])


def main():
    parser = argparse.ArgumentParser(
        description='Process sonographic images to extract subcutaneous adipose tissue measurements.'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing sonographic images to process'
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
        print(f"Warning: No image files found in: {input_dir}", file=sys.stderr)
        print("Creating empty CSV file.")
        write_csv([], output_file)
        return

    print(f"Found {len(image_files)} image(s) to process")

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
