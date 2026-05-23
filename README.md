## About

The Sono SATELLIT (Subcutaneous Adipose Tissue Extraction & Layer Localization In Tomography) scans abdominal sonographic DICOM images and measures the thickness of the subcutaneous adipose tissue.

It reads physical pixel spacing directly from DICOM metadata (`SequenceOfUltrasoundRegions`, tag `0018,6011`) and sweeps a series of horizontal rectangular masks across the full imaging depth, recording pixel intensity statistics at each depth level.

The project was created during the Healthcare Hackathon Regensburg 2026.

## Requirements

- Python >= 3.10
- Dependencies (installed automatically via pip): `opencv-python`, `numpy`, `pydicom`, `matplotlib`


## Setup

```sh
pip install -e .
```


## Usage

```sh
python -m sono_satellit --input-dir <path/to/dicom/files> [--out-csv output.csv] [--debug] [--linea-alba-threshold 0.9]
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input-dir` | yes | Directory containing DICOM files to process |
| `--out-csv` | no | Output CSV path (default: `./output.csv`) |
| `--debug` | no | Save annotated images and brightness-curve graphs to `./debug-images/` |
| `--linea-alba-threshold` | no | Fraction of peak brightness required to select the linea alba (default: `0.9`) |

DICOM files are detected automatically by their file signature — no specific extension is required (files named `IM_1232`, `*.dcm`, etc. all work).


## Output

A CSV file with **one row per image** containing the detected linea alba position:

| Column | Description |
|---|---|
| `filename` | Source DICOM filename |
| `linea_alba_x` | Pixel x coordinate of the measurement point (horizontal centre of the imaging region) |
| `linea_alba_y` | Pixel y coordinate of the measurement point (upper edge of the brightest mask strip) |
| `linea_alba_depth_cm` | Physical depth of the measurement point from the top of the imaging region (cm) |
| `sum` | Total grayscale brightness of the detected mask strip |
| `std` | Standard deviation of pixel intensities in the detected mask strip |


## How it works

1. Each DICOM file is opened with `pydicom` and the imaging region bounds and physical scale (`PhysicalDeltaY`, cm/pixel) are read from `(0018,6011) SequenceOfUltrasoundRegions`.
2. A 20-pixel-tall rectangular mask is placed at the top of the imaging region and shifted downward by 5 pixels at each step until the bottom of the region is reached. The mask spans the full horizontal width of the imaging region.
3. The mask with the highest total pixel brightness is selected as the most likely location of the linea alba.
4. The upper edge of that mask is reported as the measurement point, converted to physical depth in cm using the DICOM scale factor.


## Known limitations

- **Burned-in annotation crosses**: clinician-placed markers (skin edge, linea alba) are rendered directly into the DICOM pixel data, despite the `BurnedInAnnotation` DICOM tag reading `NO`. There is no separate clean layer. When converted to grayscale the markers (RGB ≈ 61, 142, 84; ~111 in grayscale) fall within the normal tissue intensity range and occupy roughly 484 out of ~600 000 pixels in the imaging region. The resulting error in brightness measurements is considered minor and is currently not corrected.
- **Single imaging region assumed**: only the first entry of `SequenceOfUltrasoundRegions` is used. Images with multiple regions (e.g. split-screen or Doppler overlays) are not supported.
- **No skin edge detection**: the skin surface position cannot be read from DICOM metadata and is not computed from the image. The depth values reported are relative to the top of the transducer imaging region, not the skin surface.


## License

This project is [Apache 2.0](LICENSE) licensed.
