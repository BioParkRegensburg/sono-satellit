## About

The Sono SATELLIT (Subcutaneous Adipose Tissue Extraction & Layer Localization In Tomography) scans abdominal sonographic DICOM images and measures the thickness of the subcutaneous adipose tissue.

It reads physical pixel spacing directly from DICOM metadata (`SequenceOfUltrasoundRegions`, tag `0018,6011`) and sweeps a series of horizontal rectangular masks across the full imaging depth, recording pixel intensity statistics at each depth level.


## Requirements

- Python >= 3.10
- Dependencies (installed automatically via pip): `opencv-python`, `numpy`, `pydicom`, `matplotlib`


## Setup

```sh
pip install -e .
```


## Usage

```sh
python -m sono_satellit --input-dir <path/to/dicom/files> [--out-csv output.csv]
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input-dir` | yes | Directory containing DICOM files to process |
| `--out-csv` | no | Output CSV path (default: `./output.csv`) |

DICOM files are detected automatically by their file signature — no specific extension is required (files named `IM_1232`, `*.dcm`, etc. all work).


## Output

A CSV file with one row per mask position per image:

| Column | Description |
|---|---|
| `filename` | Source DICOM filename |
| `mask_index` | 1-based index of the mask strip (top to bottom) |
| `sum` | Sum of grayscale pixel values inside the mask |
| `std` | Standard deviation of grayscale pixel values inside the mask |
| `depth_cm` | Physical depth of the mask centre from the top of the imaging region (cm) |

The number of rows per image is determined automatically from the imaging region height and a fixed step size of 5 pixels (e.g. a 660 px deep region produces 132 rows covering ~16.6 cm).


## How it works

1. Each DICOM file is opened with `pydicom` and the imaging region bounds and physical scale (`PhysicalDeltaY`, cm/pixel) are read from `(0018,6011) SequenceOfUltrasoundRegions`.
2. A 20-pixel-tall rectangular mask is placed at the top of the imaging region and shifted downward by 5 pixels at each step until the bottom of the region is reached.
3. The mask spans the full horizontal width of the imaging region.
4. For each mask position the grayscale pixel values are extracted and their sum and standard deviation are recorded alongside the physical depth in cm.


## License

This project is [Apache 2.0](LICENSE) licensed.
