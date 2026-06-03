## About

The Sono SATELLIT (Subcutaneous Adipose Tissue Extraction & Layer Localization In Tomography) scans abdominal sonographic DICOM images and measures the thickness of the subcutaneous adipose tissue.

It reads physical pixel spacing directly from DICOM metadata (`SequenceOfUltrasoundRegions`, tag `0018,6011`) and sweeps a series of horizontal rectangular masks across the full imaging depth, recording pixel intensity statistics at each depth level.

Two measurement modes are supported:

- **Linea alba** — shallow scans; detects the skin surface and linea alba, reports skin-to-linea-alba distance.
- **LVB (Lumbar Vertebral Body)** — deep scans; detects the skin surface, linea alba, and LVB, reports skin-to-linea-alba distance and LVB-to-linea-alba distance.

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
python -m sono_satellit --input-dir <path/to/dicom/files> [--out-csv output.csv] [--debug] [--linea-alba-threshold 0.9] [--linea-alba | --lvb]
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input-dir` | yes | Directory containing DICOM files to process |
| `--out-csv` | no | Output CSV path (default: `./output.csv`) |
| `--debug` | no | Save annotated images and brightness-curve graphs to `./debug-images/` |
| `--linea-alba-threshold` | no | Fraction of peak brightness required to select the target structure (default: `0.9`) |
| `--linea-alba` | no | Force linea-alba mode for all images |
| `--lvb` | no | Force LVB mode for all images |

If neither `--linea-alba` nor `--lvb` is given, the mode is auto-detected per image based on imaging depth: ≥ 20 cm → LVB, otherwise → linea alba.

DICOM files are detected automatically by their file signature — no specific extension is required (files named `IM_1232`, `*.dcm`, etc. all work).


## Output

A CSV file with **one row per image**:

| Column | Description |
|---|---|
| `filename` | Source DICOM filename |
| `measurement` | Measurement mode: `linea-alba` or `lvb` |
| `linea_alba_x` | Pixel x coordinate of the linea alba (horizontal centre of the imaging region) |
| `linea_alba_y` | Pixel y coordinate of the linea alba (upper edge of the detected mask strip) |
| `linea_alba_depth_cm` | Physical depth of the linea alba from the top of the imaging region (cm) |
| `skin_y` | Pixel y coordinate of the detected skin surface |
| `distance_cm` | Distance from skin surface to linea alba (cm) |
| `lvb_y` | Pixel y coordinate of the LVB — populated for LVB images only |
| `lvb_depth_cm` | Physical depth of the LVB from the top of the imaging region (cm) — LVB images only |
| `lvb_to_la_distance_cm` | Distance from LVB to linea alba (cm) — LVB images only |
| `sum` | Total grayscale brightness of the linea alba mask strip |
| `std` | Standard deviation of pixel intensities in the linea alba mask strip |


## How it works

1. Each DICOM file is opened with `pydicom` and the imaging region bounds and physical scale (`PhysicalDeltaY`, cm/pixel) are read from `(0018,6011) SequenceOfUltrasoundRegions`.
2. A 20-pixel-tall rectangular mask is placed at the top of the imaging region and shifted downward by 5 pixels at each step until the bottom of the region is reached. The mask spans the full horizontal width of the imaging region.
3. **Skin detection**: Canny edge detection is applied to the upper portion of the image to locate the skin surface.
4. **Linea alba detection**: scanning shallow to deep, the first mask whose brightness reaches the threshold fraction of the global maximum is selected.
5. **LVB detection** (LVB mode only): scanning deep to shallow, the bright LVB reflection is identified using the same threshold logic applied in reverse.
6. Distances are computed in physical units (cm) using the DICOM scale factor.

In debug mode (`--debug`), annotated images are saved to `./debug-images/` with colour-coded markers: green = skin, red = linea alba, blue = LVB.


## Known limitations

- **Burned-in annotation crosses**: clinician-placed markers are rendered directly into the DICOM pixel data despite the `BurnedInAnnotation` tag reading `NO`. The resulting error in brightness measurements is considered minor and is currently not corrected.
- **Single imaging region assumed**: only the first entry of `SequenceOfUltrasoundRegions` is used. Images with multiple regions (e.g. split-screen or Doppler overlays) are not supported.


## License

This project is [Apache 2.0](LICENSE) licensed.
