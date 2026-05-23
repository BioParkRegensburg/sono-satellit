# AGENTS.md

## Package structure

- `sono_satellit/__init__.py` — entire implementation; one file, no submodules
- `satellit/` — empty directory, not part of the installed package
- `extract_rectangles_sonography_copy.py` / `.ipynb` — prototype/exploration files, not installed

## Setup

```sh
pip install -e .
```

`.venv/` exists in the repo root; activate it before running commands.

## Running the CLI

```sh
python -m sono_satellit --input-dir <path/to/images> [--out-csv output.csv]
```

Accepts `.png`, `.PNG`, `.jpg`, `.JPG`, `.jpeg`, `.JPEG`. Outputs a CSV with columns: `filename`, `mask_index`, `sum`, `std`.

## Hardcoded geometry

`generate_shifted_masks()` in `__init__.py` uses a hardcoded `top_curve` array tuned to images roughly 700×500 px. Changing image dimensions requires updating those coordinates manually — there is no auto-detection.

## No test suite

No tests are configured. Verify changes by running the CLI against sample images.

## Dependencies

`opencv-python`, `numpy`, `matplotlib` (declared in `pyproject.toml`). Python >= 3.10 required.
