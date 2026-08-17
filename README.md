# OpenScan Evaluation Toolkit

This standalone project prepares an OpenScan image dataset for an external
PyTorch3D reconstruction pipeline and, in later phases, evaluates reconstructed
meshes against a reference STL.

It deliberately does **not** perform reconstruction, differentiable rendering,
mesh optimization, pose refinement, SfM/MVS, COLMAP, or Metashape processing.

## Implemented pipeline

The toolkit validates the flat dataset, classifies image quality, creates
object masks and consistent crops, applies optional preprocessing, converts
commanded OpenScan angles, exports a PyTorch3D-ready dataset, and compares an
external reconstructed mesh with the configured reference STL.

## Dataset layout

```text
dataset/
├── default_0_1.jpg
├── default_0_2.jpg
└── positions.csv
```

The CSV must contain `image`, `position_index`, `phi_deg`, and `theta_deg`.
Commanded angles are initialization metadata, not calibrated ground-truth poses.

## Install and validate

```bash
python -m pip install -e ".[dev]"
export OPENSCAN_DATASET_DIR=/path/to/dataset
openscan-eval validate
```

You can instead set `OPENSCAN_DATASET_DIR` and `OPENSCAN_REFERENCE_MESH` in the
project `.env` file. It is loaded automatically and is excluded from Git.

Run every available stage with:

```bash
openscan-eval all
```

From a source checkout, the explicit program entry point is also available:

```bash
# Full pipeline using reconstruction assets at the original image resolution
.venv/bin/python main.py all --full-resolution

# Run individual stages
.venv/bin/python main.py validate
.venv/bin/python main.py quality
.venv/bin/python main.py preprocess --full-resolution
.venv/bin/python main.py export --full-resolution
.venv/bin/python main.py compare
.venv/bin/python main.py report
```

Without `--full-resolution`, reconstruction assets use the configured crop and
`crop.output_size` (1200×900 by default). With `--full-resolution`, cropping is
disabled and RGB, mask, RGBA, and edge files retain the complete source canvas
and exact source dimensions. Mask analysis may still use a reduced working scale
before being mapped back to the source pixels.

Set `OPENSCAN_RECONSTRUCTION_MESH` in `.env` after the external reconstruction
pipeline produces a mesh. Until then, `all` skips mesh comparison. Individual
commands are `validate`, `quality`, `preprocess`, `export`, `compare`, and
`report`.

Generated files go to `output/`, never into the source dataset. They include
`quality_report.csv`, processed RGB/masks/RGBA/edges/previews, crop transforms,
PyTorch3D metadata and manifest, and (when a reconstruction exists) mesh metrics,
overlays, heatmap, histogram, and HTML report.

Automatic masks intentionally avoid a mandatory neural model. With reflective
metal touching a blue turntable, inspect `output/processed/previews`; for exact
silhouettes supply single-channel masks under `DATASET/masks/` and set
`mask.mode: external`.

Validation prints its report to the terminal and exits unsuccessfully when
errors are present. It does not save an experiment history or run record.
Original images are only read and are never modified.

Configuration defaults live in `configs/default.yaml`; pass another file with
`--config`. The reference and reconstructed mesh paths used by the later
comparison stage are also configured there. To override the environment for one
command, use `openscan-eval validate --dataset /path/to/dataset`.

> This repository evaluates the effect of imaging and preprocessing conditions
> on an external reconstruction pipeline. Because the external PyTorch3D
> workflow may use the reference STL as its initialization geometry, the
> reported mesh deviations should not be interpreted as an independent
> measurement of absolute reconstruction accuracy.
