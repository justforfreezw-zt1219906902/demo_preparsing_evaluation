# OpenScan Evaluation Toolkit

This project prepares a flat OpenScan image dataset for an **external**
PyTorch3D reconstruction project and compares an externally reconstructed mesh
with a reference STL. It does not implement reconstruction, differentiable
rendering, mesh deformation, pose refinement, SfM/MVS, COLMAP, or Metashape.

## Configuration

[`configs/default.yaml`](configs/default.yaml) is the single authoritative
default configuration. The CLI always loads it first. `--config FILE` then
recursively overrides only the supplied values.

The project automatically reads `.env` without replacing environment variables
that are already set:

```dotenv
OPENSCAN_DATASET_DIR=/path/to/dataset
OPENSCAN_U2NET_MODEL=/path/to/U2Net_v1.onnx
OPENSCAN_REFERENCE_MESH=/path/to/reference.stl
OPENSCAN_RECONSTRUCTION_MESH=/path/to/reconstructed.obj
```

The ONNX model is local runtime data and must not be committed.

## Dataset

```text
dataset/
├── positions.csv
├── default_0_1.jpg
└── ...
```

The CSV requires `image`, `position_index`, `phi_deg`, and `theta_deg`.
Commanded angles are initialization metadata, not calibrated camera poses.

## Crop semantics

```yaml
crop:
  enabled: true
  mode: manual              # manual | auto_from_masks
  roi_xyxy: [0.25, 0.18, 0.72, 0.88]
  margin_ratio: 0.08
  resize:
    enabled: true
    output_size: [1200, 1000]
```

`roi_xyxy` is `[x0, y0, x1, y1]` in normalized coordinates of the original
image. A manual crop is applied before U2Net inference and visibly affects RGBA
and previews.

`--full-resolution` means **apply the crop but do not resize the cropped
result**. It does not disable cropping.

## Segmentation and output

U2Net ONNX is the default segmentation backend. The adapter discovers tensor
names and shapes from the session, respects fixed input dimensions, reuses one
session, and restores the soft prediction to crop dimensions. Optional modes
are `background_subtraction` and `external`.

Default processed output is deliberately small:

```text
output/processed/
├── rgba/
├── previews/
└── crop_transforms.json
```

RGBA is the reconstruction handoff: processed object RGB plus soft segmentation
alpha. Set `debug.save_masks: true` to additionally write `processed/masks/`.
It is false by default. Preview panels show cropped RGB, segmentation overlay,
RGBA on a checkerboard, pose, sharpness, filename, and status.

Quality metrics are evaluated on the segmented object or its mask bounding box,
not a fixed wall-dominated rectangle. CLAHE, highlight suppression, sharpening,
and brightness normalization remain independent switches.

## Commands

```bash
python -m pip install -e ".[dev]"

# Entire available flow; crop is resized according to YAML
.venv/bin/python main.py all

# Entire flow; crop is retained at its native cropped resolution
.venv/bin/python main.py all --full-resolution

.venv/bin/python main.py validate
.venv/bin/python main.py quality
.venv/bin/python main.py preprocess --full-resolution
.venv/bin/python main.py export --full-resolution
.venv/bin/python main.py compare
.venv/bin/python main.py report
```

The PyTorch3D handoff contains only `rgba/`, `metadata.json`, and
`dataset_manifest.csv`; it does not duplicate RGB, masks, or edges.

Mesh comparison does not optimize scale. Optional ICP is rigid only and is off
by default. Metrics remain mean surface distance, P95 surface distance, and
symmetric sampled Chamfer distance.

> If the external workflow initializes from the reference STL, reported mesh
> deviations are not an independent measurement of absolute accuracy.
