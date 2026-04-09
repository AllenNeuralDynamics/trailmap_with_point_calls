# aind-capsule-template

# Brain Segmentation & Cell Detection Pipeline

Automated pipeline for whole-brain H2B-GFP cell detection from light-sheet microscopy data stored as OME-Zarr on S3. The pipeline runs UNet-based semantic segmentation followed by 3D point detection and registration to the Allen CCF (Common Coordinate Framework).

## Overview

The main entry point is [`code/run_pipeline.py`](code/run_pipeline.py), which executes a four-step pipeline for each input Zarr volume:

1. **Threshold Estimation** — Loads a low-resolution version of the volume, computes an intensity histogram, and finds the first valley to determine a background threshold.
2. **UNet Segmentation** — Applies a pre-trained Trailmap-style 3D UNet ([`models.get_net`](code/models/model.py)) to the full-resolution volume in a block-wise fashion via [`segment_zarr_volume_blockwise`](code/inference/segment_brain_zarr.py), writing predictions to a local Zarr store.
3. **Point Detection** — Processes the segmentation Zarr in parallel chunks ([`process_large_crop`](code/run_pipeline.py) / [`process_chunk`](code/run_pipeline.py)) using Difference-of-Gaussians filtering, h-maxima suppression, and `peak_local_max` to extract cell centroids.
4. **CCF Registration** — Converts detected centroids from voxel indices to Allen CCF coordinates using [`indices_to_ccf_auto_metadata`](aind-zarr-utils/src/aind_zarr_utils/pipeline_transformed.py) from the `aind-zarr-utils` library.

## Inputs

- **Dispatched task JSONs** — One or more `dispatched_zarr*.json` files placed in `../data/`, each containing a `"fn"` key pointing to an S3 Zarr URI (e.g., `s3://aind-open-data/<asset>/image_tile_fusing/<channel>.zarr`).
- **Pre-trained model weights** — [`data/FOS_model_210815/FOS_model_210815.hdf5`](data/FOS_model_210815)

## Outputs

Results are written to `../results/segmentation_and_quantification/<raw_image_name>/`:

| File | Description |
|------|-------------|
| `segmentation_<mouseID>.zarr` | Full-resolution UNet probability map (Zarr) |
| `points_raw_<mouseID>.npy` | Detected cell centroids in segmentation-level voxel coordinates (N × 3, ZYX) |
| `points_ccf_<mouseID>.npy` | Cell centroids registered to Allen CCFv3 (N × 3, physical LPS coordinates) |

## Pipeline Details

### Threshold Estimation (Step 1)

Reads multiscale level `3` of the input Zarr, downsamples 2× in each axis, then computes a 200-bin histogram. A Gaussian-smoothed derivative is used to find the first valley, which serves as the intensity threshold for segmentation preprocessing.

### UNet Segmentation (Step 2)

- Uses TensorFlow with **mixed-precision (`mixed_float16`)**.
- The model architecture is defined in [`code/models/model.py`](code/models/model.py).
- Block-wise inference is handled by [`segment_zarr_volume_blockwise`](code/inference/segment_brain_zarr.py), processing the volume at multiscale level `1` with `input_dim=64`, `output_dim=36`, and `batch_size=16`.
- Intensity scaling and offset are adjusted based on the estimated threshold.

### Point Detection (Step 3)

[`process_chunk`](code/run_pipeline.py) runs per-block in parallel (via `joblib` with `loky` backend):

1. **Difference of Gaussians (DoG)** — Two Gaussian filters (σ = 1.6/0.7/0.7 and σ = 2.6/1.4/1.4 in Z/Y/X) are subtracted.
2. **Structuring element convolution** — A custom 9×3×3 cross-shaped kernel emphasises axially elongated structures.
3. **H-maxima + peak detection** — `h_maxima(h=0.1)` suppresses noise, then `peak_local_max(min_distance=2)` extracts local maxima.
4. **Core-region filtering** — Only peaks inside the non-halo core are kept, and a final intensity threshold (`> 0.8`) is applied.

Chunks are batched (default 10 000 per batch) and processed with configurable parallelism (`n_workers=8`).

### CCF Registration (Step 4)

Detected centroids are scaled to level-0 voxel indices (`× 2`) and then transformed to Allen CCFv3 coordinates using the `aind-zarr-utils` pipeline transforms, which chain individual and template ANTs registrations.

## Environment

Dependencies are specified in:
- [`requirements.txt`](requirements.txt) — Python packages
- [`environment/Dockerfile`](environment/Dockerfile) — Container build

Key dependencies: TensorFlow/Keras, NumPy, Zarr, SciPy, scikit-image, s3fs, joblib, and `aind-zarr-utils`.

## Running

```bash
# Ensure dispatched task JSON(s) are in ../data/ after previous pipeline capsule 
python code/run_pipeline.py
```