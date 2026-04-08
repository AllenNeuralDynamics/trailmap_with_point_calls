import os
os.environ.update({
  "OMP_NUM_THREADS":"1",
  "OPENBLAS_NUM_THREADS":"1",
  "MKL_NUM_THREADS":"1",
  "NUMEXPR_NUM_THREADS":"1",
})
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import gc
import time
import numpy as np
import zarr
import faulthandler
faulthandler.enable()

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed

strel = np.zeros((9,3,3), dtype=np.uint8)
for z in range(9):
    strel[z,1,1] = 1
    strel[z,0,1] = 1
    strel[z,2,1] = 1
    strel[z,1,0] = 1
    strel[z,1,2] = 1
strel[4,1,1] = 10

def process_chunk(task):
    z0,z1,y0,y1,x0,x1, output_zarr_path, halo = task

    from scipy.ndimage import gaussian_filter
    from scipy import ndimage
    from scipy.signal import convolve
    from skimage.morphology import h_maxima
    from skimage.feature import peak_local_max
    import numpy as np
    import zarr

    arr = zarr.open(output_zarr_path, mode='r')
    eZ0, eZ1 = max(0, z0 - halo), min(arr.shape[0], z1 + halo)
    eY0, eY1 = max(0, y0 - halo), min(arr.shape[1], y1 + halo)
    eX0, eX1 = max(0, x0 - halo), min(arr.shape[2], x1 + halo)

    block_ext = arr[eZ0:eZ1, eY0:eY1, eX0:eX1]
    block_ext = np.ascontiguousarray(block_ext, dtype=np.float32)

    if not np.any(block_ext >= 0.8):
        return np.zeros((0, 3), dtype=int)
    
    if np.isnan(block_ext).any():
        print(f"Percent NaN in block z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]: {np.isnan(block_ext).mean()*100:.2f}%")
        block_ext = np.nan_to_num(block_ext, nan=0.0)

    ga1 = gaussian_filter(block_ext, sigma=(1.6, 0.7, 0.7))
    ga2 = gaussian_filter(block_ext, sigma=(2.6, 1.4, 1.4))
    dog = ga1 - ga2

    dog_conv_ext = convolve(dog, strel, mode='same')

    filtered = h_maxima(dog_conv_ext, h=0.1)
    coords_ext = peak_local_max(dog_conv_ext, min_distance=2, labels=filtered)

    if coords_ext.size == 0:
        return np.zeros((0,3), dtype=int)

    core_min = np.array([z0 - eZ0, y0 - eY0, x0 - eX0])
    core_max = np.array([z1 - eZ0, y1 - eY0, x1 - eX0])
    mask_core = np.all((coords_ext >= core_min) & (coords_ext < core_max), axis=1)
    coords_core = coords_ext[mask_core]

    if coords_core.size == 0:
        return np.zeros((0,3), dtype=int)

    vals = block_ext[coords_core[:,0], coords_core[:,1], coords_core[:,2]]
    coords_core = coords_core[vals > 0.8]

    return coords_core + np.array([eZ0, eY0, eX0])

def process_large_crop(output_zarr_path, z_range, y_range, x_range,
                       core_size=(64,128,128), halo=8, n_workers=4, batch_size=10000):

    def chunk_ranges(start, stop, chunk):
        return [(i, min(i + chunk, stop)) for i in range(start, stop, chunk)]

    z_chunks = chunk_ranges(*z_range, core_size[0])
    y_chunks = chunk_ranges(*y_range, core_size[1])
    x_chunks = chunk_ranges(*x_range, core_size[2])
    all_chunks = [(z,y,x) for z in z_chunks for y in y_chunks for x in x_chunks]
    print(f"Total chunks to process: {len(all_chunks)}")

    tasks = []
    for (z0,z1),(y0,y1),(x0,x1) in all_chunks:
        tasks.append((z0,z1,y0,y1,x0,x1, output_zarr_path, halo))

    all_results = []

    # use joblib Parallel with loky (cloudpickle) so notebook functions are serializable
    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start:batch_start+batch_size]
        try:
            results = Parallel(n_jobs=n_workers, backend='loky')(
                delayed(process_chunk)(t) for t in batch_tasks
            )
        except Exception as e:
            print("Batch failed:", e)
            results = []

        batch_results = [r for r in results if (r is not None and getattr(r, "size", 0) > 0)]
        print(f"Batch number {batch_start // batch_size + 1} complete. Found {sum(len(r) for r in batch_results)} points in this batch.")
        for r in batch_results:
            all_results.extend(r)
        gc.collect()

    if not all_results:
        return np.empty((0,3), dtype=int)
    return np.vstack(all_results)

# ---------------------------------------------------------
# Main worker
# ---------------------------------------------------------

def run():
    import time
    import os
    import numpy as np
    import zarr
    import gc
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import s3fs
    import json

    print("Worker started. Reading dispatcher task list...")

    # Read all dispatcher JSON files
    task_dir = "../data/"
    task_files = sorted(f for f in os.listdir(task_dir) 
        if f.endswith(".json") 
        and f.startswith("dispatched_zarr"))

    zarr_file_list = []
    for tf in task_files:
        print(f"task_file: {tf}")
        with open(os.path.join(task_dir, tf)) as f:
            obj = json.load(f)
            zarr_file_list.append(obj["fn"])

    print(f"Found {len(zarr_file_list)} tasks.")

    fs = s3fs.S3FileSystem(anon=False)

    for zarr_file in zarr_file_list:
        start = time.time()
        print(f"\n=== Processing {zarr_file} ===")

        raw_image_name = zarr_file.split("data/")[1].split("/image_tile_fusing/")[0]
        mouse_ID = raw_image_name.split("_")[1]

        output_dir = f"../results/segmentation_and_quantification/{raw_image_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # -------------------------------------------------
        # Step 1: load volume and build threshold
        # -------------------------------------------------
        store = s3fs.S3Map(root=zarr_file, s3=fs, check=False)
        zarr_vol = zarr.open(store, mode='r')
        arr = zarr_vol["3"]

        data = arr[::2, ::2, ::2]
        data_filtered = data[data < (data.max() / 10)]
        del data

        counts, bins = np.histogram(data_filtered, bins=200)
        smooth = gaussian_filter1d(counts, 3)
        valley_idx, _ = find_peaks(-smooth)
        valley_pos = bins[valley_idx[0]]

        print(f"Threshold: {valley_pos}")

        # -------------------------------------------------
        # Step 2: segmentation using Trailmap model
        # -------------------------------------------------
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        from models import get_net
        from inference.segment_brain_zarr import segment_zarr_volume_blockwise

        model = get_net()
        model.load_weights("../data/FOS_model_210815/FOS_model_210815.hdf5")

        output_zarr_path = os.path.join(output_dir, f"segmentation_{mouse_ID}.zarr")
        intensity_scale_factor = 3.5 if valley_pos < 100 else 10.0
        intensity_offset = 75 if valley_pos < 100 else 0

        segment_zarr_volume_blockwise(
            zarr_file, output_zarr_path, model,
            input_dim=64, output_dim=36, batch_size=16, block_size=512,
            level=1, threshold=valley_pos, z_step_reduction=6,
            intensity_scale_factor=intensity_scale_factor,
            intensity_offset=intensity_offset
        )

        # -------------------------------------------------
        # Step 3: point detection
        # -------------------------------------------------
        seg_zarr = zarr.open(output_zarr_path, mode='r')
        DV, AP, ML = [0, seg_zarr.shape[0]], [0, seg_zarr.shape[1]], [0, seg_zarr.shape[2]]

        centroids = process_large_crop(
            output_zarr_path, DV, AP, ML,
            core_size=(32, 64, 64), halo=8, n_workers=8
        )

        print(f"Centroids found: {len(centroids)}")

        np.save(os.path.join(output_dir, f"points_raw_{mouse_ID}.npy"), centroids)

        # -------------------------------------------------
        # Step 4: convert to CCF coordinates
        # -------------------------------------------------
        from aind_zarr_utils import indices_to_ccf_auto_metadata
        
        centroids_level0 = (centroids * 2).astype(int)
        ccf = indices_to_ccf_auto_metadata({mouse_ID: centroids_level0}, zarr_file)
        np.save(os.path.join(output_dir, f"points_ccf_{mouse_ID}.npy"), ccf[mouse_ID])

        # -------------------------------------------------
        # Done
        # -------------------------------------------------
        t = time.time() - start
        print(f"Done {mouse_ID}: {int(t//3600)}h {int((t%3600)//60)}m")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # ensure spawn

    import faulthandler
    faulthandler.enable()

    # Run main pipeline
    run()
