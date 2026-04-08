import zarr
import numpy as np
from tqdm import tqdm
import time
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf 
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- PATCH EXTRACTION (modified to accept z_step_reduction and intensity_scale_factor) ---
def _get_patch_dataset(block, input_dim, output_dim, normalizer, batch_size, threshold=None, z_step_reduction=0, intensity_scale_factor=10, intensity_offset=0):
    """
    block : 3D numpy array (bz, by, bx)
    z_step_reduction : int >=0. number of voxels to reduce the z step by.
                        Effective z step = max(1, output_dim - z_step_reduction)
    intensity_scale_factor : float. Value to divide the signal by before normalization.
    """
    dim_offset = (input_dim - output_dim) // 2
    bz, by, bx = block.shape

    # 1. Pre-pad the block
    pad_width = ((dim_offset, dim_offset), (dim_offset, dim_offset), (dim_offset, dim_offset))
    padded = np.pad(block, pad_width, mode='reflect').astype(np.float32)

    # 3. Compute stepping positions
    # only z step is reduced; y/x use output_dim (no change)
    z_step = max(1, output_dim - int(z_step_reduction))
    z_positions = list(range(0, bz, z_step))
    y_positions = list(range(0, by, output_dim))
    x_positions = list(range(0, bx, output_dim))

    # 4. Add extra positions aligned to far edges (same logic as before)
    # adjust last z position for aligning with far edge
    max_start = bz - output_dim
    if max_start < 0:
        raise ValueError("block smaller than output_dim!")

    # clamp to legal positions only
    z_positions = [z for z in z_positions if z <= max_start]

    # ensure last patch always aligns with far edge
    if len(z_positions) == 0 or z_positions[-1] < max_start:
        z_positions.append(max_start)

    if len(y_positions) == 0 or y_positions[-1] + output_dim > by:
        if by - output_dim >= 0:
            if len(y_positions) == 0:
                y_positions = [by - output_dim]
            else:
                y_positions[-1] = by - output_dim
    if len(x_positions) == 0 or x_positions[-1] + output_dim > bx:
        if bx - output_dim >= 0:
            if len(x_positions) == 0:
                x_positions = [bx - output_dim]
            else:
                x_positions[-1] = bx - output_dim

    # 5. Extract patches and record coordinates
    patches = []
    patch_coords = []
    for iz in z_positions:
        for iy in y_positions:
            for ix in x_positions:
                patches.append(padded[iz : iz + input_dim, iy : iy + input_dim,  ix : ix + input_dim])
                patch_coords.append((iz, iy, ix))

    patches = np.array(patches, dtype=np.float32)
    num_patches = len(patches)

    # --- Stage 1: CPU pre-pass for thresholding ---
    if threshold is not None:
        patch_maxes = patches.max(axis=(1,2,3))
        interesting = patch_maxes >= threshold
        indices = np.where(interesting)[0]
        patches = patches[interesting]
        patch_coords = [patch_coords[i] for i in indices]
    else:
        indices = np.arange(len(patches))

    # --- Create tf.data.Dataset ---
    dataset = tf.data.Dataset.from_tensor_slices(patches)
    def preprocess(patch):
        patch = tf.expand_dims(patch / intensity_scale_factor + intensity_offset, axis=-1) # divide signal to match model training range
        patch = patch / float(normalizer)
        patch = tf.clip_by_value(patch, 0.0, 1.0)
        return patch

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_patches, patch_coords, indices


# --- ASYNC WRITE FUNCTION ---
def hard_seam_write(arr_out, out_block, z0, z1, y0, y1, x0, x1, dim_offset):
    bz, by, bx = out_block.shape

    # Half of the overlap region along each axis
    z_half = min(dim_offset // 2, bz // 2)
    y_half = min(dim_offset // 2, by // 2)
    x_half = min(dim_offset // 2, bx // 2)

    # Determine which part of the block to actually write
    z_start = z_half if z0 > 0 else 0  # skip first half if not at z=0
    y_start = y_half if y0 > 0 else 0
    x_start = x_half if x0 > 0 else 0

    z_end = bz - z_half if z1 < arr_out.shape[0] else bz  # skip last half if not at end
    y_end = by - y_half if y1 < arr_out.shape[1] else by
    x_end = bx - x_half if x1 < arr_out.shape[2] else bx

    # Copy the selected region to the output volume
    arr_out[z0+z_start:z0+z_end,
            y0+y_start:y0+y_end,
            x0+x_start:x0+x_end] = \
        out_block[z_start:z_end,
                  y_start:y_end,
                  x_start:x_end]

from concurrent.futures import ThreadPoolExecutor

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def read_zarr_block_threaded(arr_in, z0, z1, y0, y1, x0, x1, num_strips=16):
    Z = z1 - z0
    strip_size = max(1, Z // num_strips)

    # allocate output
    block = np.zeros((Z, y1-y0, x1-x0), dtype=arr_in.dtype)

    tasks = []
    with ThreadPoolExecutor(max_workers=num_strips) as ex:
        for i in range(num_strips):
            zs = z0 + i*strip_size
            ze = z0 + min((i+1)*strip_size, Z)

            if zs >= z1:
                break

            tasks.append(
                ex.submit(
                    lambda s=zs, e=ze: (s-z0, arr_in[0,0,s:e,y0:y1,x0:x1])
                )
            )

        for future in tasks:
            z_offset, slice_data = future.result()
            block[z_offset : z_offset + slice_data.shape[0], :, :] = slice_data

    return block

# --- MAIN SEGMENTATION FUNCTION (add param intensity_scale_factor, pass to _get_patch_dataset) ---
def segment_zarr_volume_blockwise(input_zarr_path, output_zarr_path, model,
                                  input_dim=64, output_dim=36,
                                  batch_size=16, block_size=256, level=1,
                                  dtype_in=np.uint16, dtype_out=np.float32,
                                  normalizer=(2**16 - 1), threshold=None,
                                  z_step_reduction=0, intensity_scale_factor=10, intensity_offset=0):
    """
    z_step_reduction: int >= 0. If 3, effective z-step = output_dim - 3.
    Also we trim the first z_step_reduction slices of each output patch when writing
    back into the block (except for patches whose iz == 0).
    intensity_scale_factor: float. Value to divide the signal by before normalization.
    """
    dim_offset = (input_dim - output_dim) // 2
    if (input_dim - output_dim) % 2 != 0:
        raise ValueError("input_dim-output_dim must be even.")

    # --- Open input ---
    zarr_in = zarr.open(input_zarr_path, mode='r')
    key = level if isinstance(level, str) else str(level)
    arr_in = zarr_in[key]
    Z, Y, X = arr_in.shape[-3:]
    print(f"Input volume shape: Z={Z}, Y={Y}, X={X}")

    # --- Create output zarr ---
    out_zarr = zarr.open(output_zarr_path, mode='w',
                         shape=(Z,Y,X), dtype=dtype_out,
                         chunks=(128,128,128))

    # --- Compute block starts (unchanged) ---
    step = block_size - dim_offset
    z_blocks = list(range(0, Z, step))
    y_blocks = list(range(0, Y, step))
    x_blocks = list(range(0, X, step))

    total_blocks = len(z_blocks) * len(y_blocks) * len(x_blocks)
    print(f"Total blocks to process: {total_blocks}, z_block positions: {z_blocks}, y_block positions: {y_blocks}, x_block positions: {x_blocks}")
    pbar = tqdm(total=total_blocks, desc="Processing blocks")

    write_executor = ThreadPoolExecutor(max_workers=1)
    pending_write = None

    for z0 in z_blocks:
        for y0 in y_blocks:
            for x0 in x_blocks:
                z1 = min(z0 + block_size, Z)
                y1 = min(y0 + block_size, Y)
                x1 = min(x0 + block_size, X)

                # --- Read block ---
                block = read_zarr_block_threaded(arr_in, z0, z1, y0, y1, x0, x1, num_strips=16).astype(dtype_in)

                # robust threshold check
                if (threshold is not None) and (block.max() < threshold):
                    pbar.update(1)
                    continue

                bz, by, bx = block.shape
                if bz==0 or by==0 or bx==0:
                    pbar.update(1)
                    continue

                # --- Extract patches (pass z_step_reduction and intensity_scale_factor) ---
                patch_dataset, num_patches, patch_coords, interesting_indices = _get_patch_dataset(
                    block, input_dim, output_dim, normalizer, batch_size, threshold=threshold,
                    z_step_reduction=z_step_reduction, intensity_scale_factor=intensity_scale_factor,
                    intensity_offset=intensity_offset
                )

                # --- GPU inference ---
                if len(interesting_indices) > 0:
                    outputs = []

                    for batch in patch_dataset:
                        preds = model(batch, training=False)
                        outputs.append(preds.numpy())

                    out_patches = np.concatenate(outputs, axis=0)

                    # out_patches = model.predict(patch_dataset, verbose=0)
                else:
                    out_patches = np.zeros((0, output_dim, output_dim, output_dim), dtype=np.float32)

                # --- Reconstruct block (apply z-trimming) ---
                out_block = np.zeros((bz, by, bx), dtype=dtype_out)
                written = np.zeros((bz, by, bx), dtype=bool)
                z_trim = int(z_step_reduction) if z_step_reduction > 0 else 0

                for patch_out, (iz,iy,ix) in zip(out_patches, patch_coords):
                    # squeeze channel if needed
                    if patch_out.ndim == 4 and patch_out.shape[-1] == 1:
                        patch_out = np.squeeze(patch_out, axis=-1)

                    # Determine trim: only trim the first z_trim slices when the patch is not at the absolute start
                    trim = z_trim if (z_trim > 0 and iz > 0) else 0

                    # source patch shape
                    pz, py, px = patch_out.shape

                    # Source start/end (after trimming)
                    src_z0 = trim
                    src_z1 = pz

                    # Destination start in block
                    dst_z0 = iz + src_z0
                    dst_z1 = min(dst_z0 + (src_z1 - src_z0), bz)

                    # y/x cropping same as before
                    y_end = min(iy + output_dim, by)
                    x_end = min(ix + output_dim, bx)

                    y_crop = y_end - iy
                    x_crop = x_end - ix

                    # source y/x ranges
                    src_y0 = 0
                    src_y1 = src_y0 + y_crop
                    src_x0 = 0
                    src_x1 = src_x0 + x_crop

                    # ensure sizes are consistent
                    z_write_len = dst_z1 - dst_z0
                    if z_write_len <= 0 or y_crop <= 0 or x_crop <= 0:
                        continue

                    out_block[dst_z0:dst_z1, iy:y_end, ix:x_end] = patch_out[src_z0:src_z0 + z_write_len,
                                                                              src_y0:src_y1,
                                                                              src_x0:src_x1]
                    written[dst_z0:dst_z1, iy:y_end, ix:x_end] = True

                # --- Async write ---
                if pending_write is not None:
                    pending_write.result()
                    pending_write = None

                out_block[~written] = 0

                pending_write = write_executor.submit(
                    hard_seam_write, out_zarr, out_block.copy(), z0, z1, y0, y1, x0, x1, dim_offset
                )

                pbar.update(1)

    # --- Finish pending writes ---
    if pending_write is not None:
        pending_write.result()
    write_executor.shutdown()
    pbar.close()
    print("Segmentation complete.")
