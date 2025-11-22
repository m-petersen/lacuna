
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from lacuna import MaskData
from lacuna.analysis import FunctionalNetworkMapping


def create_mock_data(tmp_path):
    """Create mock connectome and lesion data."""
    connectome_path = tmp_path / "mock_connectome.h5"

    # Create realistic test data dimensions
    n_subjects = 5
    n_timepoints = 50
    n_voxels = 200000  # Realistic number of gray matter voxels

    print(f"Creating mock connectome with {n_voxels} voxels...")

    # Create random timeseries data
    timeseries = np.random.randn(n_subjects, n_timepoints, n_voxels).astype(np.float32)

    # Create mask indices (3, n_voxels) format
    # Simulate a 3D volume of 100x100x100
    coords = np.unravel_index(np.random.choice(100*100*100, n_voxels, replace=False), (100, 100, 100))
    mask_indices = np.array(coords)

    # Create affine matrix (2mm MNI152 space)
    mask_affine = np.array([
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    mask_shape = (100, 100, 100)

    # Write to HDF5
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=timeseries)
        f.create_dataset("mask_indices", data=mask_indices)
        f.create_dataset("mask_affine", data=mask_affine)
        f.attrs["mask_shape"] = mask_shape

    # Create a large lesion (e.g., 5000 voxels)
    print("Creating mock lesion...")
    mask_data_arr = np.zeros(mask_shape, dtype=np.uint8)

    # Select some random voxels to be part of the lesion
    # Ensure some overlap with connectome mask
    lesion_indices = np.random.choice(n_voxels, 5000, replace=False)
    lesion_coords = mask_indices[:, lesion_indices]
    mask_data_arr[lesion_coords[0], lesion_coords[1], lesion_coords[2]] = 1

    mask_img = nib.Nifti1Image(mask_data_arr, mask_affine)
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    lesion = MaskData.from_nifti(
        str(lesion_path),
        metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    return connectome_path, lesion

def run_benchmark():
    # Create temporary directory
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        connectome_path, lesion = create_mock_data(tmp_dir)

        analysis = FunctionalNetworkMapping(
            connectome_path=str(connectome_path),
            method="boes",
            verbose=False
        )

        # Pre-load mask info to isolate _get_lesion_voxel_indices time
        analysis._load_mask_info()

        print("Starting benchmark of _get_lesion_voxel_indices...")
        start_time = time.time()

        # Run the function being optimized
        voxel_indices = analysis._get_lesion_voxel_indices(lesion)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Found {len(voxel_indices)} overlapping voxels")
        print(f"Execution time: {duration:.4f} seconds")

    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    run_benchmark()
