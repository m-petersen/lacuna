"""Utilities for working with GSP1000 connectome data.

This module provides functions to convert GSP1000 functional data into
optimized HDF5 batch files for efficient lesion network mapping.
"""

import glob
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def create_connectome_batches(
    gsp_dir: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    subjects_per_batch: int = 50,
    pattern: str = "sub-*/func/*bld001_rest_*_finalmask.nii.gz",
    verbose: bool = True,
) -> list[Path]:
    """Create HDF5 batch files from GSP1000 functional data.

    Scans a directory of functional NIfTI files, extracts time-series from
    within a brain mask, and saves the data into multiple smaller HDF5 batch
    files optimized for memory-efficient lesion network mapping.

    Parameters
    ----------
    gsp_dir : str or Path
        Directory containing GSP1000 subject functional data.
        Expected structure: sub-*/func/*bld001_rest_*_finalmask.nii.gz
    mask_path : str or Path
        Path to brain mask NIfTI file (e.g., MNI152_T1_2mm_Brain_Mask.nii.gz).
        Defines which voxels to extract.
    output_dir : str or Path
        Directory where HDF5 batch files will be saved.
    subjects_per_batch : int, default=50
        Number of subjects to include in each batch file.
        Larger batches = fewer files but more memory per batch.
    pattern : str, optional
        Glob pattern to find functional files within gsp_dir.
        Default matches standard GSP1000 structure.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    list of Path
        Paths to created HDF5 batch files, sorted by name.

    Notes
    -----
    Each HDF5 batch file contains:
    - 'timeseries': (n_subjects, n_timepoints, n_voxels) float32 array
    - 'mask_indices': (3, n_voxels) array of mask coordinates
    - 'mask_affine': (4, 4) affine transformation matrix
    - Attributes: n_subjects, n_timepoints, n_voxels, mask_shape

    The batch files are designed for sequential loading during analysis,
    minimizing memory footprint while maintaining processing speed.

    Examples
    --------
    >>> from ldk.utils.gsp1000 import create_connectome_batches
    >>> batch_files = create_connectome_batches(
    ...     gsp_dir="/data/GSP1000",
    ...     mask_path="/data/MNI152_T1_2mm_Brain_Mask.nii.gz",
    ...     output_dir="/data/connectomes/gsp1000_batches",
    ...     subjects_per_batch=100
    ... )
    >>> print(f"Created {len(batch_files)} batch files")
    """
    gsp_dir = Path(gsp_dir)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)

    if verbose:
        print("🚀 Creating connectome batch files from GSP1000 data...")

    # 1. Find all functional NIfTI files
    search_pattern = str(gsp_dir / pattern)
    all_subject_files = sorted(glob.glob(search_pattern))

    if not all_subject_files:
        raise FileNotFoundError(
            f"No NIfTI files found matching pattern: {search_pattern}\n"
            f"Expected structure: {gsp_dir}/sub-*/func/*bld001_rest_*_finalmask.nii.gz"
        )

    n_total_subjects = len(all_subject_files)
    if verbose:
        print(f"✓ Found {n_total_subjects} subject files")

    # 2. Load brain mask and extract metadata
    if verbose:
        print(f"Loading brain mask: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    mask_affine = mask_img.affine
    in_mask_indices = np.where(mask_data)
    n_voxels = len(in_mask_indices[0])

    # Get number of timepoints from first subject
    first_img = nib.load(all_subject_files[0])
    n_timepoints = first_img.shape[3]

    if verbose:
        print(f"✓ Mask contains {n_voxels:,} in-brain voxels")
        print(f"✓ Detected {n_timepoints} timepoints per subject")

    # 3. Split subjects into batches
    subject_batches = [
        all_subject_files[i : i + subjects_per_batch]
        for i in range(0, n_total_subjects, subjects_per_batch)
    ]
    n_batches = len(subject_batches)

    if verbose:
        print(f"✓ Will create {n_batches} batch files ({subjects_per_batch} subjects/batch)")

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # 4. Process each batch
    progress_iter = tqdm(subject_batches, desc="Creating batches") if verbose else subject_batches

    for batch_idx, batch_files in enumerate(progress_iter):
        batch_filename = output_dir / f"connectome_batch_{batch_idx:03d}.h5"
        n_subjects_in_batch = len(batch_files)

        with h5py.File(batch_filename, "w") as hf:
            # Create dataset for this batch
            timeseries_dset = hf.create_dataset(
                "timeseries",
                shape=(n_subjects_in_batch, n_timepoints, n_voxels),
                dtype=np.float32,
                chunks=(1, n_timepoints, n_voxels),
                compression="gzip",
                compression_opts=1,  # Fast compression
            )

            # Store metadata (makes each batch self-contained)
            hf.create_dataset("mask_indices", data=np.vstack(in_mask_indices))
            hf.create_dataset("mask_affine", data=mask_affine)
            hf.attrs["n_subjects"] = n_subjects_in_batch
            hf.attrs["n_timepoints"] = n_timepoints
            hf.attrs["n_voxels"] = n_voxels
            hf.attrs["mask_shape"] = mask_data.shape

            # Process subjects in current batch
            for subj_idx, func_path in enumerate(batch_files):
                func_img = nib.load(func_path)
                func_data = func_img.get_fdata()

                # Extract in-mask voxels and transpose to (timepoints, voxels)
                subject_timeseries = func_data[in_mask_indices].T

                # Save to HDF5
                timeseries_dset[subj_idx, :, :] = subject_timeseries

        created_files.append(batch_filename)

    if verbose:
        print(f"\n✅ Created {len(created_files)} batch files in {output_dir}")
        total_size_mb = sum(f.stat().st_size for f in created_files) / (1024**2)
        print(f"✓ Total size: {total_size_mb:.1f} MB")

    return created_files


def validate_connectome_batches(batch_dir: str | Path, verbose: bool = True) -> dict:
    """Validate integrity of HDF5 connectome batch files.

    Parameters
    ----------
    batch_dir : str or Path
        Directory containing HDF5 batch files.
    verbose : bool, default=True
        Print validation results.

    Returns
    -------
    dict
        Validation summary with keys: n_batches, total_subjects, n_timepoints,
        n_voxels, mask_shape, consistent, errors
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.h5"))

    if not batch_files:
        raise FileNotFoundError(f"No HDF5 files found in {batch_dir}")

    errors = []
    total_subjects = 0
    reference_metadata = None

    if verbose:
        print(f"Validating {len(batch_files)} batch files...")

    for batch_file in batch_files:
        try:
            with h5py.File(batch_file, "r") as hf:
                # Check required datasets
                required = ["timeseries", "mask_indices", "mask_affine"]
                for key in required:
                    if key not in hf:
                        errors.append(f"{batch_file.name}: Missing dataset '{key}'")

                # Extract metadata
                n_subjects = hf.attrs["n_subjects"]
                n_timepoints = hf.attrs["n_timepoints"]
                n_voxels = hf.attrs["n_voxels"]
                mask_shape = tuple(hf.attrs["mask_shape"])

                total_subjects += n_subjects

                # Check consistency with first batch
                if reference_metadata is None:
                    reference_metadata = {
                        "n_timepoints": n_timepoints,
                        "n_voxels": n_voxels,
                        "mask_shape": mask_shape,
                    }
                else:
                    if n_timepoints != reference_metadata["n_timepoints"]:
                        errors.append(
                            f"{batch_file.name}: Inconsistent n_timepoints "
                            f"({n_timepoints} vs {reference_metadata['n_timepoints']})"
                        )
                    if n_voxels != reference_metadata["n_voxels"]:
                        errors.append(
                            f"{batch_file.name}: Inconsistent n_voxels "
                            f"({n_voxels} vs {reference_metadata['n_voxels']})"
                        )

        except Exception as e:
            errors.append(f"{batch_file.name}: Error reading file - {e}")

    summary = {
        "n_batches": len(batch_files),
        "total_subjects": total_subjects,
        "consistent": len(errors) == 0,
        "errors": errors,
    }
    summary.update(reference_metadata or {})

    if verbose:
        if summary["consistent"]:
            print("✅ All batches valid!")
            print(f"  - {summary['n_batches']} batches")
            print(f"  - {summary['total_subjects']} total subjects")
            print(f"  - {summary['n_timepoints']} timepoints")
            print(f"  - {summary['n_voxels']:,} voxels")
        else:
            print(f"❌ Found {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")

    return summary
