"""
Connectome conversion utilities for preparing user data.

Converts raw connectome data from various sources (GSP1000, HCP, etc.)
into LDK-compatible HDF5 format.
"""

import glob
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def gsp1000_to_ldk(
    gsp_dir: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    subjects_per_chunk: int = 10,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """
    Convert GSP1000 functional data to LDK-compatible HDF5 chunks.

    Scans a directory of functional NIfTI files from the GSP1000 dataset,
    extracts time-series from within a brain mask, and saves the data into
    multiple smaller HDF5 chunk files for efficient analysis.

    Expected GSP1000 directory structure:
        gsp_dir/
        └── sub-*/
            └── func/
                └── *bld001_rest_*_finalmask.nii.gz

    Parameters
    ----------
    gsp_dir : str | Path
        Path to the GSP1000 dataset directory
    mask_path : str | Path
        Path to MNI152 brain mask (.nii.gz)
    output_dir : str | Path
        Directory where chunk HDF5 files will be saved
    subjects_per_chunk : int, default=10
        Number of subjects to include in each chunk file
    overwrite : bool, default=False
        Whether to overwrite existing chunk files

    Returns
    -------
    list[Path]
        List of created chunk file paths

    Raises
    ------
    FileNotFoundError
        If GSP directory or mask file not found
    ValueError
        If no matching NIfTI files found in GSP directory

    Examples
    --------
    >>> chunk_files = gsp1000_to_ldk(
    ...     gsp_dir="/data/GSP1000",
    ...     mask_path="/data/templates/MNI152_T1_2mm_Brain_Mask.nii.gz",
    ...     output_dir="/data/connectomes/gsp1000_chunks",
    ...     subjects_per_chunk=10
    ... )
    >>> print(f"Created {len(chunk_files)} chunk files")

    Notes
    -----
    - Each chunk file is self-contained with all necessary metadata
    - Timeseries are NOT preprocessed (demeaning, variance normalization)
      to preserve raw data - preprocessing happens during analysis
    - HDF5 files use chunking (1, n_timepoints, n_voxels) for efficient
      subject-wise access
    """
    gsp_dir = Path(gsp_dir)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)

    print("🚀 Starting GSP1000 to LDK conversion...")

    # Validate inputs
    if not gsp_dir.exists():
        raise FileNotFoundError(f"GSP directory not found: {gsp_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Find all functional NIfTI files
    search_pattern = str(gsp_dir / "sub-*" / "func" / "*bld001_rest_*_finalmask.nii.gz")
    all_subject_files = sorted(glob.glob(search_pattern))

    if not all_subject_files:
        raise ValueError(
            f"No NIfTI files found matching pattern: {search_pattern}\n"
            "Expected GSP1000 structure: sub-*/func/*bld001_rest_*_finalmask.nii.gz"
        )

    n_total_subjects = len(all_subject_files)
    print(f"Found {n_total_subjects} subject files")

    # Load brain mask metadata once
    print(f"Loading brain mask from: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    mask_affine = mask_img.affine
    in_mask_indices = np.where(mask_data)
    n_voxels = len(in_mask_indices[0])

    # Get number of timepoints from first subject
    first_img = nib.load(all_subject_files[0])
    n_timepoints = first_img.shape[3]

    print(f"Mask contains {n_voxels:,} in-brain voxels")
    print(f"Detected {n_timepoints} timepoints per subject")

    # Split subjects into chunks
    subject_chunks = [
        all_subject_files[i : i + subjects_per_chunk]
        for i in range(0, n_total_subjects, subjects_per_chunk)
    ]
    print(f"Data will be split into {len(subject_chunks)} chunk files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each chunk
    created_files = []
    for chunk_idx, chunk_files in enumerate(tqdm(subject_chunks, desc="Processing chunks")):
        chunk_filename = output_dir / f"gsp1000_chunk_{chunk_idx:03d}.h5"

        if chunk_filename.exists() and not overwrite:
            print(f"  Skipping existing chunk: {chunk_filename.name}")
            created_files.append(chunk_filename)
            continue

        n_subjects_in_chunk = len(chunk_files)

        with h5py.File(chunk_filename, "w") as hf:
            # Create timeseries dataset with chunking for efficient access
            timeseries_dset = hf.create_dataset(
                "timeseries",
                shape=(n_subjects_in_chunk, n_timepoints, n_voxels),
                dtype=np.float32,
                chunks=(1, n_timepoints, n_voxels),
                compression="gzip",
                compression_opts=1,  # Minimal compression for speed
            )

            # Store metadata (makes each chunk self-contained)
            hf.create_dataset("mask_indices", data=np.vstack(in_mask_indices).T)
            hf.create_dataset("mask_affine", data=mask_affine)

            # Attributes
            hf.attrs["n_subjects"] = n_subjects_in_chunk
            hf.attrs["n_timepoints"] = n_timepoints
            hf.attrs["n_voxels"] = n_voxels
            hf.attrs["mask_shape"] = mask_data.shape
            hf.attrs["space"] = "MNI152_2mm"
            hf.attrs["description"] = f"GSP1000 functional connectome chunk {chunk_idx}"
            hf.attrs["source"] = "Harvard Dataverse doi:10.7910/DVN/ILXIKS"

            # Process subjects in this chunk
            for subj_idx, file_path in enumerate(
                tqdm(
                    chunk_files,
                    desc=f"  Chunk {chunk_idx + 1}/{len(subject_chunks)}",
                    leave=False,
                )
            ):
                # Load 4D functional data
                func_img = nib.load(file_path)
                func_data = func_img.get_fdata()

                # Extract timeseries from masked voxels and transpose
                # Shape: (n_timepoints, n_voxels)
                subject_timeseries = func_data[in_mask_indices].T

                # Store in HDF5
                timeseries_dset[subj_idx, :, :] = subject_timeseries

        created_files.append(chunk_filename)

    print("\n✅ Conversion complete!")
    print(f"Created {len(created_files)} chunk files in: {output_dir}")
    print("\nTo use in analyses, set:")
    print(f"  export LDK_CONNECTOME_DIR={output_dir}")

    return created_files


def tractogram_to_ldk(
    tractogram_path: str | Path,
    output_path: str | Path,
    *,
    tdi_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Convert MRtrix3 tractogram to LDK-compatible format.

    Stores tractogram metadata and optionally a precomputed track density
    image (TDI) for structural network mapping analyses.

    Parameters
    ----------
    tractogram_path : str | Path
        Path to MRtrix3 .tck tractogram file
    output_path : str | Path
        Output path for HDF5 file
    tdi_path : str | Path, optional
        Path to precomputed whole-brain TDI (.nii.gz)
        If not provided, TDI will be computed during analysis
    overwrite : bool, default=False
        Whether to overwrite existing output file

    Returns
    -------
    Path
        Path to created HDF5 file

    Raises
    ------
    FileNotFoundError
        If tractogram or TDI file not found
    RuntimeError
        If MRtrix3 tools not available

    Examples
    --------
    >>> output = tractogram_to_ldk(
    ...     tractogram_path="/data/hcp_tractogram.tck",
    ...     output_path="/data/connectomes/hcp_structural.h5",
    ...     tdi_path="/data/hcp_tdi.nii.gz"
    ... )

    Notes
    -----
    This function stores metadata about the tractogram rather than the full
    streamline data. Actual tractography filtering (tckedit) happens during
    StructuralNetworkMapping analysis using the original .tck file.
    """
    tractogram_path = Path(tractogram_path)
    output_path = Path(output_path)

    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram not found: {tractogram_path}")

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    print("🚀 Converting tractogram to LDK format...")
    print(f"Input: {tractogram_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        # Store tractogram metadata
        hf.attrs["tractogram_path"] = str(tractogram_path.resolve())
        hf.attrs["connectome_type"] = "structural"
        hf.attrs["space"] = "MNI152_2mm"

        # If TDI provided, store it
        if tdi_path:
            tdi_path = Path(tdi_path)
            if not tdi_path.exists():
                raise FileNotFoundError(f"TDI file not found: {tdi_path}")

            print(f"Loading TDI from: {tdi_path}")
            tdi_img = nib.load(tdi_path)
            tdi_data = tdi_img.get_fdata()

            hf.create_dataset("tdi", data=tdi_data, compression="gzip")
            hf.create_dataset("tdi_affine", data=tdi_img.affine)
            hf.attrs["tdi_shape"] = tdi_data.shape
            hf.attrs["has_tdi"] = True
        else:
            hf.attrs["has_tdi"] = False
            print("No TDI provided - will be computed during analysis")

        hf.attrs["description"] = "Structural connectome for LDK"

    print(f"✅ Conversion complete: {output_path}")
    return output_path
