"""
Connectome conversion utilities for preparing user data.

Converts raw connectome data from various sources (GSP1000, HCP, etc.)
into Lacuna-compatible HDF5 format and tractogram formats.
"""

import glob
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def gsp1000_to_hdf5(
    gsp_dir: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    subjects_per_chunk: int = 10,
    *,
    max_subjects: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Convert GSP1000 functional data to Lacuna-compatible HDF5 chunks.

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
    max_subjects : int, optional
        Maximum number of subjects to process. If set, only the first
        ``max_subjects`` files are used. Useful for test mode.
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
    >>> chunk_files = gsp1000_to_hdf5(
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

    if max_subjects is not None and len(all_subject_files) > max_subjects:
        all_subject_files = all_subject_files[:max_subjects]

    n_total_subjects = len(all_subject_files)
    print(f"Found {n_total_subjects} subject files")

    # Load brain mask metadata once. Reorient to canonical RAS+ so subjects
    # stored in a different orientation (e.g. FSL's radiological MNI152) can be
    # aligned to the mask grid by an axis flip rather than resampling.
    print(f"Loading brain mask from: {mask_path}")
    mask_img = nib.as_closest_canonical(nib.load(mask_path))
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
            hf.attrs["space"] = "MNI152NLin6Asym"
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
                # Load 4D functional data, reorienting to canonical RAS+ to
                # match the mask grid. GSP1000 is distributed in FSL's
                # radiological orientation, which differs from the templateflow
                # mask only by an axis flip; canonicalizing both makes the voxel
                # grids identical without resampling (and without an L/R swap).
                func_img = nib.as_closest_canonical(nib.load(file_path))

                # Validate this subject is on the mask's grid before indexing —
                # otherwise the mask voxel indices extract the wrong anatomical
                # voxels (or raise an opaque IndexError) and silently corrupt the
                # connectome.
                if func_img.shape[:3] != mask_data.shape:
                    raise ValueError(
                        f"Functional image '{file_path}' has spatial shape "
                        f"{tuple(func_img.shape[:3])} but the brain mask is "
                        f"{tuple(mask_data.shape)}. All subjects must share the mask grid."
                    )
                if not np.allclose(func_img.affine, mask_affine, atol=1e-3):
                    raise ValueError(
                        f"Functional image '{file_path}' affine does not match the brain "
                        "mask affine; the data are on different grids (even after "
                        "reorienting to canonical RAS+).\n"
                        f"  image affine (RAS+):\n{func_img.affine}\n"
                        f"  mask affine (RAS+):\n{mask_affine}"
                    )
                if func_img.shape[3] != n_timepoints:
                    raise ValueError(
                        f"Functional image '{file_path}' has {func_img.shape[3]} timepoints "
                        f"but {n_timepoints} were expected (from the first subject)."
                    )

                func_data = func_img.get_fdata()

                # Extract timeseries from masked voxels and transpose
                # Shape: (n_timepoints, n_voxels)
                subject_timeseries = func_data[in_mask_indices].T

                # Store in HDF5
                timeseries_dset[subj_idx, :, :] = subject_timeseries

        created_files.append(chunk_filename)

    print("\n✅ Conversion complete!")
    print(f"Created {len(created_files)} chunk files in: {output_dir}")

    return created_files


def merge_trk_to_tck(
    source_dir: str | Path,
    output_path: str | Path,
    *,
    exclude_patterns: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Merge multiple TrackVis .trk/.trk.gz tractograms into a single MRtrix3 .tck file.

    Recursively finds all .trk and .trk.gz files in the source directory,
    loads their streamlines (excluding files matching specified patterns),
    and saves them as a single merged .tck tractogram.

    Parameters
    ----------
    source_dir : str | Path
        Directory containing .trk/.trk.gz tract files (searched recursively).
    output_path : str | Path
        Output path for the merged .tck file.
    exclude_patterns : list[str], optional
        List of patterns to match against file paths for exclusion.
        Files whose path contains any of these strings (case-insensitive)
        are skipped. Default: ``["cranial nerve", "cranial_nerve"]``.
    overwrite : bool, default=False
        Whether to overwrite an existing output file.

    Returns
    -------
    Path
        Path to the created .tck file.

    Raises
    ------
    FileNotFoundError
        If source directory not found.
    ValueError
        If no .trk/.trk.gz files found or output is not .tck format.
    RuntimeError
        If merging fails.

    Examples
    --------
    >>> tck_path = merge_trk_to_tck(
    ...     source_dir="/data/hcp1065_tracts",
    ...     output_path="/data/hcp1065.tck",
    ... )
    """
    from nibabel.streamlines import TckFile, Tractogram

    source_dir = Path(source_dir)
    output_path = Path(output_path)

    if exclude_patterns is None:
        exclude_patterns = ["cranial nerve", "cranial_nerve"]

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if output_path.suffix != ".tck":
        raise ValueError(f"Output must be .tck format, got: {output_path.suffix}")

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    # Find all .trk and .trk.gz files
    trk_files = sorted(source_dir.rglob("*.trk.gz")) + sorted(source_dir.rglob("*.trk"))

    if not trk_files:
        raise ValueError(
            f"No .trk or .trk.gz files found in: {source_dir}\n"
            "Expected directory containing tractography files."
        )

    # Filter out excluded patterns
    exclude_lower = [p.lower() for p in exclude_patterns]
    filtered_files = []
    for f in trk_files:
        path_str = str(f).lower()
        if any(pattern in path_str for pattern in exclude_lower):
            continue
        filtered_files.append(f)

    if not filtered_files:
        raise ValueError(
            f"All {len(trk_files)} tract files were excluded by patterns: {exclude_patterns}"
        )

    print(
        f"Found {len(filtered_files)} tract files ({len(trk_files) - len(filtered_files)} excluded)"
    )

    # Load and merge streamlines
    all_streamlines = []
    files_processed = 0

    print("Loading and merging streamlines...")
    for trk_path in tqdm(filtered_files, desc="Merging tracts"):
        try:
            trk = nib.streamlines.load(str(trk_path))
            all_streamlines.extend(trk.streamlines)
            files_processed += 1
        except Exception as e:
            print(f"  Warning: Error loading {trk_path.name}: {e}")

    if not all_streamlines:
        raise RuntimeError("No streamlines loaded from any tract file.")

    print(f"Processed {files_processed} files, {len(all_streamlines)} total streamlines")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create merged tractogram and save
    print(f"Saving merged tractogram to {output_path}...")
    try:
        tractogram = Tractogram(
            streamlines=all_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        tck = TckFile(tractogram)
        tck.save(str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save merged tractogram: {e}") from e

    print(f"Merge complete: {output_path}")
    return output_path


def trk_to_tck(
    trk_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Convert TrackVis .trk tractogram to MRtrix3 .tck format using nibabel.

    This conversion is necessary because StructuralNetworkMapping uses MRtrix3
    tools (tckedit, tckmap, mrcalc) which require .tck format. The default
    dTOR985 tractogram is distributed in .trk format.

    Uses nibabel's streamlines module for pure Python conversion without
    requiring MRtrix3 to be installed.

    Parameters
    ----------
    trk_path : str | Path
        Path to input TrackVis .trk file (e.g., dTOR985.trk)
    output_path : str | Path
        Output path for MRtrix3 .tck file
    overwrite : bool, default=False
        Whether to overwrite existing output file

    Returns
    -------
    Path
        Path to created .tck file

    Raises
    ------
    FileNotFoundError
        If trk file not found
    ValueError
        If input is not .trk or output is not .tck format
    RuntimeError
        If conversion fails

    Examples
    --------
    >>> # Convert dTOR985 tractogram
    >>> tck_path = trk_to_tck(
    ...     trk_path="/data/dTOR985.trk",
    ...     output_path="/data/dTOR985.tck"
    ... )
    >>>
    >>> # Later use in analysis:
    >>> analysis = StructuralNetworkMapping(tractogram_path="/data/dTOR985.tck")

    Notes
    -----
    - Uses nibabel for pure Python conversion (no external dependencies)
    - Preserves streamline coordinates and header information
    - The .tck file can be much larger than .trk due to format differences
    - For dTOR985: expect ~5-10GB .tck file from ~2GB .trk file

    See Also
    --------
    nibabel.streamlines: https://nipy.org/nibabel/reference/nibabel.streamlines.html
    """
    import nibabel as nib

    trk_path = Path(trk_path)
    output_path = Path(output_path)

    # Validate formats
    if trk_path.suffix != ".trk":
        raise ValueError(
            f"Input must be .trk format, got: {trk_path.suffix}\n"
            "Expected TrackVis .trk file (e.g., dTOR985.trk)"
        )

    if output_path.suffix != ".tck":
        raise ValueError(
            f"Output must be .tck format, got: {output_path.suffix}\n"
            "MRtrix3 tools require .tck format"
        )

    if not trk_path.exists():
        raise FileNotFoundError(f"TRK file not found: {trk_path}")

    if output_path.exists() and not overwrite:
        print(f"Output file already exists: {output_path}")
        return output_path

    print("🚀 Converting .trk to .tck format...")
    print(f"Input:  {trk_path}")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Stream streamlines from disk via nibabel's lazy interface so the full
        # tractogram is never held in memory. The eager TrkFile.load() path
        # materializes every streamline at once (~32 GB for dTOR985's ~11M
        # streamlines); lazy loading keeps peak memory to a single streamline
        # plus I/O buffers, regardless of tractogram size.
        print("Converting .trk -> .tck (streaming, low memory)...")
        affine = nib.streamlines.load(str(trk_path), lazy_load=True).affine

        def _streamlines():
            # Re-open per pass so the generator is replayable — the TCK writer
            # iterates the streamlines more than once (write + finalize count).
            src = nib.streamlines.load(str(trk_path), lazy_load=True)
            yield from src.streamlines

        lazy_out = nib.streamlines.LazyTractogram(streamlines=_streamlines, affine_to_rasmm=affine)
        nib.streamlines.save(lazy_out, str(output_path))

    except Exception as e:
        raise RuntimeError(
            f"Conversion failed: {e}\n"
            "Check that .trk file is valid and nibabel is properly installed."
        ) from e

    print(f"✅ Conversion complete: {output_path}")
    print("Note: Keep .tck file for StructuralNetworkMapping analyses")

    return output_path
