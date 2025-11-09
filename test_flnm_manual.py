#!/usr/bin/env python3
"""
Manual test script for Functional Lesion Network Mapping (fLNM) analysis.

This script tests the FunctionalNetworkMapping implementation with real data.
Configure the paths below to point to your data files.

Requirements:
- Lesion mask in MNI152 space (NIfTI format)
- Functional connectome in HDF5 format with structure:
  - timeseries: (n_subjects, n_timepoints, n_voxels)
  - mask_indices: (n_voxels, 3) or (3, n_voxels)
  - mask_affine: (4, 4)
  - mask_shape: attribute (tuple)
"""

import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR DATA
# ============================================================================

# Path to lesion mask (must be in MNI152 space, binary mask)
LESION_PATH = "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/raw/lesion_masks/acuteinfarct/CAS_005_infarct.nii.gz"

# Path to functional connectome (HDF5 format)
# Can be either:
# - A single HDF5 file: "/path/to/connectome.h5"
# - A directory with multiple batch files: "/path/to/connectome_batches/"
# Example: GSP1000 connectome converted using ldk.io.convert.gsp1000_to_ldk()
CONNECTOME_PATH = "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/connectomes/gsp1000_batches"

# Optional: Path to save output results
OUTPUT_DIR = (
    "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/processed/fLNM_LDK"
)

# Analysis parameters
METHOD = "boes"  # Options: "boes" or "pini"
PINI_PERCENTILE = 20  # Only used if method="pini"
VERBOSE = True  # Show detailed progress during analysis
COMPUTE_T_MAP = True  # Compute t-statistic maps
T_THRESHOLD = 2.0  # Create binary threshold map (|t| > threshold), set to None to disable

# ============================================================================
# TEST SCRIPT - NO NEED TO EDIT BELOW THIS LINE
# ============================================================================


def check_file_exists(path, description):
    """Check if file or directory exists and provide helpful error message."""
    path = Path(path)
    if not path.exists():
        print(f"❌ ERROR: {description} not found at: {path}")
        print("   Please update the path in this script.")
        return False

    # Show what type of path it is
    if path.is_dir():
        print(f"✓ Found {description} (directory): {path}")
    else:
        print(f"✓ Found {description} (file): {path}")
    return True


def inspect_connectome(connectome_path):
    """Inspect connectome HDF5 file structure."""
    from pathlib import Path

    import h5py

    print("\n" + "=" * 70)
    print("CONNECTOME INSPECTION")
    print("=" * 70)

    path = Path(connectome_path)

    # Check if it's a directory or single file
    if path.is_dir():
        h5_files = sorted(path.glob("*.h5"))
        if not h5_files:
            print(f"\n❌ No HDF5 files found in directory: {path}")
            return False
        print(f"\nFound {len(h5_files)} HDF5 batch files:")
        for i, f in enumerate(h5_files[:5], 1):  # Show first 5
            print(f"  {i}. {f.name}")
        if len(h5_files) > 5:
            print(f"  ... and {len(h5_files) - 5} more")

        # Inspect first file
        first_file = h5_files[0]
        print(f"\nInspecting first batch: {first_file.name}")
    else:
        first_file = path
        print("\nSingle HDF5 file mode")

    with h5py.File(first_file, "r") as f:
        print("\nDatasets in HDF5 file:")
        for key in f.keys():
            dataset = f[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")

        print("\nAttributes:")
        for key in f.attrs.keys():
            print(f"  - {key}: {f.attrs[key]}")

        # Check expected structure
        print("\n" + "-" * 70)
        print("Structure Validation:")
        has_timeseries = "timeseries" in f
        has_mask_indices = "mask_indices" in f
        has_mask_affine = "mask_affine" in f
        has_mask_shape = "mask_shape" in f.attrs

        print(f"  ✓ timeseries dataset: {'✓' if has_timeseries else '✗'}")
        print(f"  ✓ mask_indices dataset: {'✓' if has_mask_indices else '✗'}")
        print(f"  ✓ mask_affine dataset: {'✓' if has_mask_affine else '✗'}")
        print(f"  ✓ mask_shape attribute: {'✓' if has_mask_shape else '✗'}")

        if has_timeseries:
            shape = f["timeseries"].shape
            print(f"\n  Timeseries shape: {shape}")
            print(f"    - n_subjects: {shape[0]}")
            print(f"    - n_timepoints: {shape[1]}")
            print(f"    - n_voxels: {shape[2]}")

        if has_mask_indices:
            shape = f["mask_indices"].shape
            print(f"\n  Mask indices shape: {shape}")
            if shape[0] == 3:
                print("    - Format: (3, n_voxels) - CORRECT")
            elif shape[1] == 3:
                print("    - Format: (n_voxels, 3) - Will be transposed automatically")
            else:
                print("    - Format: UNEXPECTED - may cause errors")

        return has_timeseries and has_mask_indices and has_mask_affine and has_mask_shape


def inspect_lesion(lesion_path, connectome_path=None):
    """Inspect lesion mask and check compatibility with connectome."""
    print("\n" + "=" * 70)
    print("LESION MASK INSPECTION")
    print("=" * 70)

    img = nib.load(lesion_path)
    data = img.get_fdata()

    print("\nImage properties:")
    print(f"  - Shape: {img.shape}")
    print(f"  - Affine:\n{img.affine}")
    print(f"  - Data type: {data.dtype}")

    unique_vals = np.unique(data)
    print(f"\nUnique values in mask: {unique_vals}")

    is_binary = np.all(np.isin(unique_vals, [0, 1]))
    print(f"  - Binary (0/1 only): {'✓' if is_binary else '✗'}")

    n_lesion_voxels = np.sum(data > 0)
    total_voxels = np.prod(data.shape)
    lesion_percent = (n_lesion_voxels / total_voxels) * 100

    print("\nLesion statistics:")
    print(f"  - Lesion voxels: {n_lesion_voxels:,}")
    print(f"  - Total voxels: {total_voxels:,}")
    print(f"  - Lesion coverage: {lesion_percent:.2f}%")

    # Estimate lesion volume
    voxel_dims = img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)
    lesion_volume_mm3 = n_lesion_voxels * voxel_volume_mm3
    lesion_volume_ml = lesion_volume_mm3 / 1000

    print(f"  - Voxel size: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {voxel_dims[2]:.2f} mm")
    print(f"  - Lesion volume: {lesion_volume_mm3:.1f} mm³ ({lesion_volume_ml:.2f} mL)")

    # Check compatibility with connectome if provided
    if connectome_path:
        from pathlib import Path

        import h5py

        path = Path(connectome_path)
        if path.is_dir():
            h5_files = sorted(path.glob("*.h5"))
            first_file = h5_files[0] if h5_files else None
        else:
            first_file = path

        if first_file and first_file.exists():
            with h5py.File(first_file, "r") as f:
                connectome_shape = tuple(f.attrs["mask_shape"])

            print("\n" + "-" * 70)
            print("Compatibility Check:")
            print(f"  - Lesion shape: {img.shape}")
            print(f"  - Connectome shape: {connectome_shape}")

            if img.shape == connectome_shape:
                print("  ✓ Shapes match - no resampling needed")
            else:
                print("  ⚠️  Shapes differ - automatic resampling will occur")
                print(f"     Lesion will be resampled from {img.shape} to {connectome_shape}")

    return is_binary

    print(f"  - Voxel size: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {voxel_dims[2]:.2f} mm")
    print(f"  - Lesion volume: {lesion_volume_mm3:.1f} mm³ ({lesion_volume_ml:.2f} mL)")

    return is_binary


def run_flnm_analysis():
    """Run the Functional Lesion Network Mapping analysis."""
    print("\n" + "=" * 70)
    print("RUNNING FUNCTIONAL NETWORK MAPPING ANALYSIS")
    print("=" * 70)

    # Import LDK modules
    try:
        from ldk import LesionData
        from ldk.analysis import FunctionalNetworkMapping

        print("✓ Successfully imported ldk modules")
    except ImportError as e:
        print(f"❌ Failed to import ldk: {e}")
        print("   Make sure the package is installed: pip install -e .")
        return None

    # Load lesion data
    print("\nLoading lesion data...")
    start = time.time()

    lesion = LesionData.from_nifti(
        LESION_PATH,
        metadata={
            "subject_id": "test_subject",
            "space": "MNI152_2mm",  # Adjust if using 1mm
        },
    )
    print(f"✓ Loaded lesion data in {time.time() - start:.2f}s")

    # Create analysis object
    print("\nCreating FunctionalNetworkMapping analysis...")
    print(f"  - Method: {METHOD}")
    if METHOD == "pini":
        print(f"  - PINI percentile: {PINI_PERCENTILE}")
    print(f"  - Verbose mode: {VERBOSE}")
    print(f"  - Compute t-map: {COMPUTE_T_MAP}")
    if T_THRESHOLD is not None:
        print(f"  - T-threshold: {T_THRESHOLD}")

    analysis = FunctionalNetworkMapping(
        connectome_path=CONNECTOME_PATH,
        method=METHOD,
        pini_percentile=PINI_PERCENTILE,
        verbose=VERBOSE,
        compute_t_map=COMPUTE_T_MAP,
        t_threshold=T_THRESHOLD,
    )
    print("✓ Analysis object created")

    # Run analysis
    print("\nRunning analysis (this may take a few minutes)...")
    print("💡 Note: If lesion and connectome dimensions differ, automatic resampling will occur")
    start = time.time()

    try:
        result = analysis.run(lesion)
        elapsed = time.time() - start
        print(f"✓ Analysis completed in {elapsed:.2f}s ({elapsed / 60:.1f} min)")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    return result


def display_results(result):
    """Display analysis results."""
    if result is None:
        return

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # Check what's in results
    flnm_results = result.results.get("FunctionalNetworkMapping", {})

    print("\nAvailable outputs:")
    for key in flnm_results.keys():
        value = flnm_results[key]
        if hasattr(value, "shape"):
            print(f"  - {key}: {type(value).__name__} with shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  - {key}: dict with {len(value)} items")
        else:
            print(f"  - {key}: {type(value).__name__}")

    # Display summary statistics
    if "summary_statistics" in flnm_results:
        print("\nSummary statistics:")
        stats = flnm_results["summary_statistics"]
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    # Display mean correlation
    if "mean_correlation" in flnm_results:
        print(f"\nMean correlation across brain: {flnm_results['mean_correlation']:.4f}")

    # Check correlation map
    if "correlation_map" in flnm_results:
        corr_map = flnm_results["correlation_map"]
        corr_data = corr_map.get_fdata()

        print("\nCorrelation map statistics:")
        print(f"  - Shape: {corr_data.shape}")
        print(f"  - Min: {np.min(corr_data):.4f}")
        print(f"  - Max: {np.max(corr_data):.4f}")
        print(f"  - Mean: {np.mean(corr_data):.4f}")
        print(f"  - Std: {np.std(corr_data):.4f}")

        # Count non-zero voxels
        n_nonzero = np.sum(corr_data != 0)
        total = np.prod(corr_data.shape)
        print(f"  - Non-zero voxels: {n_nonzero:,} ({n_nonzero / total * 100:.1f}%)")

    return result


def save_results(result, output_dir):
    """Save results to files."""
    if result is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    flnm_results = result.results.get("FunctionalNetworkMapping", {})

    # Save correlation map
    if "correlation_map" in flnm_results:
        output_path = output_dir / "correlation_map.nii.gz"
        nib.save(flnm_results["correlation_map"], output_path)
        print(f"✓ Saved correlation map: {output_path}")

    # Save z-map
    if "z_map" in flnm_results:
        output_path = output_dir / "z_map.nii.gz"
        nib.save(flnm_results["z_map"], output_path)
        print(f"✓ Saved z-map: {output_path}")

    # Save t-map
    if "t_map" in flnm_results:
        output_path = output_dir / "t_map.nii.gz"
        nib.save(flnm_results["t_map"], output_path)
        print(f"✓ Saved t-map: {output_path}")

    # Save thresholded t-map
    if "t_threshold_map" in flnm_results:
        output_path = output_dir / "t_threshold_map.nii.gz"
        nib.save(flnm_results["t_threshold_map"], output_path)
        print(f"✓ Saved t-threshold map: {output_path}")

    # Save summary statistics as JSON
    if "summary_statistics" in flnm_results:
        import json

        output_path = output_dir / "summary_statistics.json"
        with open(output_path, "w") as f:
            json.dump(flnm_results["summary_statistics"], f, indent=2)
        print(f"✓ Saved summary statistics: {output_path}")

    print(f"\nAll results saved to: {output_dir}")


def main():
    """Main test function."""
    print("=" * 70)
    print("FUNCTIONAL LESION NETWORK MAPPING TEST")
    print("=" * 70)

    # Check if paths are configured
    if LESION_PATH == "/path/to/your/lesion_mni152.nii.gz":
        print("\n❌ ERROR: Please configure the file paths in this script first!")
        print("   Edit the CONFIGURATION section at the top of the file.")
        sys.exit(1)

    # Check files exist
    print("\nChecking input files...")
    lesion_ok = check_file_exists(LESION_PATH, "Lesion mask")
    connectome_ok = check_file_exists(CONNECTOME_PATH, "Connectome")

    if not (lesion_ok and connectome_ok):
        print("\n❌ Cannot proceed without input files.")
        sys.exit(1)

    # Inspect files
    try:
        connectome_valid = inspect_connectome(CONNECTOME_PATH)
        lesion_binary = inspect_lesion(LESION_PATH, CONNECTOME_PATH)
    except Exception as e:
        print(f"\n❌ Error inspecting files: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not connectome_valid:
        print("\n❌ Connectome structure is invalid.")
        print("   Expected HDF5 structure:")
        print("   - timeseries: (n_subjects, n_timepoints, n_voxels)")
        print("   - mask_indices: (3, n_voxels) or (n_voxels, 3)")
        print("   - mask_affine: (4, 4)")
        print("   - mask_shape: tuple attribute")
        sys.exit(1)

    if not lesion_binary:
        print("\n⚠️  WARNING: Lesion mask is not binary (0/1 values only).")
        print("   The analysis may fail. Consider binarizing the mask first.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)

    # Run analysis
    result = run_flnm_analysis()

    if result is None:
        print("\n❌ Analysis failed.")
        sys.exit(1)

    # Display results
    display_results(result)

    # Save results if output directory is configured
    if OUTPUT_DIR != "/path/to/output/directory":
        save_results(result, OUTPUT_DIR)
    else:
        print("\n💡 TIP: Set OUTPUT_DIR in the configuration to save results.")

    print("\n" + "=" * 70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
