#!/usr/bin/env python3
"""
Batch processing test for Functional Lesion Network Mapping (fLNM).

This script demonstrates VECTORIZED batch processing which is 10-50x faster
than sequential processing. It automatically uses the VectorizedStrategy
when processing multiple lesions.

Configure the paths below to point to your data directories.
"""

import sys
import time
from pathlib import Path

import nibabel as nib

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR DATA
# ============================================================================

# Directory containing lesion masks (all .nii.gz files will be processed)
LESION_DIR = "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/raw/lesion_masks/acuteinfarct_test/"

# Path to functional connectome (single file or directory with batches)
CONNECTOME_PATH = "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/connectomes/gsp1000_batches_100"

# Directory to save output results
OUTPUT_DIR = (
    "/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY/data/processed/fLNM_LDK"
)

# Analysis parameters
METHOD = "boes"  # Options: "boes" or "pini"
PINI_PERCENTILE = 20  # Only used if method="pini"
COMPUTE_T_MAP = True  # Compute t-statistic maps
T_THRESHOLD = 9  # Create binary threshold map (|t| > threshold)
VERBOSE = True  # Show detailed progress

# Batch processing mode
USE_VECTORIZED = True  # Use vectorized batch processing (10-50x faster!)
LESION_BATCH_SIZE = (
    50  # Process 50 lesions at a time to control memory usage
    # None = process all together (fastest but high memory)
    # 10-50 = balanced speed/memory for 50-200 lesions
    # Adjust based on available RAM: ~2-4 GB per batch of 20 lesions
)

# ============================================================================
# BATCH PROCESSING SCRIPT
# ============================================================================


def find_lesion_files(lesion_dir):
    """Find all NIfTI lesion mask files in directory."""
    lesion_dir = Path(lesion_dir)
    lesion_files = sorted(lesion_dir.glob("*.nii.gz"))

    if not lesion_files:
        lesion_files = sorted(lesion_dir.glob("*.nii"))

    return lesion_files


def get_subject_id(lesion_path):
    """Extract subject ID from filename."""
    # Remove extensions
    name = lesion_path.name
    for ext in [".nii.gz", ".nii"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def save_results(subject_id, result, output_dir, verbose=True):
    """Save analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flnm_results = result.results.get("FunctionalNetworkMapping", {})

    saved_files = []

    # Save correlation map (r-map)
    if "correlation_map" in flnm_results:
        output_path = output_dir / f"{subject_id}_rmap.nii.gz"
        nib.save(flnm_results["correlation_map"], output_path)
        saved_files.append("rmap")

    # Save z-map
    if "z_map" in flnm_results:
        output_path = output_dir / f"{subject_id}_zmap.nii.gz"
        nib.save(flnm_results["z_map"], output_path)
        saved_files.append("zmap")

    # Save t-map
    if "t_map" in flnm_results:
        output_path = output_dir / f"{subject_id}_tmap.nii.gz"
        nib.save(flnm_results["t_map"], output_path)
        saved_files.append("tmap")

    # Save thresholded map
    if "t_threshold_map" in flnm_results:
        output_path = output_dir / f"{subject_id}_tthresh.nii.gz"
        nib.save(flnm_results["t_threshold_map"], output_path)
        saved_files.append("tthresh")

    # Save summary statistics
    if "summary_statistics" in flnm_results:
        import json

        output_path = output_dir / f"{subject_id}_stats.json"
        with open(output_path, "w") as f:
            json.dump(flnm_results["summary_statistics"], f, indent=2)
        saved_files.append("stats")

    if verbose:
        print(f"  💾 Saved: {', '.join(saved_files)}")


def main():
    """Main batch processing function."""
    print("=" * 70)
    print("FUNCTIONAL NETWORK MAPPING - VECTORIZED BATCH PROCESSING")
    print("=" * 70)

    # Validate configuration
    if LESION_DIR == "/path/to/lesion_masks/":
        print("\n❌ ERROR: Please configure the paths in this script first!")
        print("   Edit the CONFIGURATION section at the top of the file.")
        sys.exit(1)

    # Check paths exist
    lesion_dir = Path(LESION_DIR)
    connectome_path = Path(CONNECTOME_PATH)
    output_dir = Path(OUTPUT_DIR)

    if not lesion_dir.exists():
        print(f"❌ ERROR: Lesion directory not found: {lesion_dir}")
        sys.exit(1)

    if not connectome_path.exists():
        print(f"❌ ERROR: Connectome not found: {connectome_path}")
        sys.exit(1)

    # Find lesion files
    lesion_files = find_lesion_files(lesion_dir)
    if not lesion_files:
        print(f"❌ ERROR: No lesion files found in {lesion_dir}")
        sys.exit(1)

    print("\n📁 Configuration:")
    print(f"  - Lesion directory: {lesion_dir}")
    print(f"  - Connectome: {connectome_path}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Method: {METHOD}")
    if METHOD == "pini":
        print(f"  - PINI percentile: {PINI_PERCENTILE}")
    print(f"  - Compute t-map: {COMPUTE_T_MAP}")
    if T_THRESHOLD is not None:
        print(f"  - T-threshold: {T_THRESHOLD}")
    print(f"  - Vectorized processing: {USE_VECTORIZED}")
    if USE_VECTORIZED:
        if LESION_BATCH_SIZE is not None:
            print(f"  - Lesion batch size: {LESION_BATCH_SIZE} (memory-controlled)")
        else:
            print("  - Lesion batch size: ALL (process all together - high memory)")
    print(f"\n📊 Found {len(lesion_files)} lesion files to process")

    # Import Lacuna
    try:
        from lacuna import LesionData
        from lacuna.analysis import FunctionalNetworkMapping
        from lacuna.batch import batch_process

        print("✓ Successfully imported ldk modules")
    except ImportError as e:
        print(f"❌ Failed to import lacuna: {e}")
        print("   Make sure the package is installed: pip install -e .")
        sys.exit(1)

    # Create analysis object
    print("\n🔧 Creating FunctionalNetworkMapping analysis...")
    analysis = FunctionalNetworkMapping(
        connectome_path=str(connectome_path),
        method=METHOD,
        pini_percentile=PINI_PERCENTILE,
        verbose=VERBOSE,
        compute_t_map=COMPUTE_T_MAP,
        t_threshold=T_THRESHOLD,
    )
    print(f"  ✓ Analysis created (batch_strategy = {analysis.batch_strategy})")

    # Load all lesions
    print(f"\n📥 Loading {len(lesion_files)} lesions...")
    lesions = []
    skipped = []

    for lesion_path in lesion_files:
        subject_id = get_subject_id(lesion_path)

        # Check if already processed
        output_rmap = output_dir / f"{subject_id}_rmap.nii.gz"
        if output_rmap.exists():
            skipped.append(subject_id)
            continue

        try:
            lesion = LesionData.from_nifti(
                str(lesion_path),
                metadata={"subject_id": subject_id, "space": "MNI152_2mm"},
            )
            lesions.append(lesion)
        except Exception as e:
            print(f"  ⚠️  Failed to load {subject_id}: {e}")

    if skipped:
        print(f"  ⏭️  Skipped {len(skipped)} already processed")

    if not lesions:
        print("\n✅ All lesions already processed. Exiting.")
        sys.exit(0)

    print(f"  ✓ Loaded {len(lesions)} lesions to process")

    # Process with vectorized batch processing
    print(f"\n{'=' * 70}")
    print("RUNNING VECTORIZED BATCH PROCESSING")
    print(f"{'=' * 70}")

    start_time = time.time()

    # Define callback to save results immediately after each batch
    def save_batch_results(batch_results):
        """Save results immediately to free memory."""
        for result in batch_results:
            subject_id = result.metadata.get("subject_id", "unknown")
            save_results(subject_id, result, output_dir, verbose=False)

    if USE_VECTORIZED:
        # Use explicit vectorized strategy with optional batch size
        results = batch_process(
            lesions,
            analysis,
            strategy="vectorized",  # Force vectorized
            lesion_batch_size=LESION_BATCH_SIZE,  # Process lesions in batches
            batch_result_callback=save_batch_results
            if LESION_BATCH_SIZE
            else None,  # Save immediately
            show_progress=True,
        )
    else:
        # Fall back to parallel processing (slower)
        results = batch_process(
            lesions,
            analysis,
            n_jobs=-1,  # Use all cores
            show_progress=True,
        )

    elapsed = time.time() - start_time

    # Save any remaining results (if callback wasn't used)
    if not LESION_BATCH_SIZE or not USE_VECTORIZED:
        print(f"\n{'=' * 70}")
        print("SAVING RESULTS")
        print(f"{'=' * 70}")

        for result in results:
            subject_id = result.metadata.get("subject_id", "unknown")
            save_results(subject_id, result, output_dir, verbose=VERBOSE)
    else:
        # Results were already saved by callback
        print(f"\n{'=' * 70}")
        print("RESULTS SAVED INCREMENTALLY")
        print(f"{'=' * 70}")
        print(f"✓ Results saved after each batch of {LESION_BATCH_SIZE} lesions")

    # Print summary
    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✅ Successfully processed: {len(results)} lesions")
    if skipped:
        print(f"⏭️  Skipped (already done): {len(skipped)} lesions")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"⚡ Average time per lesion: {elapsed / len(results):.1f} seconds")
    print(f"💾 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
