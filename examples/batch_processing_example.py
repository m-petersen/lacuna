#!/usr/bin/env python3
"""
Batch Processing Example Script

This script demonstrates the batch processing functionality without Jupyter
notebook limitations. Use this for production batch processing workflows.

Unlike Jupyter notebooks, standalone Python scripts don't have pickling issues
with parallel processing, so this will leverage all CPU cores efficiently.

Usage:
    python examples/batch_processing_example.py --lesion-dir /path/to/lesions --output-dir /path/to/output

"""

import argparse
import time
from pathlib import Path

from ldk import LesionData, batch_process
from ldk.analysis import AtlasAggregation, RegionalDamage
from ldk.io import batch_export_to_csv, batch_export_to_tsv


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple lesion masks with parallel processing"
    )
    parser.add_argument(
        "--lesion-dir",
        type=Path,
        required=True,
        help="Directory containing lesion NIfTI files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nii.gz",
        help="Glob pattern for lesion files (default: *.nii.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/batch_results").expanduser(),
        help="Output directory for results (default: ~/batch_results)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = use all cores, 1 = sequential)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of subjects to process (for testing)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["regional", "atlas"],
        default="regional",
        help="Type of analysis to run",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.lesion_dir.exists():
        print(f"❌ Error: Directory not found: {args.lesion_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find lesion files
    lesion_paths = list(args.lesion_dir.glob(args.pattern))
    if args.limit:
        lesion_paths = lesion_paths[: args.limit]

    if not lesion_paths:
        print(f"❌ Error: No files matching '{args.pattern}' found in {args.lesion_dir}")
        return 1

    print("=" * 70)
    print("BATCH PROCESSING CONFIGURATION")
    print("=" * 70)
    print(f"Lesion directory: {args.lesion_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Found: {len(lesion_paths)} lesion files")
    print(f"Analysis: {args.analysis}")
    print(
        f"Parallel workers: {args.n_jobs} ({'all cores' if args.n_jobs == -1 else 'sequential' if args.n_jobs == 1 else f'{args.n_jobs} cores'})"
    )
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    print()

    # Load lesion data
    print("Loading lesion files...")
    lesions = []
    failed_loads = []

    for lesion_path in lesion_paths:
        try:
            # Extract subject ID from filename
            subject_id = lesion_path.stem.split("_")[0]

            # Load lesion
            lesion = LesionData.from_nifti(lesion_path)
            lesion.metadata["subject_id"] = subject_id

            lesions.append(lesion)
            print(f"  ✓ {subject_id}")
        except Exception as e:
            failed_loads.append((lesion_path, str(e)))
            print(f"  ✗ {lesion_path.name}: {e}")

    print()
    print(f"Successfully loaded: {len(lesions)} subjects")
    if failed_loads:
        print(f"Failed to load: {len(failed_loads)} subjects")
    print()

    if not lesions:
        print("❌ Error: No lesions successfully loaded")
        return 1

    # Create analysis instance
    if args.analysis == "regional":
        analysis = RegionalDamage()
    else:
        analysis = AtlasAggregation(method="percent_overlap")

    print(f"Analysis: {analysis.__class__.__name__}")
    print(f"Batch strategy: {analysis.batch_strategy}")
    print()

    # Sequential processing (baseline)
    if args.n_jobs != 1:
        print("-" * 70)
        print("BASELINE: Sequential Processing (n_jobs=1)")
        print("-" * 70)
        start_time = time.time()
        results_sequential = batch_process(
            lesion_data_list=lesions,
            analysis=analysis,
            n_jobs=1,
            show_progress=True,
        )
        sequential_time = time.time() - start_time
        print("\n✓ Sequential processing complete")
        print(f"  Processed: {len(results_sequential)}/{len(lesions)} subjects")
        print(f"  Time: {sequential_time:.2f}s ({sequential_time / len(lesions):.2f}s/subject)")
        print()

    # Parallel processing
    if args.n_jobs != 1:
        print("-" * 70)
        print(f"PARALLEL PROCESSING (n_jobs={args.n_jobs})")
        print("-" * 70)

    start_time = time.time()
    results = batch_process(
        lesion_data_list=lesions,
        analysis=analysis,
        n_jobs=args.n_jobs,
        show_progress=True,
    )
    processing_time = time.time() - start_time

    print("\n✓ Batch processing complete")
    print(f"  Processed: {len(results)}/{len(lesions)} subjects")
    print(f"  Time: {processing_time:.2f}s ({processing_time / len(lesions):.2f}s/subject)")

    if args.n_jobs != 1 and "sequential_time" in locals():
        speedup = sequential_time / processing_time
        print(f"  🚀 Speedup: {speedup:.2f}x faster than sequential")
        print(f"  ⚡ Time saved: {sequential_time - processing_time:.2f}s")

    print()

    # Export results
    print("-" * 70)
    print("EXPORTING RESULTS")
    print("-" * 70)

    # CSV export
    csv_path = args.output_dir / f"batch_{args.analysis}_results.csv"
    batch_export_to_csv(results, csv_path)
    print(f"✓ CSV exported: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024:.1f} KB")

    # TSV export (BIDS-compatible)
    tsv_path = args.output_dir / f"batch_{args.analysis}_results.tsv"
    batch_export_to_tsv(results, tsv_path)
    print(f"✓ TSV exported: {tsv_path}")
    print(f"  Size: {tsv_path.stat().st_size / 1024:.1f} KB")

    print()
    print("=" * 70)
    print("✅ BATCH PROCESSING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
