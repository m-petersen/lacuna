"""
Demonstration of Multi-Space Neuroimaging Architecture

This example shows how to:
1. Load a lesion image with automatic space detection
2. Download templates and transforms from TemplateFlow
3. Transform between coordinate spaces (NLin6Asym <-> NLin2009cAsym)
4. Process with automatic space alignment
5. Track provenance of all transformations

Usage:
    python examples/space_transformation_demo.py /path/to/your/lesion.nii.gz

Requirements:
    - nibabel
    - nitransforms
    - templateflow (optional, for automatic downloads)
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def create_synthetic_lesion(output_path: Path) -> Path:
    """Create a synthetic lesion mask in MNI152NLin6Asym space for testing.

    Args:
        output_path: Path to save the synthetic lesion

    Returns:
        Path to created lesion file
    """
    print("Creating synthetic lesion mask in MNI152NLin6Asym space (2mm)...")

    # MNI152NLin6Asym 2mm affine
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Standard MNI152 2mm dimensions
    shape = (91, 109, 91)

    # Create binary lesion mask with a small sphere
    data = np.zeros(shape, dtype=np.uint8)

    # Add spherical lesion in left motor cortex region (approximate coordinates)
    center = (35, 50, 55)  # Voxel coordinates
    radius = 5

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2)
                if dist <= radius:
                    data[i, j, k] = 1

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)

    # Save
    nib.save(img, output_path)
    print(f"  ✓ Created synthetic lesion: {output_path}")
    print("  ✓ Space: MNI152NLin6Asym")
    print("  ✓ Resolution: 2mm")
    print(f"  ✓ Lesion volume: {np.sum(data)} voxels ({np.sum(data) * 8} mm³)")

    return output_path


def demo_space_detection(lesion_path: Path) -> None:
    """Demonstrate automatic space detection from filename and header."""
    print_section("1. Automatic Space Detection")

    from lacuna.core.spaces import (
        detect_space_from_filename,
        detect_space_from_header,
        get_image_space,
    )

    # Load image
    img = nib.load(lesion_path)

    # Method 1: From filename
    print("Method 1: Detection from filename")
    try:
        space_from_name = detect_space_from_filename(lesion_path)
        print(f"  ✓ Detected from filename: {space_from_name}")
    except Exception as e:
        print(f"  ✗ Could not detect from filename: {e}")

    # Method 2: From header/affine
    print("\nMethod 2: Detection from image header")
    space_from_header = detect_space_from_header(img)
    print(f"  ✓ Detected from header: {space_from_header}")

    # Method 3: Unified detection
    print("\nMethod 3: Unified detection (get_image_space)")
    detected_space = get_image_space(img, filepath=lesion_path)
    print(f"  ✓ Detected space: {detected_space.identifier}")
    print(f"  ✓ Resolution: {detected_space.resolution}mm")
    print("  ✓ Reference affine:")
    print(f"    {detected_space.reference_affine}")


def demo_mask_data_loading(lesion_path: Path) -> "MaskData":
    """Demonstrate loading lesion data with automatic space detection."""
    print_section("2. Loading MaskData with Automatic Space Detection")

    from lacuna.core.mask_data import MaskData

    # Load with automatic space detection
    print("Loading lesion data...")
    mask_data = MaskData.from_nifti(lesion_path, metadata={"subject_id": "demo-001"})

    print("  ✓ Loaded successfully")
    print(f"  ✓ Subject ID: {mask_data.metadata['subject_id']}")
    print(f"  ✓ Detected space: {mask_data.metadata['space']}")
    print(f"  ✓ Resolution: {mask_data.metadata['resolution']}mm")
    print(f"  ✓ Lesion volume: {mask_data.get_volume_mm3():.1f} mm³")

    return mask_data


def demo_query_capabilities() -> None:
    """Demonstrate querying supported spaces and transforms."""
    print_section("3. Query Supported Spaces and Transforms")

    from lacuna.core.spaces import REFERENCE_AFFINES, query_supported_spaces
    from lacuna.spatial.transform import query_available_transforms

    # Query supported spaces
    print("Supported coordinate spaces:")
    spaces = query_supported_spaces()
    for space in spaces:
        print(f"  • {space}")

    # Query available transforms
    print("\nAvailable transformations:")
    transforms = query_available_transforms()
    for source, target in transforms:
        print(f"  • {source:25s} → {target}")

    # Query supported resolutions
    print("\nSupported space/resolution combinations:")
    for (space, res), _affine in REFERENCE_AFFINES.items():
        print(f"  • {space:25s} @ {res}mm")


def demo_transform_validation(mask_data: "MaskData") -> None:
    """Demonstrate space validation and transform availability checking."""
    print_section("4. Space Validation and Transform Availability")

    from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace, SpaceValidator
    from lacuna.spatial.transform import can_transform_between

    # Get source space
    source_space = CoordinateSpace(
        identifier=mask_data.metadata["space"],
        resolution=mask_data.metadata["resolution"],
        reference_affine=REFERENCE_AFFINES[
            (mask_data.metadata["space"], mask_data.metadata["resolution"])
        ],
    )

    # Define target space
    target_space = CoordinateSpace(
        identifier="MNI152NLin2009cAsym",
        resolution=2,
        reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
    )

    print(f"Source space: {source_space.identifier} @ {source_space.resolution}mm")
    print(f"Target space: {target_space.identifier} @ {target_space.resolution}mm")

    # Check if transformation is possible
    can_transform = can_transform_between(source_space, target_space)
    print(f"\nCan transform between spaces: {'✓ YES' if can_transform else '✗ NO'}")

    # Use validator
    validator = SpaceValidator()

    # Validate space declaration
    is_valid = validator.validate_space_declaration(
        source_space,
        mask_data.mask_img,  # Property access, not method call
    )
    print(f"Space declaration valid: {'✓ YES' if is_valid else '✗ NO'}")

    # Check if spaces differ
    spaces_differ = validator.detect_mismatch(source_space, target_space)
    print(f"Spaces differ: {'✓ YES' if spaces_differ else '✗ NO'}")


def demo_transformation(mask_data: "MaskData") -> "MaskData":
    """Demonstrate spatial transformation between coordinate spaces."""
    print_section("5. Spatial Transformation")

    from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
    from lacuna.spatial.transform import TransformationStrategy, transform_mask_data

    # Define target space
    target_space = CoordinateSpace(
        identifier="MNI152NLin2009cAsym",
        resolution=2,
        reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
    )

    print(f"Source: {mask_data.metadata['space']} @ {mask_data.metadata['resolution']}mm")
    print(f"Target: {target_space.identifier} @ {target_space.resolution}mm")

    # Show transformation strategy
    strategy = TransformationStrategy()
    source_space = CoordinateSpace(
        identifier=mask_data.metadata["space"],
        resolution=mask_data.metadata["resolution"],
        reference_affine=REFERENCE_AFFINES[
            (mask_data.metadata["space"], mask_data.metadata["resolution"])
        ],
    )

    direction = strategy.determine_direction(source_space, target_space)
    print(f"\nTransformation direction: {direction}")

    # Determine interpolation method
    interp_method = strategy.select_interpolation(mask_data.mask_img)
    print(f"Interpolation method: {interp_method.value} (auto-detected from data)")

    # Check if we can actually transform
    print("\n" + "=" * 60)
    print("Attempting actual transformation...")
    print("=" * 60)

    try:
        # Try to perform the actual transformation
        print("\nStep 1: Loading transform from lacuna.assets...")
        print("  (This may download from TemplateFlow on first use)")

        transformed_data = transform_mask_data(mask_data, target_space)

        print("  ✓ Transform loaded successfully")
        print("  ✓ Transformation applied")
        print("  ✓ New MaskData created")
        print("  ✓ Provenance record added")

        # Show results
        print("\nTransformation Results:")
        print(f"  Original volume: {mask_data.get_volume_mm3():.1f} mm³")
        print(f"  Transformed volume: {transformed_data.get_volume_mm3():.1f} mm³")
        print(f"  Original space: {mask_data.metadata['space']}")
        print(f"  Transformed space: {transformed_data.metadata['space']}")
        print(
            f"  Provenance records: {len(mask_data.provenance)} → {len(transformed_data.provenance)}"
        )

        return transformed_data

    except ImportError as e:
        print(f"\n⚠️  Cannot perform transformation: {e}")
        print("\nMissing dependencies. Please install:")
        print("  pip install nitransforms templateflow")
        print("\nShowing API structure instead...")
        print("\nAPI call for transformation:")
        print("  transformed = transform_mask_data(mask_data, target_space)")
        return mask_data

    except FileNotFoundError as e:
        print(f"\n⚠️  Transform file not available: {e}")
        print("\nTransform files are downloaded from TemplateFlow on first use.")
        print("Make sure you have an internet connection for the initial download.")
        print("\nTransform files (~200MB) are cached at ~/.cache/templateflow/")
        print("\nAPI call demonstrated:")
        print("  transformed = transform_mask_data(mask_data, target_space)")
        return mask_data

    except Exception as e:
        print(f"\n⚠️  Transformation failed: {e}")
        print("\nThis might be due to:")
        print("  - Missing transform files (need TemplateFlow integration)")
        print("  - Network issues during download")
        print("  - Unsupported transformation pair")
        print("\nAPI call for transformation:")
        print("  transformed = transform_mask_data(mask_data, target_space)")
        return mask_data


def demo_provenance(mask_data: "MaskData") -> None:
    """Demonstrate provenance tracking."""
    print_section("6. Provenance Tracking")

    from datetime import datetime, timezone

    from lacuna.core.provenance import TransformationRecord

    # Show current provenance
    provenance = mask_data.provenance  # Property access
    print(f"Current provenance records: {len(provenance)}")

    if provenance:
        print("\nActual transformation record:")
        for i, record in enumerate(provenance, 1):
            print(f"\n  Record {i}:")
            for key, value in record.items():
                if isinstance(value, str) and len(value) > 60:
                    print(f"    {key}: {value[:57]}...")
                else:
                    print(f"    {key}: {value}")
    else:
        print("\n  (No transformations performed yet)")

    # Show example TransformationRecord structure
    print("\n" + "-" * 60)
    print("Example TransformationRecord structure:")
    print("-" * 60)
    example_record = TransformationRecord(
        source_space="MNI152NLin6Asym",
        source_resolution=2,
        target_space="MNI152NLin2009cAsym",
        target_resolution=2,
        method="nitransforms",
        interpolation="nearest",
        timestamp=datetime.now(timezone.utc).isoformat(),
        rationale="Automatic transformation for atlas alignment",
        transform_file="tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
    )

    record_dict = example_record.to_dict()
    print("\nTransformationRecord fields:")
    for key, value in record_dict.items():
        print(f"  {key}: {value}")


def demo_analysis_integration(mask_data: "MaskData") -> None:
    """Demonstrate how analysis modules use automatic space alignment."""
    print_section("7. Analysis Module Integration")

    print("Analysis modules can now automatically handle space alignment:")
    print()

    # Show the helper method
    print("Example usage in analysis module:")
    print(
        """
    class MyAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data: MaskData) -> MaskData:
            # Automatically transform to required space if needed
            mask_data = self._validate_and_transform_space(
                mask_data,
                required_space='MNI152NLin2009cAsym',
                required_resolution=2
            )
            return mask_data
    """
    )

    print("\nThis helper method:")
    print("  ✓ Checks current space from metadata")
    print("  ✓ Determines if transformation is needed")
    print("  ✓ Automatically transforms if spaces differ")
    print("  ✓ Returns original data if already in required space")
    print("  ✓ Adds transformation to provenance")


def demo_asset_management() -> None:
    """Demonstrate data asset management."""
    print_section("8. Data Asset Management")

    from lacuna.assets import load_template, load_transform

    print("lacuna.assets handles:")
    print("  • Transform files (~200MB each)")
    print("  • MNI templates (various resolutions)")
    print("  • Atlas files (from TemplateFlow)")
    print()

    # Check for transforms
    print("Checking transform availability:")
    try:
        transform_path = load_transform("MNI152NLin6Asym_to_MNI152NLin2009cAsym")
        if transform_path:
            print(f"  ✓ Transform found: {transform_path}")
        else:
            print("  ⚠️  Transform not yet available (would download from TemplateFlow)")
    except Exception as e:
        print(f"  ⚠️  {e}")

    # Check for templates
    print("\nChecking template availability:")
    try:
        template_path = load_template("MNI152NLin6Asym_res-2")
        if template_path:
            print(f"  ✓ Template found: {template_path}")
        else:
            print("  ⚠️  Template not yet available")
    except Exception as e:
        print(f"  ℹ️  {e}")

    # Convenience function
    print("\nConvenience function for transform retrieval:")
    print("  transform_path = get_transform_path('MNI152NLin6Asym', 'MNI152NLin2009cAsym')")


def main():
    """Run the complete demonstration."""
    print_section("Multi-Space Neuroimaging Architecture Demo")

    # Check for input file
    if len(sys.argv) > 1:
        lesion_path = Path(sys.argv[1])
        if not lesion_path.exists():
            print(f"Error: File not found: {lesion_path}")
            print("Creating synthetic lesion instead...")
            lesion_path = Path("/tmp/synthetic_lesion_NLin6_2mm.nii.gz")
            create_synthetic_lesion(lesion_path)
    else:
        print("No input file specified. Creating synthetic lesion...")
        lesion_path = Path("/tmp/synthetic_lesion_NLin6_2mm.nii.gz")
        create_synthetic_lesion(lesion_path)

    try:
        # Run demonstrations
        demo_space_detection(lesion_path)
        mask_data = demo_mask_data_loading(lesion_path)
        demo_query_capabilities()
        demo_transform_validation(mask_data)
        transformed_data = demo_transformation(mask_data)

        # Show provenance of the transformed data (or original if transformation failed)
        demo_provenance(transformed_data)
        demo_analysis_integration(transformed_data)
        demo_asset_management()

        print_section("Demo Complete!")
        print("Key Features Demonstrated:")
        print("  ✓ Automatic space detection from filename and header")
        print("  ✓ Space validation and transform availability checking")
        print("  ✓ Query APIs for supported spaces and transforms")
        print("  ✓ Transformation strategy selection (direction, interpolation)")

        # Check if transformation was successful
        if len(transformed_data.provenance) > 0:
            print("  ✓ Actual spatial transformation performed!")
            print("  ✓ Transform downloaded from TemplateFlow")
        else:
            print("  ⚠ Transformation API demonstrated (requires nitransforms/templateflow)")

        print("  ✓ Provenance tracking with TransformationRecord")
        print("  ✓ Analysis module integration pattern")
        print("  ✓ Data asset management (TemplateFlow integration)")
        print()

        if len(transformed_data.provenance) > 0:
            print("Success! Full transformation pipeline completed.")
            print(f"Original space: {mask_data.metadata['space']}")
            print(f"Transformed to: {transformed_data.metadata['space']}")
        else:
            print("Next Steps:")
            print("  • Ensure nitransforms is installed: pip install nitransforms")
            print("  • Ensure templateflow is installed: pip install templateflow")
            print("  • Transform download will happen automatically on first use")
        print("  • Try with your own lesion data!")
        print()

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
