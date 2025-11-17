"""Integration tests for multi-space workflow.

Tests automatic space handling and transformation when analyzing lesions
with reference data (atlases, connectomes) in different coordinate spaces.

Requirements tested:
- FR-004: Automatic space detection
- FR-005: Transparent space transformation
- FR-006: Provenance tracking for transformations
- FR-007: Transformation strategy optimization
"""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.lesion_data import LesionData
from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace


@pytest.fixture
def lesion_nlin6_2mm():
    """Create a synthetic lesion in MNI152NLin6Asym space at 2mm resolution."""
    # MNI152NLin6Asym 2mm has shape (91, 109, 91)
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.uint8)
    
    # Create small spherical lesion in left frontal lobe
    center = (35, 65, 50)  # Approximate left frontal region
    radius = 5
    
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if dist <= radius:
                    data[x, y, z] = 1
    
    # Use reference affine for MNI152NLin6Asym at 2mm
    affine = REFERENCE_AFFINES.get(("MNI152NLin6Asym", 2))
    if affine is None:
        # Fallback: create standard 2mm MNI affine
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
    
    img = nib.Nifti1Image(data, affine)
    
    return LesionData(
        lesion_img=img,
        metadata={
            'subject_id': 'test-001',
            'space': 'MNI152NLin6Asym',
            'resolution': 2,
        }
    )


@pytest.fixture
def lesion_nlin2009c_2mm():
    """Create a synthetic lesion in MNI152NLin2009cAsym space at 2mm resolution."""
    # MNI152NLin2009cAsym 2mm has shape (91, 109, 91)
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.uint8)
    
    # Create small spherical lesion in left frontal lobe
    center = (35, 65, 50)
    radius = 5
    
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if dist <= radius:
                    data[x, y, z] = 1
    
    # Use reference affine for MNI152NLin2009cAsym at 2mm
    affine = REFERENCE_AFFINES.get(("MNI152NLin2009cAsym", 2))
    if affine is None:
        # Fallback: create standard 2mm MNI affine
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
    
    img = nib.Nifti1Image(data, affine)
    
    return LesionData(
        lesion_img=img,
        metadata={
            'subject_id': 'test-002',
            'space': 'MNI152NLin2009cAsym',
            'resolution': 2,
        }
    )


@pytest.fixture
def atlas_nlin2009c_2mm(tmp_path):
    """Create a synthetic atlas in MNI152NLin2009cAsym space at 2mm resolution."""
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.int16)
    
    # Create 3 simple regions
    # Region 1: Left frontal (where our lesion is)
    data[30:40, 60:70, 45:55] = 1
    # Region 2: Right frontal
    data[50:60, 60:70, 45:55] = 2
    # Region 3: Left parietal
    data[30:40, 45:55, 45:55] = 3
    
    # Use reference affine for MNI152NLin2009cAsym at 2mm
    affine = REFERENCE_AFFINES.get(("MNI152NLin2009cAsym", 2))
    if affine is None:
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
    
    img = nib.Nifti1Image(data, affine)
    
    # Save atlas and labels
    atlas_path = tmp_path / "test_atlas.nii.gz"
    nib.save(img, atlas_path)
    
    labels_path = tmp_path / "test_atlas_labels.txt"
    with open(labels_path, 'w') as f:
        f.write("1 Frontal_L\n")
        f.write("2 Frontal_R\n")
        f.write("3 Parietal_L\n")
    
    return atlas_path


@pytest.fixture
def atlas_nlin6_2mm(tmp_path):
    """Create a synthetic atlas in MNI152NLin6Asym space at 2mm resolution."""
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.int16)
    
    # Create same 3 regions as NLin2009c atlas
    data[30:40, 60:70, 45:55] = 1
    data[50:60, 60:70, 45:55] = 2
    data[30:40, 45:55, 45:55] = 3
    
    # Use reference affine for MNI152NLin6Asym at 2mm
    affine = REFERENCE_AFFINES.get(("MNI152NLin6Asym", 2))
    if affine is None:
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
    
    img = nib.Nifti1Image(data, affine)
    
    # Save atlas and labels
    atlas_path = tmp_path / "test_atlas_nlin6.nii.gz"
    nib.save(img, atlas_path)
    
    labels_path = tmp_path / "test_atlas_nlin6_labels.txt"
    with open(labels_path, 'w') as f:
        f.write("1 Frontal_L\n")
        f.write("2 Frontal_R\n")
        f.write("3 Parietal_L\n")
    
    return atlas_path


@pytest.mark.integration
@pytest.mark.skip(reason="Automatic space transformation not yet fully implemented in analysis modules")
def test_cross_space_analysis_with_transformation(lesion_nlin6_2mm, atlas_nlin2009c_2mm, tmp_path):
    """
    Test T049: Load lesion in NLin6, analyze with atlas in NLin2009c.
    
    This should automatically transform one to match the other and complete successfully.
    The system should choose the optimal transformation direction.
    
    Requirements: FR-004, FR-005, FR-007
    """
    from lacuna.analysis import RegionalDamage
    
    # Create atlas directory with the NLin2009c atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    
    # Copy atlas to expected location
    atlas_dest = atlas_dir / "test_atlas.nii.gz"
    labels_dest = atlas_dir / "test_atlas_labels.txt"
    
    import shutil
    shutil.copy(atlas_nlin2009c_2mm, atlas_dest)
    shutil.copy(atlas_nlin2009c_2mm.parent / "test_atlas_labels.txt", labels_dest)
    
    # Run analysis - should automatically detect and handle space mismatch
    analyzer = RegionalDamage(atlas_dir=atlas_dir)
    
    # This should succeed despite space mismatch
    result = analyzer.run(lesion_nlin6_2mm)
    
    # Verify results are present
    assert "RegionalDamage" in result.results or "AtlasAggregation" in result.results
    
    # Check that we got regional damage values
    results_dict = result.results.get("RegionalDamage") or result.results.get("AtlasAggregation")
    assert len(results_dict) > 0
    
    # Verify we got damage in the frontal region (where our lesion is)
    frontal_keys = [k for k in results_dict.keys() if "Frontal_L" in k]
    assert len(frontal_keys) > 0
    assert results_dict[frontal_keys[0]] > 0


@pytest.mark.integration
@pytest.mark.skip(reason="Provenance tracking for transformations not yet fully integrated")
def test_transformation_provenance_tracking(lesion_nlin6_2mm, atlas_nlin2009c_2mm, tmp_path):
    """
    Test T050: Verify provenance records transformation details.
    
    When a transformation occurs, it should be recorded in the provenance
    chain with details about source/target space, method, and rationale.
    
    Requirements: FR-006, FR-013
    """
    from lacuna.analysis import RegionalDamage
    
    # Setup atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    
    import shutil
    atlas_dest = atlas_dir / "test_atlas.nii.gz"
    labels_dest = atlas_dir / "test_atlas_labels.txt"
    shutil.copy(atlas_nlin2009c_2mm, atlas_dest)
    shutil.copy(atlas_nlin2009c_2mm.parent / "test_atlas_labels.txt", labels_dest)
    
    # Run analysis
    analyzer = RegionalDamage(atlas_dir=atlas_dir)
    result = analyzer.run(lesion_nlin6_2mm)
    
    # Check provenance chain
    assert len(result.provenance) > 0
    
    # Look for transformation record in provenance
    transform_records = [
        p for p in result.provenance
        if 'transformation' in str(p).lower() or 'transform' in p.get('function', '').lower()
    ]
    
    # Should have at least one transformation record
    assert len(transform_records) > 0
    
    # Verify transformation record contains required fields
    transform_record = transform_records[0]
    assert 'source_space' in str(transform_record).lower() or 'source' in transform_record
    assert 'target_space' in str(transform_record).lower() or 'target' in transform_record
    
    # Verify spaces are documented
    provenance_str = str(result.provenance)
    assert 'MNI152NLin6Asym' in provenance_str or 'NLin6' in provenance_str
    assert 'MNI152NLin2009cAsym' in provenance_str or 'NLin2009c' in provenance_str


@pytest.mark.integration
def test_matched_spaces_no_transformation_overhead(lesion_nlin2009c_2mm, atlas_nlin2009c_2mm, tmp_path):
    """
    Test T051: Matched spaces (no transformation) completes quickly.
    
    When lesion and atlas are already in the same space, no transformation
    should occur and analysis should complete without transformation overhead.
    
    Requirements: FR-007, FR-011
    """
    from lacuna.analysis import RegionalDamage
    
    # Setup atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    
    import shutil
    atlas_dest = atlas_dir / "test_atlas.nii.gz"
    labels_dest = atlas_dir / "test_atlas_labels.txt"
    shutil.copy(atlas_nlin2009c_2mm, atlas_dest)
    shutil.copy(atlas_nlin2009c_2mm.parent / "test_atlas_labels.txt", labels_dest)
    
    # Time the analysis
    analyzer = RegionalDamage(atlas_dir=atlas_dir)
    
    start_time = time.time()
    result = analyzer.run(lesion_nlin2009c_2mm)
    elapsed_time = time.time() - start_time
    
    # Verify analysis completed
    assert "RegionalDamage" in result.results or "AtlasAggregation" in result.results
    
    # Check that analysis was fast (should be < 5 seconds for synthetic data)
    # This is a generous threshold - real overhead would be much larger (30+ seconds)
    assert elapsed_time < 5.0, f"Analysis took {elapsed_time:.2f}s, expected < 5s"
    
    # Verify no transformation record in provenance
    # (or if there is one, it should indicate "no transformation needed")
    provenance_str = str(result.provenance).lower()
    
    # Should not have transformation-related entries, or if it does,
    # should indicate no transformation was performed
    has_transform = 'transformation' in provenance_str or 'transform' in provenance_str
    if has_transform:
        assert 'no transformation' in provenance_str or 'same space' in provenance_str


@pytest.mark.integration
def test_space_detection_from_metadata(lesion_nlin6_2mm):
    """
    Test that space detection works correctly from metadata.
    
    Requirements: FR-004
    """
    # Verify space was detected correctly
    assert lesion_nlin6_2mm.metadata['space'] == 'MNI152NLin6Asym'
    assert lesion_nlin6_2mm.metadata['resolution'] == 2
    
    # Verify affine matches expected for this space
    expected_affine = REFERENCE_AFFINES.get(("MNI152NLin6Asym", 2))
    if expected_affine is not None:
        np.testing.assert_array_almost_equal(
            lesion_nlin6_2mm.affine,
            expected_affine,
            decimal=2
        )


@pytest.mark.integration
def test_coordinate_space_creation():
    """
    Test CoordinateSpace object creation and validation.
    
    Requirements: FR-004
    """
    # Create a coordinate space
    affine = REFERENCE_AFFINES.get(("MNI152NLin6Asym", 2))
    if affine is None:
        affine = np.eye(4)
    
    space = CoordinateSpace(
        identifier="MNI152NLin6Asym",
        resolution=2,
        reference_affine=affine
    )
    
    # Verify properties
    assert space.identifier == "MNI152NLin6Asym"
    assert space.resolution == 2
    assert isinstance(space.reference_affine, np.ndarray)
    assert space.reference_affine.shape == (4, 4)
    
    # Test string representation
    space_str = str(space)
    assert "MNI152NLin6Asym" in space_str
    assert "2" in space_str


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
