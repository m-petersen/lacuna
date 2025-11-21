"""
Integration tests for automatic TARGET_SPACE transformation in analyses.

Tests that analyses automatically transform lesions to their declared
TARGET_SPACE and TARGET_RESOLUTION before computation.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis.base import BaseAnalysis


@pytest.fixture
def lesion_mni152_2mm(tmp_path):
    """Create a synthetic lesion in MNI152NLin6Asym @ 2mm."""
    shape = (91, 109, 91)  # Standard MNI152 2mm shape
    data = np.zeros(shape)
    data[45:50, 54:59, 45:50] = 1  # Small lesion

    # MNI152NLin6Asym 2mm affine
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    img = nib.Nifti1Image(data, affine)

    return MaskData(mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})


@pytest.fixture
def lesion_mni152_1mm(tmp_path):
    """Create a synthetic lesion in MNI152NLin6Asym @ 1mm."""
    shape = (182, 218, 182)  # Standard MNI152 1mm shape
    data = np.zeros(shape)
    data[90:100, 108:118, 90:100] = 1  # Small lesion

    # MNI152NLin6Asym 1mm affine
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 90.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    img = nib.Nifti1Image(data, affine)

    return MaskData(mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 1})


class TestAnalysis2mm(BaseAnalysis):
    """Test analysis that requires MNI152NLin6Asym @ 2mm."""

    TARGET_SPACE = "MNI152NLin6Asym"
    TARGET_RESOLUTION = 2

    def _validate_inputs(self, mask_data):
        # Check that transformation happened correctly
        space = mask_data.metadata.get("space")
        resolution = mask_data.metadata.get("resolution")

        if space != self.TARGET_SPACE:
            raise ValueError(f"Expected {self.TARGET_SPACE}, got {space}")
        if resolution != self.TARGET_RESOLUTION:
            raise ValueError(f"Expected {self.TARGET_RESOLUTION}mm, got {resolution}mm")

    def _run_analysis(self, mask_data):
        # Verify shape matches 2mm template
        assert mask_data.mask_img.shape == (
            91,
            109,
            91,
        ), f"Expected 2mm shape (91, 109, 91), got {mask_data.mask_img.shape}"
        return {
            "space": mask_data.metadata.get("space"),
            "resolution": mask_data.metadata.get("resolution"),
        }


class TestAnalysis1mm(BaseAnalysis):
    """Test analysis that requires MNI152NLin2009cAsym @ 1mm."""

    TARGET_SPACE = "MNI152NLin2009cAsym"
    TARGET_RESOLUTION = 1

    def _validate_inputs(self, mask_data):
        space = mask_data.metadata.get("space")
        resolution = mask_data.metadata.get("resolution")

        if space != self.TARGET_SPACE:
            raise ValueError(f"Expected {self.TARGET_SPACE}, got {space}")
        if resolution != self.TARGET_RESOLUTION:
            raise ValueError(f"Expected {self.TARGET_RESOLUTION}mm, got {resolution}mm")

    def _run_analysis(self, mask_data):
        return {
            "space": mask_data.metadata.get("space"),
            "resolution": mask_data.metadata.get("resolution"),
        }


class TestAnalysisNoTarget(BaseAnalysis):
    """Test analysis with no target space (adaptive)."""

    def _validate_inputs(self, mask_data):
        pass

    def _run_analysis(self, mask_data):
        return {
            "space": mask_data.metadata.get("space"),
            "resolution": mask_data.metadata.get("resolution"),
        }


def test_no_transformation_when_already_in_target_space(lesion_mni152_2mm):
    """Test that no transformation occurs if lesion is already in target space."""
    analysis = TestAnalysis2mm()
    result = analysis.run(lesion_mni152_2mm)

    # Should succeed without transformation
    assert result.results["TestAnalysis2mm"]["space"] == "MNI152NLin6Asym"
    assert result.results["TestAnalysis2mm"]["resolution"] == 2


@pytest.mark.slow
@pytest.mark.requires_templateflow
def test_automatic_transformation_to_target_space(lesion_mni152_1mm):
    """Test that lesion is automatically transformed from 1mm to 2mm via resampling."""
    analysis = TestAnalysis2mm()
    result = analysis.run(lesion_mni152_1mm)

    # Should transform from 1mm to 2mm
    assert result.results["TestAnalysis2mm"]["space"] == "MNI152NLin6Asym"
    assert result.results["TestAnalysis2mm"]["resolution"] == 2

    # Verify transformation was recorded in provenance
    transform_records = [r for r in result.provenance if r.get("type") == "spatial_transformation"]
    assert len(transform_records) > 0, "Expected transformation record in provenance"
    assert transform_records[0]["target_space"] == "MNI152NLin6Asym"
    assert transform_records[0]["target_resolution"] == 2
    assert (
        transform_records[0]["method"] == "nilearn_resample"
    )  # Should use resampling not transform


@pytest.mark.slow
@pytest.mark.requires_templateflow
def test_automatic_transformation_different_space(lesion_mni152_2mm):
    """Test that lesion is transformed to different coordinate space."""
    analysis = TestAnalysis1mm()
    result = analysis.run(lesion_mni152_2mm)

    # Should transform to MNI152NLin2009cAsym @ 1mm
    assert result.results["TestAnalysis1mm"]["space"] == "MNI152NLin2009cAsym"
    assert result.results["TestAnalysis1mm"]["resolution"] == 1

    # Verify transformation was recorded in provenance
    transform_records = [r for r in result.provenance if r.get("type") == "spatial_transformation"]
    assert len(transform_records) > 0, "Expected transformation record in provenance"
    assert transform_records[0]["target_space"] == "MNI152NLin2009cAsym"
    assert transform_records[0]["target_resolution"] == 1


def test_no_transformation_when_target_not_specified(lesion_mni152_2mm):
    """Test that no transformation occurs when TARGET_SPACE is not defined."""
    analysis = TestAnalysisNoTarget()
    result = analysis.run(lesion_mni152_2mm)

    # Should preserve original space
    assert result.results["TestAnalysisNoTarget"]["space"] == "MNI152NLin6Asym"
    assert result.results["TestAnalysisNoTarget"]["resolution"] == 2


def test_no_transformation_when_target_not_specified_1mm(lesion_mni152_1mm):
    """Test that adaptive analysis preserves 1mm space."""
    analysis = TestAnalysisNoTarget()
    result = analysis.run(lesion_mni152_1mm)

    # Should preserve original space
    assert result.results["TestAnalysisNoTarget"]["space"] == "MNI152NLin6Asym"
    assert result.results["TestAnalysisNoTarget"]["resolution"] == 1


def test_target_space_class_attributes_exist():
    """Test that TARGET_SPACE attributes are properly defined on analysis classes."""
    from lacuna.analysis import (
        FunctionalNetworkMapping,
        ParcelAggregation,
        RegionalDamage,
        StructuralNetworkMapping,
    )

    # FunctionalNetworkMapping should have explicit target
    assert hasattr(FunctionalNetworkMapping, "TARGET_SPACE")
    assert FunctionalNetworkMapping.TARGET_SPACE == "MNI152NLin6Asym"
    assert FunctionalNetworkMapping.TARGET_RESOLUTION == 2

    # StructuralNetworkMapping should have explicit target
    assert hasattr(StructuralNetworkMapping, "TARGET_SPACE")
    assert StructuralNetworkMapping.TARGET_SPACE == "MNI152NLin2009cAsym"
    assert StructuralNetworkMapping.TARGET_RESOLUTION == 1

    # ParcelAggregation should be adaptive (no transformation)
    assert hasattr(ParcelAggregation, "TARGET_SPACE")
    assert ParcelAggregation.TARGET_SPACE is None
    assert ParcelAggregation.TARGET_RESOLUTION is None

    # RegionalDamage inherits from ParcelAggregation
    assert RegionalDamage.TARGET_SPACE is None
    assert RegionalDamage.TARGET_RESOLUTION is None


def test_error_when_lesion_missing_space_metadata():
    """Test that analysis raises error when lesion data lacks space metadata."""
    # Create lesion without space metadata
    shape = (91, 109, 91)
    data = np.zeros(shape)
    data[45:50, 54:59, 45:50] = 1

    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    img = nib.Nifti1Image(data, affine)

    # This should raise error during MaskData creation
    with pytest.raises(ValueError, match="metadata must contain 'space' key"):
        MaskData(mask_img=img)


def test_provenance_records_target_space():
    """Test that transformation to target space is recorded in provenance."""
    analysis = TestAnalysis2mm()
    lesion = MaskData(
        mask_img=nib.Nifti1Image(
            np.zeros((91, 109, 91)),
            np.array(
                [
                    [-2.0, 0.0, 0.0, 90.0],
                    [0.0, 2.0, 0.0, -126.0],
                    [0.0, 0.0, 2.0, -72.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        metadata={"space": "MNI152NLin6Asym", "resolution": 2},
    )

    result = analysis.run(lesion)

    # Check that analysis ran successfully
    assert "TestAnalysis2mm" in result.results

    # Check provenance was recorded
    assert len(result.provenance) > 0
    last_provenance = result.provenance[-1]
    assert "TestAnalysis2mm" in last_provenance["function"]
