"""Integration tests for transformation optimization strategy.

NOTE: The original tests were removed because they tested methods that
were never implemented:
- determine_direction() with source_img, target_img, check_cache parameters
- compare_resolutions()
- estimate_data_size()

The actual TransformationStrategy.determine_direction() only takes
source and target CoordinateSpace objects and determines direction
based on space identifiers, not data sizes.

TODO: Implement size-based optimization if needed, then add tests.

For now, this file contains tests for existing functionality.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
from lacuna.spatial.transform import TransformationStrategy


class TestTransformationDirectionChoices:
    """Test automatic transformation direction determination."""

    def test_forward_direction_nlin6_to_nlin2009c(self):
        """Test forward transformation direction from NLin6 to NLin2009c."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        strategy = TransformationStrategy()
        direction = strategy.determine_direction(source, target)

        assert direction == "forward"

    def test_reverse_direction_nlin2009c_to_nlin6(self):
        """Test reverse transformation direction from NLin2009c to NLin6."""
        source = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )

        strategy = TransformationStrategy()
        direction = strategy.determine_direction(source, target)

        assert direction == "reverse"

    def test_no_transform_same_space(self):
        """Test no transformation needed for same space."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )

        strategy = TransformationStrategy()
        direction = strategy.determine_direction(source, target)

        assert direction == "none"

    def test_resample_same_space_different_resolution(self):
        """Test resampling for same space with different resolution."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=1,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 1)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )

        strategy = TransformationStrategy()
        direction = strategy.determine_direction(source, target)

        assert direction == "resample"


class TestInterpolationSelection:
    """Test interpolation method selection."""

    def test_select_cubic_default(self):
        """Test that cubic is selected by default for continuous data."""
        from lacuna.spatial.transform import InterpolationMethod

        data = np.random.rand(91, 109, 91).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))

        strategy = TransformationStrategy()
        interp = strategy.select_interpolation(img)

        # Cubic B-spline is now the default for continuous data
        assert interp == InterpolationMethod.CUBIC

    def test_select_nearest_for_binary(self):
        """Test that nearest is selected for binary mask data."""
        from lacuna.spatial.transform import InterpolationMethod

        data = np.zeros((91, 109, 91), dtype=np.uint8)
        data[40:50, 40:50, 40:50] = 1
        img = nib.Nifti1Image(data, np.eye(4))

        strategy = TransformationStrategy()
        interp = strategy.select_interpolation(img)

        assert interp == InterpolationMethod.NEAREST

    def test_override_interpolation_method(self):
        """Test that interpolation method can be overridden."""
        from lacuna.spatial.transform import InterpolationMethod

        data = np.random.rand(91, 109, 91).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))

        strategy = TransformationStrategy()
        interp = strategy.select_interpolation(img, method=InterpolationMethod.NEAREST)

        assert interp == InterpolationMethod.NEAREST


@pytest.mark.slow
class TestTransformationOptimizationIntegration:
    """Integration tests for full transformation optimization workflow."""

    def test_rationale_appears_in_provenance(self):
        """Test that transformation rationale is recorded in provenance."""
        from lacuna.core.provenance import TransformationRecord

        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1,
            method="nitransforms",
            interpolation="linear",
            rationale="Forward transformation to target space",
        )

        record_dict = record.to_dict()
        assert "rationale" in record_dict
        assert record_dict["rationale"] == "Forward transformation to target space"
