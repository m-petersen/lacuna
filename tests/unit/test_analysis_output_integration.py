"""Tests for analysis modules using polymorphic output architecture.

Following TDD: these tests define the expected behavior of analyses
returning DataContainer objects instead of plain dicts.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis import (
    FunctionalNetworkMapping,
    ParcelAggregation,
)
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)
from lacuna.core import MaskData
from lacuna.core.data_types import (
    DataContainer,
    ParcelData,
    ScalarMetric,
    VoxelMap,
)
from lacuna.core.spaces import CoordinateSpace

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_mask_data():
    """Create sample lesion data for testing."""
    data = np.zeros((91, 109, 91), dtype=np.uint8)
    data[40:50, 50:60, 40:50] = 1  # Binary lesion mask
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    mask_img = nib.Nifti1Image(data, affine)

    space_affine = np.eye(4)
    space_affine[0, 0] = space_affine[1, 1] = space_affine[2, 2] = 2.0
    space = CoordinateSpace(
        identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=space_affine
    )

    return MaskData(
        mask_img=mask_img,
        metadata={
            "subject_id": "test_subject",
            "space": "MNI152NLin6Asym",
            "resolution": 2.0,
            "coordinate_space": space,
        },
    )


@pytest.fixture
def mock_atlas_file(tmp_path):
    """Create mock atlas file."""
    atlas_data = np.random.randint(0, 100, (91, 109, 91), dtype=np.int32)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    atlas_img = nib.Nifti1Image(atlas_data, affine)

    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(atlas_img, atlas_path)
    return atlas_path


# ============================================================================
# ParcelAggregation Tests
# ============================================================================


class TestParcelAggregationOutputs:
    """Test ParcelAggregation returns correct output types."""

    def test_run_analysis_returns_list_of_results(self, sample_mask_data, mock_atlas_file):
        """ParcelAggregation._run_analysis returns list[DataContainer]."""
        # Use a real bundled atlas instead of trying to mock everything
        # Note: This may return empty results if lesion doesn't overlap with atlas
        # The test is primarily checking the return type structure
        analysis = ParcelAggregation(parcel_names=["Schaefer2018_100Parcels7Networks"])

        results = analysis._run_analysis(sample_mask_data)

        assert isinstance(results, dict)
        # May be empty if no overlap, but should still be a dict of DataContainer types
        assert all(isinstance(r, DataContainer) for r in results.values())

    def test_atlas_aggregation_returns_roi_result(self, sample_mask_data, mock_atlas_file):
        """ParcelAggregation returns ParcelData with region data."""
        analysis = ParcelAggregation(parcel_names=["Schaefer2018_100Parcels7Networks"])

        results = analysis._run_analysis(sample_mask_data)

        # Should contain ParcelData (may be empty if no overlap)
        roi_results = [r for r in results if isinstance(r, ParcelData)]
        # All results should be ParcelData type
        assert all(isinstance(r, ParcelData) for r in results)

        if len(roi_results) > 0:
            # Check ParcelData structure
            roi_result = roi_results[0]
            assert isinstance(roi_result.data, dict)
            assert roi_result.name is not None
            assert roi_result.parcel_names is not None
            assert roi_result.aggregation_method is not None

    def test_atlas_aggregation_roi_result_has_metadata(self, sample_mask_data, mock_atlas_file):
        """ParcelData from ParcelAggregation includes analysis metadata."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"], aggregation="percent"
        )

        results = analysis._run_analysis(sample_mask_data)

        if len(results) > 0:
            roi_result = [r for r in results if isinstance(r, ParcelData)][0]
            assert roi_result.aggregation_method == "percent"
            assert roi_result.metadata is not None


# ============================================================================
# FunctionalNetworkMapping Tests
# ============================================================================


class TestFunctionalNetworkMappingOutputs:
    """Test FunctionalNetworkMapping returns correct output types."""

    @pytest.fixture
    def mock_connectome(self, tmp_path):
        """Create mock HDF5 connectome file."""
        import h5py

        # Create connectome structure that FunctionalNetworkMapping expects
        n_subjects = 5
        n_timepoints = 100
        n_voxels = 1000

        # Mock mask indices - ensure some overlap with lesion area [40:50, 50:60, 40:50]
        # Create voxels in and around the lesion area
        x_coords = np.random.randint(35, 55, n_voxels)  # Around lesion x: 40-50
        y_coords = np.random.randint(45, 65, n_voxels)  # Around lesion y: 50-60
        z_coords = np.random.randint(35, 55, n_voxels)  # Around lesion z: 40-50
        mask_indices = np.array([x_coords, y_coords, z_coords])

        # Mock timeseries data
        timeseries = np.random.rand(n_subjects, n_timepoints, n_voxels).astype(np.float32)

        # Mock affine
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        # Create HDF5 file
        connectome_path = tmp_path / "connectome.h5"
        with h5py.File(connectome_path, "w") as hf:
            hf.create_dataset("timeseries", data=timeseries)
            hf.create_dataset("mask_indices", data=mask_indices)
            hf.create_dataset("mask_affine", data=affine)  # Use mask_affine not affine
            # Add mask_shape attribute
            hf.attrs["mask_shape"] = (91, 109, 91)

        return connectome_path

    def test_run_analysis_returns_dict_of_results(self, sample_mask_data, mock_connectome):
        """FunctionalNetworkMapping._run_analysis returns dict[str, DataContainer]."""
        from lacuna.assets.connectomes import (
            register_functional_connectome,
            unregister_functional_connectome,
        )

        # Register the mock connectome
        register_functional_connectome(
            name="test_func_connectome",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=mock_connectome,
            n_subjects=5,
            description="Test connectome"
        )

        try:
            analysis = FunctionalNetworkMapping(connectome_name="test_func_connectome")
            results = analysis._run_analysis(sample_mask_data)

            assert isinstance(results, dict)
            assert len(results) > 0
            assert all(isinstance(r, DataContainer) for r in results.values())
        finally:
            unregister_functional_connectome("test_func_connectome")

    def test_functional_mapping_returns_voxel_map_results(self, sample_mask_data, mock_connectome):
        """FunctionalNetworkMapping returns dictionary of VoxelMap objects for brain maps."""
        register_functional_connectome(
            name="test_func_connectome",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=mock_connectome,
            n_subjects=5,
            description="Test connectome"
        )

        try:
            analysis = FunctionalNetworkMapping(connectome_name="test_func_connectome")
            results = analysis._run_analysis(sample_mask_data)

            # Results should now be a dict, not a list
            assert isinstance(results, dict)
            # Should contain VoxelMapResults for CorrelationMap, ZMap
            voxel_results = [r for r in results.values() if isinstance(r, VoxelMap)]
            assert len(voxel_results) >= 2  # At least CorrelationMap and ZMap

            # Check for expected result names
            result_names = [r.name for r in voxel_results]
            assert "CorrelationMap" in result_names
            assert "ZMap" in result_names
        finally:
            unregister_functional_connectome("test_func_connectome")

    def test_functional_mapping_voxel_results_have_spaces(self, sample_mask_data, mock_connectome):
        """VoxelMapResults from FunctionalNetworkMapping have space and resolution."""
        register_functional_connectome(
            name="test_voxel_connectome",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=mock_connectome,
            n_subjects=5,
            description="Test connectome"
        )

        try:
            analysis = FunctionalNetworkMapping(connectome_name="test_voxel_connectome")

            results = analysis._run_analysis(sample_mask_data)
            voxel_results = [r for r in results.values() if isinstance(r, VoxelMap)]

            for voxel_result in voxel_results:
                assert voxel_result.space is not None
                assert voxel_result.resolution is not None
                assert isinstance(voxel_result.space, str)
                assert isinstance(voxel_result.resolution, float)
        finally:
            unregister_functional_connectome("test_voxel_connectome")

    def test_functional_mapping_returns_misc_result_for_scalars(
        self, sample_mask_data, mock_connectome
    ):
        """FunctionalNetworkMapping returns dictionary with ScalarMetric for summary statistics."""
        register_functional_connectome(
            name="test_scalar_connectome",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=mock_connectome,
            n_subjects=5,
            description="Test connectome"
        )

        try:
            analysis = FunctionalNetworkMapping(connectome_name="test_scalar_connectome")

            results = analysis._run_analysis(sample_mask_data)

            # Results should now be a dict, not a list
            assert isinstance(results, dict)
            # Should contain ScalarMetric for summary statistics
            misc_results = [r for r in results.values() if isinstance(r, ScalarMetric)]
            assert len(misc_results) > 0

            # Check for summary_statistics result
            summary_results = [r for r in misc_results if "summary" in r.name.lower()]
            assert len(summary_results) > 0
        finally:
            unregister_functional_connectome("test_scalar_connectome")
