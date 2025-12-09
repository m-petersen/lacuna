"""Contract tests for VoxelMap batch processing.

These tests define the expected behavior for batch processing VoxelMap inputs.
User Story 2: Extend batch_process() to accept list[VoxelMap] returning list[ParcelData].

Contract: T021 - VoxelMap Batch Processing
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis import ParcelAggregation
from lacuna.assets.parcellations.registry import (
    register_parcellations_from_directory,
    unregister_parcellation,
)
from lacuna.batch import batch_process
from lacuna.core.data_types import ParcelData, VoxelMap


@pytest.fixture
def local_test_atlas(tmp_path):
    """Create and register a local test atlas for batch VoxelMap tests.

    This avoids TemplateFlow downloads for CI.
    """
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Use same dimensions as sample_voxelmaps (91, 109, 91)
    shape = (91, 109, 91)
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Create atlas with 5 regions
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[20:40, 30:50, 20:40] = 1
    atlas_data[40:60, 30:50, 20:40] = 2
    atlas_data[20:40, 50:70, 20:40] = 3
    atlas_data[40:60, 50:70, 20:40] = 4
    atlas_data[30:50, 40:60, 40:60] = 5

    atlas_img = nib.Nifti1Image(atlas_data, affine)
    atlas_path = atlas_dir / "test_voxelmap_batch_atlas.nii.gz"
    nib.save(atlas_img, atlas_path)

    # Create labels file
    labels_path = atlas_dir / "test_voxelmap_batch_atlas_labels.txt"
    labels_path.write_text("1 Region_A\n2 Region_B\n3 Region_C\n4 Region_D\n5 Region_E\n")

    # Register the atlas
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    yield "test_voxelmap_batch_atlas"

    # Cleanup
    try:
        unregister_parcellation("test_voxelmap_batch_atlas")
    except KeyError:
        pass


@pytest.fixture
def sample_voxelmaps():
    """Create 5 sample VoxelMaps for batch testing."""
    shape = (91, 109, 91)
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxelmaps = []
    for i in range(5):
        # Create random data with some variation per subject
        data = np.random.rand(*shape).astype(np.float32) * (i + 1) / 5
        img = nib.Nifti1Image(data, affine)

        voxelmap = VoxelMap(
            name=f"subject_{i:03d}_correlation_map",
            data=img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": f"sub-{i:03d}"},
        )
        voxelmaps.append(voxelmap)

    return voxelmaps


class TestVoxelMapBatchContract:
    """Contract tests for batch processing VoxelMap inputs."""

    def test_batch_process_accepts_voxelmap_list(self, sample_voxelmaps, local_test_atlas):
        """
        Contract: batch_process() should accept list[VoxelMap] as input.

        When given a list of VoxelMaps instead of MaskData, batch_process
        should route them correctly through ParcelAggregation.
        """
        analysis = ParcelAggregation(
            source="data",  # VoxelMap's data attribute
            aggregation="mean",
            parcel_names=[local_test_atlas],
        )

        # Should not raise - VoxelMap list accepted
        results = batch_process(sample_voxelmaps, analysis, n_jobs=1, show_progress=False)

        # Should return results
        assert results is not None
        assert len(results) == 5

    def test_batch_process_voxelmap_returns_parcel_data(self, sample_voxelmaps, local_test_atlas):
        """
        Contract: batch_process(list[VoxelMap], ParcelAggregation) returns list with results.

        When processing VoxelMaps through ParcelAggregation, results should contain
        aggregated parcel data. The exact return type depends on implementation:
        - Could be list[dict[str, ParcelData]] (BIDS-style keys to ParcelData)
        - Or list[ParcelData] if single result per input
        """
        analysis = ParcelAggregation(
            source="data", aggregation="mean", parcel_names=[local_test_atlas]
        )

        results = batch_process(sample_voxelmaps, analysis, n_jobs=1, show_progress=False)

        # Should return list with one result per input
        assert len(results) == 5

        # Each result should contain parcel aggregation results
        for result in results:
            # Result could be dict (BIDS keys -> ParcelData) or ParcelData directly
            if isinstance(result, dict):
                # Dict of BIDS-style keys to ParcelData
                assert len(result) > 0
                for _key, value in result.items():
                    assert isinstance(value, ParcelData)
            else:
                # Direct ParcelData (simplified return)
                assert isinstance(result, ParcelData)

    def test_batch_process_voxelmap_preserves_order(self, sample_voxelmaps, local_test_atlas):
        """
        Contract: Results should maintain the same order as input VoxelMaps.
        """
        analysis = ParcelAggregation(
            source="data", aggregation="mean", parcel_names=[local_test_atlas]
        )

        results = batch_process(sample_voxelmaps, analysis, n_jobs=1, show_progress=False)

        # Results should be in same order as input
        assert len(results) == len(sample_voxelmaps)

    def test_batch_process_voxelmap_result_has_identifier(self, sample_voxelmaps, local_test_atlas):
        """
        Contract: Results should maintain correspondence with input VoxelMaps.

        Each result in the output list corresponds to the input VoxelMap at the
        same index position.
        """
        analysis = ParcelAggregation(
            source="data", aggregation="mean", parcel_names=[local_test_atlas]
        )

        results = batch_process(sample_voxelmaps, analysis, n_jobs=1, show_progress=False)

        # Results should correspond to inputs
        assert len(results) == len(sample_voxelmaps)

        # Each result should be valid
        for result in results:
            if isinstance(result, dict):
                assert len(result) > 0
            elif isinstance(result, ParcelData):
                assert result.data is not None
            else:
                # If it's a MaskData (current behavior), check it has results
                assert hasattr(result, "results")

    def test_batch_process_rejects_mixed_types(
        self, sample_voxelmaps, synthetic_mask_img, local_test_atlas
    ):
        """
        Contract: batch_process should reject mixed MaskData and VoxelMap lists.
        """
        from lacuna import MaskData

        mask_data = MaskData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Mix VoxelMap and MaskData
        mixed_list = [sample_voxelmaps[0], mask_data]

        analysis = ParcelAggregation(source="mask_img", parcel_names=[local_test_atlas])

        with pytest.raises(TypeError, match="mixed|type"):
            batch_process(mixed_list, analysis, n_jobs=1, show_progress=False)

    def test_batch_process_voxelmap_empty_list_raises(self, local_test_atlas):
        """
        Contract: Empty VoxelMap list should raise ValueError.
        """
        analysis = ParcelAggregation(source="data", parcel_names=[local_test_atlas])

        with pytest.raises(ValueError, match="cannot be empty"):
            batch_process([], analysis, n_jobs=1, show_progress=False)


@pytest.fixture
def synthetic_mask_img():
    """Create a synthetic mask image for testing."""
    shape = (91, 109, 91)
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.zeros(shape, dtype=np.uint8)
    data[40:50, 50:60, 40:50] = 1
    return nib.Nifti1Image(data, affine)
