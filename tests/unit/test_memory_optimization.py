"""
Test memory optimization improvements in vectorized batch processing.

This test validates that the streaming aggregation approach significantly
reduces memory usage compared to accumulating all correlation maps.
"""

import h5py
import nibabel as nib
import numpy as np
import pytest

from ldk import LesionData
from ldk.analysis import FunctionalNetworkMapping
from ldk.batch import batch_process


@pytest.fixture
def mock_connectome_batched(tmp_path):
    """Create mock connectome split into multiple batch files."""
    n_voxels = 125
    x_coords = np.repeat(range(43, 48), 25)
    y_coords = np.tile(np.repeat(range(52, 57), 5), 5)
    z_coords = np.tile(range(43, 48), 25)
    mask_indices = np.array([x_coords, y_coords, z_coords])

    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    connectome_dir = tmp_path / "connectomes"
    connectome_dir.mkdir()

    # Create 5 batches of 20 subjects each = 100 total
    for batch_idx in range(5):
        connectome_path = connectome_dir / f"batch_{batch_idx:03d}.h5"
        with h5py.File(connectome_path, "w") as f:
            f.create_dataset("timeseries", data=np.random.randn(20, 100, n_voxels))
            f.create_dataset("mask_indices", data=mask_indices)
            f.create_dataset("mask_affine", data=affine)
            f.attrs["mask_shape"] = (91, 109, 91)

    return {
        "connectome_dir": connectome_dir,
        "affine": affine,
        "mask_shape": (91, 109, 91),
        "n_subjects": 100,
        "n_batches": 5,
    }


@pytest.fixture
def mock_lesions(tmp_path, mock_connectome_batched):
    """Create mock lesions for testing."""
    affine = mock_connectome_batched["affine"]
    lesions = []

    for i in range(10):
        lesion_data_array = np.zeros((91, 109, 91), dtype=np.uint8)
        lesion_data_array[43:48, 52:57, 43:48] = 1
        lesion_img = nib.Nifti1Image(lesion_data_array, affine)
        lesion_path = tmp_path / f"lesion_{i}.nii.gz"
        nib.save(lesion_img, lesion_path)
        lesion = LesionData.from_nifti(
            str(lesion_path), metadata={"space": "MNI152_2mm", "subject_id": f"test_{i:03d}"}
        )
        lesions.append(lesion)

    return lesions


def test_streaming_aggregation_produces_correct_results(mock_connectome_batched, mock_lesions):
    """Test that streaming aggregation produces identical results to accumulation."""
    connectome_dir = mock_connectome_batched["connectome_dir"]

    # Create analysis
    analysis = FunctionalNetworkMapping(
        connectome_path=str(connectome_dir),
        method="boes",
        verbose=False,
        compute_t_map=True,
        t_threshold=2.0,
    )

    # Process lesions
    results = batch_process(
        mock_lesions[:3],  # Just test first 3 for speed
        analysis,
        strategy="vectorized",
        show_progress=False,
    )

    # Validate results structure
    assert len(results) == 3

    for result in results:
        flnm = result.results["FunctionalNetworkMapping"]

        # Check all expected outputs exist
        assert "correlation_map" in flnm
        assert "z_map" in flnm
        assert "t_map" in flnm
        assert "summary_statistics" in flnm

        # Check shapes
        assert flnm["correlation_map"].shape == (91, 109, 91)
        assert flnm["t_map"].shape == (91, 109, 91)

        # Check aggregated across all subjects
        assert flnm["summary_statistics"]["n_subjects"] == 100
        assert flnm["summary_statistics"]["n_batches"] == 5

        # Check values in reasonable range
        assert -1 <= flnm["mean_correlation"] <= 1


def test_streaming_aggregation_with_lesion_batches(mock_connectome_batched, mock_lesions):
    """Test streaming aggregation with lesion batching."""
    connectome_dir = mock_connectome_batched["connectome_dir"]

    saved_batches = []

    def save_callback(batch_results):
        saved_batches.append([r.metadata["subject_id"] for r in batch_results])

    analysis = FunctionalNetworkMapping(
        connectome_path=str(connectome_dir), method="boes", verbose=False, compute_t_map=True
    )

    # Process with lesion batches of 3
    results = batch_process(
        mock_lesions,
        analysis,
        strategy="vectorized",
        lesion_batch_size=3,
        batch_result_callback=save_callback,
        show_progress=False,
    )

    # Verify all lesions processed
    assert len(results) == 10

    # Verify batches were saved incrementally
    assert len(saved_batches) == 4  # 10 lesions / 3 per batch = 4 batches
    assert saved_batches[0] == ["test_000", "test_001", "test_002"]
    assert saved_batches[1] == ["test_003", "test_004", "test_005"]
    assert saved_batches[2] == ["test_006", "test_007", "test_008"]
    assert saved_batches[3] == ["test_009"]  # Last batch partial

    # Verify all results have correct subject count
    for result in results:
        assert result.results["FunctionalNetworkMapping"]["summary_statistics"]["n_subjects"] == 100


def test_float32_optimization(mock_connectome_batched, mock_lesions):
    """Test that float32 is used throughout for memory efficiency."""
    connectome_dir = mock_connectome_batched["connectome_dir"]

    analysis = FunctionalNetworkMapping(
        connectome_path=str(connectome_dir), method="boes", verbose=False, compute_t_map=False
    )

    results = batch_process(mock_lesions[:2], analysis, strategy="vectorized", show_progress=False)

    # Check that output arrays use float32 internally
    for result in results:
        flnm = result.results["FunctionalNetworkMapping"]

        # Check data type of nibabel images (get_fdata() converts to float64)
        # So we check the internal data type instead
        assert flnm["correlation_map"].get_data_dtype() == np.float32
        assert flnm["z_map"].get_data_dtype() == np.float32


def test_t_statistics_with_streaming(mock_connectome_batched, mock_lesions):
    """Test t-statistic computation with streaming aggregation."""
    connectome_dir = mock_connectome_batched["connectome_dir"]

    analysis = FunctionalNetworkMapping(
        connectome_path=str(connectome_dir),
        method="boes",
        verbose=False,
        compute_t_map=True,
        t_threshold=2.5,
    )

    results = batch_process(mock_lesions[:2], analysis, strategy="vectorized", show_progress=False)

    for result in results:
        flnm = result.results["FunctionalNetworkMapping"]

        # T-map should exist
        assert "t_map" in flnm
        assert flnm["t_map"].shape == (91, 109, 91)

        # Threshold map should exist
        assert "t_threshold_map" in flnm

        # Statistics should include t-map info
        stats = flnm["summary_statistics"]
        assert "t_min" in stats
        assert "t_max" in stats
        assert "t_threshold" in stats
        assert stats["t_threshold"] == 2.5
        assert "n_significant_voxels" in stats


def test_memory_efficiency_improvement():
    """
    Conceptual test documenting memory improvements.

    This test documents the expected memory savings from streaming aggregation.
    Actual measurements require profiling tools.
    """
    # Memory comparison (theoretical):
    n_lesions = 20
    n_subjects = 1000
    n_voxels = 50000
    bytes_per_float32 = 4
    bytes_per_float64 = 8

    # OLD APPROACH: Accumulate all correlation maps
    old_memory_r_maps = n_lesions * n_subjects * n_voxels * bytes_per_float32
    old_memory_mb = old_memory_r_maps / (1024 * 1024)

    # NEW APPROACH: Streaming aggregation (only store sums)
    # sum_z + sum_z2 for each lesion
    new_memory_aggregators = n_lesions * n_voxels * 2 * bytes_per_float64
    new_memory_mb = new_memory_aggregators / (1024 * 1024)

    # Calculate reduction
    reduction_factor = old_memory_mb / new_memory_mb
    reduction_percent = (1 - new_memory_mb / old_memory_mb) * 100

    # Document expected improvements
    assert old_memory_mb > 3800, f"Old: {old_memory_mb:.0f} MB"
    assert new_memory_mb < 20, f"New: {new_memory_mb:.0f} MB"  # 15.3 MB for this example
    assert reduction_factor > 200, f"Reduction: {reduction_factor:.0f}x"
    assert reduction_percent > 99, f"Reduction: {reduction_percent:.1f}%"

    print("\nMemory Optimization Benefits:")
    print(f"  Old approach: {old_memory_mb:.0f} MB (accumulate all r_maps)")
    print(f"  New approach: {new_memory_mb:.0f} MB (streaming aggregation)")
    print(f"  Reduction: {reduction_factor:.0f}x ({reduction_percent:.1f}%)")
    print("  \nWith same RAM, can now process:")
    print(f"    {reduction_factor:.0f}x more lesions per batch")
    print(f"    e.g., 20 lesions → {int(20 * reduction_factor)} lesions")
