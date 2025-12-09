"""Performance benchmarks for FunctionalNetworkMapping optimization.

This module benchmarks the critical bottleneck: _get_lesion_voxel_indices().

Current implementation: O(N × M) nested loop
Optimized implementation: O(N) vectorized lookup

Benchmark scenarios:
- Small lesion: 100 voxels (typical small stroke)
- Medium lesion: 1,000 voxels (typical medium stroke)
- Large lesion: 10,000 voxels (large stroke/multi-focal)

Expected results:
- Current: ~200M operations for 1K lesion × 200K brain voxels
- Optimized: ~1K operations (direct lookup)
- Speedup: ~200x for coordinate matching
"""

import time

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping
from lacuna.core.mask_data import MaskData


@pytest.fixture
def mock_connectome_info():
    """Create mock connectome mask info for benchmarking.

    Uses realistic MNI152NLin6Asym 2mm dimensions (91×109×91).
    Simulates ~200,000 brain voxels (typical for 2mm brain mask).
    """
    shape = (91, 109, 91)

    # Create realistic brain mask (ellipsoid centered in MNI space)
    center = np.array([45, 54, 45])
    radii = np.array([40, 50, 40])

    # Generate coordinates
    x, y, z = np.ogrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]

    # Ellipsoid equation: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² < 1
    mask_3d = (
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    ) < 1

    # Get indices of brain voxels
    mask_indices = np.where(mask_3d)
    n_voxels = len(mask_indices[0])

    # Create affine (2mm isotropic MNI space)
    affine = np.array(
        [[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]], dtype=np.float64
    )

    print(f"Mock connectome: {shape} shape, {n_voxels:,} brain voxels")

    return {
        "mask_shape": shape,
        "mask_indices": mask_indices,
        "mask_affine": affine,
        "n_voxels": n_voxels,
    }


@pytest.fixture
def create_lesion_mask(mock_connectome_info):
    """Factory to create lesion masks of different sizes."""
    shape = mock_connectome_info["mask_shape"]
    affine = mock_connectome_info["mask_affine"]
    brain_mask_indices = mock_connectome_info["mask_indices"]

    def _create_lesion(n_voxels: int, seed: int = 42) -> MaskData:
        """Create a lesion mask with specified number of voxels.

        Parameters
        ----------
        n_voxels : int
            Target number of lesion voxels
        seed : int
            Random seed for reproducibility

        Returns
        -------
        MaskData
            Lesion mask in MNI152NLin6Asym @ 2mm
        """
        rng = np.random.RandomState(seed)

        # Sample random voxels from brain mask
        n_brain_voxels = len(brain_mask_indices[0])
        indices = rng.choice(n_brain_voxels, size=min(n_voxels, n_brain_voxels), replace=False)

        # Create 3D lesion mask
        lesion_3d = np.zeros(shape, dtype=np.uint8)
        lesion_3d[
            brain_mask_indices[0][indices],
            brain_mask_indices[1][indices],
            brain_mask_indices[2][indices],
        ] = 1

        # Create NIfTI image
        lesion_img = nib.Nifti1Image(lesion_3d, affine)

        # Create MaskData
        mask_data = MaskData(
            mask_img=lesion_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"lesion_size": n_voxels},
        )

        actual_voxels = np.sum(lesion_3d)
        assert actual_voxels == n_voxels, f"Expected {n_voxels}, got {actual_voxels}"

        return mask_data

    return _create_lesion


class TestGetLesionVoxelIndicesPerformance:
    """Benchmark _get_lesion_voxel_indices() (current vectorized version)."""

    def _setup_analysis(self, mock_connectome_info):
        """Create FNM instance with mocked connectome info."""
        # Note: We bypass normal __init__ by creating a minimal instance
        # and injecting mock data to avoid connectome registry lookup
        analysis = object.__new__(FunctionalNetworkMapping)

        # Initialize minimal required attributes
        analysis.connectome_name = "mock_gsp1000"
        analysis.method = "boes"
        analysis.pini_percentile = 20
        analysis.n_jobs = 1
        analysis.compute_t_map = True
        analysis.t_threshold = None
        analysis.logger = type(
            "obj",
            (object,),
            {
                "warning": lambda *args, **kwargs: None,
                "success": lambda *args, **kwargs: None,
            },
        )()  # Mock logger

        # Inject mock connectome info (bypass _load_mask_info)
        analysis._mask_info = mock_connectome_info

        return analysis

    def _benchmark_get_indices(self, analysis, mask_data, label: str):
        """Benchmark the index retrieval and print results."""
        # Warm-up run (JIT compilation, cache warming)
        _ = analysis._get_lesion_voxel_indices(mask_data)

        # Timed run
        n_trials = 3
        times = []

        for _ in range(n_trials):
            start = time.perf_counter()
            indices = analysis._get_lesion_voxel_indices(mask_data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n{label}:")
        print(f"  Lesion size: {len(indices):,} voxels")
        print(f"  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Indices found: {len(indices):,}")

        return {
            "label": label,
            "n_voxels": len(indices),
            "time_ms": avg_time * 1000,
            "std_ms": std_time * 1000,
        }

    @pytest.mark.slow
    def test_benchmark_small_lesion(self, mock_connectome_info, create_lesion_mask):
        """Benchmark small lesion (100 voxels) - typical small stroke."""
        analysis = self._setup_analysis(mock_connectome_info)
        mask_data = create_lesion_mask(n_voxels=100)

        result = self._benchmark_get_indices(analysis, mask_data, "Small lesion (100 voxels)")

        # Small lesion should be fast even with current implementation
        assert result["n_voxels"] == 100
        assert result["time_ms"] < 5000  # Should complete in < 5 seconds

    @pytest.mark.slow
    def test_benchmark_medium_lesion(self, mock_connectome_info, create_lesion_mask):
        """Benchmark medium lesion (1,000 voxels) - typical medium stroke."""
        analysis = self._setup_analysis(mock_connectome_info)
        mask_data = create_lesion_mask(n_voxels=1000)

        result = self._benchmark_get_indices(analysis, mask_data, "Medium lesion (1,000 voxels)")

        assert result["n_voxels"] == 1000
        # Current implementation may be slow - no assertion, just measure baseline

    @pytest.mark.slow
    def test_benchmark_large_lesion(self, mock_connectome_info, create_lesion_mask):
        """Benchmark large lesion (10,000 voxels) - large stroke."""
        analysis = self._setup_analysis(mock_connectome_info)
        mask_data = create_lesion_mask(n_voxels=10000)

        result = self._benchmark_get_indices(analysis, mask_data, "Large lesion (10,000 voxels)")

        assert result["n_voxels"] == 10000
        # This will be VERY slow with current O(N×M) implementation - just measure

    @pytest.mark.slow
    def test_benchmark_comparison_all_sizes(self, mock_connectome_info, create_lesion_mask):
        """Compare performance across all lesion sizes."""
        analysis = self._setup_analysis(mock_connectome_info)

        sizes = [100, 500, 1000, 5000, 10000]
        results = []

        print("\n" + "=" * 60)
        print("PERFORMANCE BASELINE - Current Implementation")
        print("=" * 60)

        for size in sizes:
            mask_data = create_lesion_mask(n_voxels=size)
            result = self._benchmark_get_indices(
                analysis, mask_data, f"Lesion size: {size:,} voxels"
            )
            results.append(result)

        # Check that time scales roughly linearly with lesion size
        # (confirming O(N) behavior in lesion voxels, though still O(M) in brain voxels)
        print("\n" + "=" * 60)
        print("SCALING ANALYSIS")
        print("=" * 60)

        for i in range(1, len(results)):
            prev_size = results[i - 1]["n_voxels"]
            curr_size = results[i]["n_voxels"]
            prev_time = results[i - 1]["time_ms"]
            curr_time = results[i]["time_ms"]

            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time

            print(f"{prev_size:,} → {curr_size:,} voxels:")
            print(f"  Size ratio: {size_ratio:.2f}x")
            print(f"  Time ratio: {time_ratio:.2f}x")
            print(f"  Efficiency: {time_ratio/size_ratio:.2f} (expect ~1.0 for linear scaling)")


class TestMemoryFootprint:
    """Measure memory footprint of lookup array approach."""

    def test_lookup_array_size_2mm(self):
        """Calculate lookup array size for MNI152 @ 2mm."""
        shape = (91, 109, 91)
        dtype_bytes = 4  # int32

        array_size_bytes = np.prod(shape) * dtype_bytes
        array_size_mb = array_size_bytes / (1024**2)

        print("\nLookup array for MNI152NLin6Asym @ 2mm:")
        print(f"  Shape: {shape}")
        print(f"  Dtype: int32 ({dtype_bytes} bytes)")
        print(f"  Total size: {array_size_mb:.2f} MB")

        # Verify it's acceptable
        assert array_size_mb < 5, "Lookup array should be < 5 MB for 2mm"

    def test_lookup_array_size_1mm(self):
        """Calculate lookup array size for MNI152 @ 1mm."""
        shape = (182, 218, 182)
        dtype_bytes = 4  # int32

        array_size_bytes = np.prod(shape) * dtype_bytes
        array_size_mb = array_size_bytes / (1024**2)

        print("\nLookup array for MNI152NLin6Asym @ 1mm:")
        print(f"  Shape: {shape}")
        print(f"  Dtype: int32 ({dtype_bytes} bytes)")
        print(f"  Total size: {array_size_mb:.2f} MB")

        # Verify it's acceptable
        assert array_size_mb < 35, "Lookup array should be < 35 MB for 1mm"
