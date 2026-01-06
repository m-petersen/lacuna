"""Unit tests for FunctionalNetworkMapping p-value and FDR map functionality.

These tests verify the new pmap and pfdrmap output capabilities:
1. P-value map computation from t-statistics
2. FDR-corrected p-value map using Benjamini-Hochberg
3. Configuration options for enabling/disabling these outputs
"""

import h5py
import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)


@pytest.fixture
def valid_connectome(tmp_path):
    """Create a valid connectome file for testing."""
    connectome_path = tmp_path / "test_connectome.h5"

    # Create realistic mask indices that map properly to a brain volume
    n_voxels = 100
    mask_shape = (10, 10, 10)

    # Generate valid indices within the mask shape
    np.random.seed(42)
    x_indices = np.random.randint(0, mask_shape[0], n_voxels)
    y_indices = np.random.randint(0, mask_shape[1], n_voxels)
    z_indices = np.random.randint(0, mask_shape[2], n_voxels)
    mask_indices = np.array([x_indices, y_indices, z_indices])

    # Create affine for 2mm resolution
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0

    # Create timeseries data
    n_subjects = 20
    n_timepoints = 50
    timeseries = np.random.randn(n_subjects, n_timepoints, n_voxels).astype(np.float32)

    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=timeseries)
        f.create_dataset("mask_indices", data=mask_indices)
        f.create_dataset("mask_affine", data=affine)
        f.attrs["mask_shape"] = mask_shape

    return connectome_path, mask_shape, affine


@pytest.fixture
def valid_mask_img(valid_connectome):
    """Create a valid mask image that overlaps with connectome."""
    _, mask_shape, affine = valid_connectome

    # Create a mask with some voxels in the connectome mask region
    mask_data = np.zeros(mask_shape, dtype=np.uint8)
    mask_data[3:7, 3:7, 3:7] = 1  # Create a cube of lesion voxels

    mask_img = nib.Nifti1Image(mask_data, affine)
    return mask_img


class TestPValueMapInitialization:
    """Test initialization parameters for p-value computation."""

    def test_compute_p_map_default_true(self, valid_connectome):
        """Test that compute_p_map defaults to True."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_pmap_default",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pmap_default",
                verbose=False,
            )
            assert fnm.compute_p_map is True
        finally:
            unregister_functional_connectome("test_pmap_default")

    def test_compute_p_map_can_be_disabled(self, valid_connectome):
        """Test that compute_p_map can be set to False."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_pmap_disable",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pmap_disable",
                compute_p_map=False,
                verbose=False,
            )
            assert fnm.compute_p_map is False
        finally:
            unregister_functional_connectome("test_pmap_disable")

    def test_fdr_alpha_default(self, valid_connectome):
        """Test that fdr_alpha defaults to 0.05."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_fdr_default",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_fdr_default",
                verbose=False,
            )
            assert fnm.fdr_alpha == 0.05
        finally:
            unregister_functional_connectome("test_fdr_default")

    def test_fdr_alpha_can_be_none(self, valid_connectome):
        """Test that fdr_alpha can be set to None to disable FDR correction."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_fdr_none",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_fdr_none",
                fdr_alpha=None,
                verbose=False,
            )
            assert fnm.fdr_alpha is None
        finally:
            unregister_functional_connectome("test_fdr_none")

    def test_fdr_alpha_custom_value(self, valid_connectome):
        """Test that fdr_alpha accepts custom values."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_fdr_custom",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_fdr_custom",
                fdr_alpha=0.01,
                verbose=False,
            )
            assert fnm.fdr_alpha == 0.01
        finally:
            unregister_functional_connectome("test_fdr_custom")


class TestPValueMapOutput:
    """Test p-value map output from run()."""

    def test_pmap_in_results_when_enabled(self, valid_connectome, valid_mask_img):
        """Test that pmap is in results when compute_p_map=True."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pmap_output",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pmap_output",
                compute_p_map=True,
                fdr_alpha=None,  # Disable FDR for this test
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pmap is in results
            assert "pmap" in result.results["FunctionalNetworkMapping"]
            pmap = result.results["FunctionalNetworkMapping"]["pmap"]

            # Verify it's a VoxelMap
            from lacuna.core.data_types import VoxelMap

            assert isinstance(pmap, VoxelMap)

            # Verify p-values are in valid range [0, 1]
            pmap_data = pmap.get_data().get_fdata()
            assert np.all(pmap_data >= 0)
            assert np.all(pmap_data <= 1)

        finally:
            unregister_functional_connectome("test_pmap_output")

    def test_pmap_not_in_results_when_disabled(self, valid_connectome, valid_mask_img):
        """Test that pmap is not in results when compute_p_map=False."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pmap_disabled_output",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pmap_disabled_output",
                compute_p_map=False,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pmap is NOT in results
            assert "pmap" not in result.results["FunctionalNetworkMapping"]
            assert "pfdrmap" not in result.results["FunctionalNetworkMapping"]

        finally:
            unregister_functional_connectome("test_pmap_disabled_output")

    def test_pfdrmap_in_results_when_fdr_enabled(self, valid_connectome, valid_mask_img):
        """Test that pfdrmap is in results when fdr_alpha is set."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pfdr_output",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pfdr_output",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pfdrmap is in results
            assert "pfdrmap" in result.results["FunctionalNetworkMapping"]
            pfdrmap = result.results["FunctionalNetworkMapping"]["pfdrmap"]

            # Verify it's a VoxelMap
            from lacuna.core.data_types import VoxelMap

            assert isinstance(pfdrmap, VoxelMap)

            # Verify FDR p-values are in valid range [0, 1]
            pfdr_data = pfdrmap.get_data().get_fdata()
            assert np.all(pfdr_data >= 0)
            assert np.all(pfdr_data <= 1)

            # Verify metadata contains FDR info
            assert pfdrmap.metadata["fdr_alpha"] == 0.05
            assert pfdrmap.metadata["correction_method"] == "benjamini_hochberg"

        finally:
            unregister_functional_connectome("test_pfdr_output")

    def test_pfdrmap_not_in_results_when_fdr_disabled(self, valid_connectome, valid_mask_img):
        """Test that pfdrmap is not in results when fdr_alpha=None."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pfdr_disabled_output",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pfdr_disabled_output",
                compute_p_map=True,
                fdr_alpha=None,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pmap IS in results but pfdrmap is NOT
            assert "pmap" in result.results["FunctionalNetworkMapping"]
            assert "pfdrmap" not in result.results["FunctionalNetworkMapping"]

        finally:
            unregister_functional_connectome("test_pfdr_disabled_output")


class TestPValueMapCorrectness:
    """Test correctness of p-value computation."""

    def test_pvalue_computation_matches_scipy(self, valid_connectome, valid_mask_img):
        """Verify p-value computation matches scipy.stats.t.sf."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pvalue_correctness",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pvalue_correctness",
                compute_p_map=True,
                fdr_alpha=None,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Get t-map and p-map
            tmap_data = result.results["FunctionalNetworkMapping"]["tmap"].get_data().get_fdata()
            pmap_data = result.results["FunctionalNetworkMapping"]["pmap"].get_data().get_fdata()

            # Find non-zero voxels
            nonzero_mask = tmap_data != 0
            t_values = tmap_data[nonzero_mask]
            p_values = pmap_data[nonzero_mask]

            # Compute expected p-values using scipy
            df = 20 - 1  # n_subjects - 1
            expected_p = 2 * stats.t.sf(np.abs(t_values), df)

            # Compare (allow small numerical differences)
            np.testing.assert_allclose(p_values, expected_p, rtol=1e-5)

        finally:
            unregister_functional_connectome("test_pvalue_correctness")

    def test_fdr_correction_is_monotonic(self, valid_connectome, valid_mask_img):
        """Test that FDR-corrected p-values are monotonic with original p-values."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_fdr_monotonic",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_fdr_monotonic",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Get p-map and fdr p-map
            pmap_data = result.results["FunctionalNetworkMapping"]["pmap"].get_data().get_fdata()
            pfdrmap_data = (
                result.results["FunctionalNetworkMapping"]["pfdrmap"].get_data().get_fdata()
            )

            # Find non-zero voxels
            nonzero_mask = pmap_data != 0
            p_values = pmap_data[nonzero_mask]
            pfdr_values = pfdrmap_data[nonzero_mask]

            # FDR-corrected p-values should be >= original p-values
            assert np.all(pfdr_values >= p_values - 1e-10)

            # FDR-corrected p-values should be <= 1
            assert np.all(pfdr_values <= 1.0 + 1e-10)

        finally:
            unregister_functional_connectome("test_fdr_monotonic")


class TestReturnInInputSpace:
    """Test return_in_input_space functionality."""

    def test_return_in_input_space_default_true(self, valid_connectome):
        """Test that return_in_input_space defaults to True."""
        connectome_path, _, _ = valid_connectome

        register_functional_connectome(
            name="test_input_space_default",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_input_space_default",
                verbose=False,
            )
            assert fnm.return_in_input_space is True
        finally:
            unregister_functional_connectome("test_input_space_default")

    def test_pmaps_transformed_when_return_in_input_space(self, valid_connectome, valid_mask_img):
        """Test that p-maps are transformed to input space when requested."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pmap_transform",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pmap_transform",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=True,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Get pmap and check it was transformed
            pmap = result.results["FunctionalNetworkMapping"]["pmap"]
            pfdrmap = result.results["FunctionalNetworkMapping"]["pfdrmap"]

            # Output space should match input space
            assert pmap.space == "MNI152NLin6Asym"
            assert pfdrmap.space == "MNI152NLin6Asym"

            # Shape should match input mask shape
            input_shape = valid_mask_img.shape
            assert pmap.get_data().shape == input_shape
            assert pfdrmap.get_data().shape == input_shape

        finally:
            unregister_functional_connectome("test_pmap_transform")


class TestSummaryStatistics:
    """Test that summary statistics include p-value information."""

    def test_summary_includes_p_stats(self, valid_connectome, valid_mask_img):
        """Test that summary statistics include p-value min/max."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_summary_pstats",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_summary_pstats",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Get summary statistics
            summary = result.results["FunctionalNetworkMapping"]["summarystatistics"]
            summary_data = summary.get_data()

            # Check p-value stats are present
            assert "p_min" in summary_data
            assert "p_max" in summary_data
            assert 0 <= summary_data["p_min"] <= 1
            assert 0 <= summary_data["p_max"] <= 1

            # Check FDR stats are present
            assert "n_significant_fdr" in summary_data
            assert "pct_significant_fdr" in summary_data
            assert "fdr_alpha" in summary_data
            assert summary_data["fdr_alpha"] == 0.05

        finally:
            unregister_functional_connectome("test_summary_pstats")


class TestPFdrThresholdMap:
    """Test the binary FDR significance threshold map output."""

    def test_pfdrthresholdmap_in_results_when_fdr_enabled(self, valid_connectome, valid_mask_img):
        """Test that pfdrthresholdmap is in results when fdr_alpha is set."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pfdrthresh_output",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pfdrthresh_output",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pfdrthresholdmap is in results
            assert "pfdrthresholdmap" in result.results["FunctionalNetworkMapping"]
            pfdrthresh = result.results["FunctionalNetworkMapping"]["pfdrthresholdmap"]

            # Verify it's a VoxelMap
            from lacuna.core.data_types import VoxelMap

            assert isinstance(pfdrthresh, VoxelMap)

            # Verify it's a binary mask (only 0 and 1)
            thresh_data = pfdrthresh.get_data().get_fdata()
            unique_vals = np.unique(thresh_data)
            assert np.all(np.isin(unique_vals, [0, 1]))

            # Verify metadata contains FDR info
            assert pfdrthresh.metadata["fdr_alpha"] == 0.05
            assert pfdrthresh.metadata["statistic"] == "fdr_significant_binary"

        finally:
            unregister_functional_connectome("test_pfdrthresh_output")

    def test_pfdrthresholdmap_not_in_results_when_fdr_disabled(
        self, valid_connectome, valid_mask_img
    ):
        """Test that pfdrthresholdmap is not in results when fdr_alpha=None."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pfdrthresh_disabled",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pfdrthresh_disabled",
                compute_p_map=True,
                fdr_alpha=None,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Check pfdrthresholdmap is NOT in results
            assert "pfdrthresholdmap" not in result.results["FunctionalNetworkMapping"]

        finally:
            unregister_functional_connectome("test_pfdrthresh_disabled")

    def test_pfdrthresholdmap_matches_pfdrmap_threshold(self, valid_connectome, valid_mask_img):
        """Test that pfdrthresholdmap correctly thresholds pfdrmap at fdr_alpha."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_pfdrthresh_matches",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fdr_alpha = 0.05
            fnm = FunctionalNetworkMapping(
                connectome_name="test_pfdrthresh_matches",
                compute_p_map=True,
                fdr_alpha=fdr_alpha,
                return_in_input_space=False,
                verbose=False,
            )

            subject = SubjectData(
                mask_img=valid_mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
            )

            result = fnm.run(subject)

            # Get both maps
            pfdrmap_data = (
                result.results["FunctionalNetworkMapping"]["pfdrmap"].get_data().get_fdata()
            )
            pfdrthresh_data = (
                result.results["FunctionalNetworkMapping"]["pfdrthresholdmap"]
                .get_data()
                .get_fdata()
            )

            # Only compare brain voxels (where rmap is non-zero as proxy for brain mask)
            # Background voxels have pfdrmap=0 (less than alpha) but should NOT be marked significant
            rmap_data = result.results["FunctionalNetworkMapping"]["rmap"].get_data().get_fdata()
            brain_mask = rmap_data != 0

            # Within brain voxels, verify threshold map matches where pfdrmap < fdr_alpha
            pfdr_brain = pfdrmap_data[brain_mask]
            thresh_brain = pfdrthresh_data[brain_mask]
            expected_thresh_brain = (pfdr_brain < fdr_alpha).astype(np.float32)
            np.testing.assert_array_equal(thresh_brain, expected_thresh_brain)

            # Verify background voxels are 0 in threshold map (not marked significant)
            thresh_background = pfdrthresh_data[~brain_mask]
            assert np.all(thresh_background == 0)

        finally:
            unregister_functional_connectome("test_pfdrthresh_matches")


class TestBatchModePValueMaps:
    """Test p-value and FDR maps in vectorized batch mode (run_batch)."""

    def test_batch_mode_includes_pmap(self, valid_connectome, valid_mask_img):
        """Test that batch mode (run_batch) includes pmap in results."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_batch_pmap",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_batch_pmap",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            # Create multiple subjects
            subjects = [
                SubjectData(
                    mask_img=valid_mask_img,
                    space="MNI152NLin6Asym",
                    resolution=2.0,
                    metadata={"subject_id": f"sub-{i:03d}"},
                )
                for i in range(3)
            ]

            # Run batch mode
            results = fnm.run_batch(subjects)

            # Check all results have pmap, pfdrmap, and pfdrthresholdmap
            for i, result in enumerate(results):
                assert (
                    "pmap" in result.results["FunctionalNetworkMapping"]
                ), f"Subject {i} missing pmap"
                assert (
                    "pfdrmap" in result.results["FunctionalNetworkMapping"]
                ), f"Subject {i} missing pfdrmap"
                assert (
                    "pfdrthresholdmap" in result.results["FunctionalNetworkMapping"]
                ), f"Subject {i} missing pfdrthresholdmap"

                # Verify p-values are valid
                pmap = result.results["FunctionalNetworkMapping"]["pmap"]
                pmap_data = pmap.get_data().get_fdata()
                assert np.all(pmap_data >= 0)
                assert np.all(pmap_data <= 1)

        finally:
            unregister_functional_connectome("test_batch_pmap")

    def test_batch_mode_pmap_correctness(self, valid_connectome, valid_mask_img):
        """Test that batch mode p-values match scipy computation."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_batch_pmap_correct",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_batch_pmap_correct",
                compute_p_map=True,
                fdr_alpha=None,  # Disable FDR to test p-value only
                return_in_input_space=False,
                verbose=False,
            )

            subjects = [
                SubjectData(
                    mask_img=valid_mask_img,
                    space="MNI152NLin6Asym",
                    resolution=2.0,
                    metadata={"subject_id": "sub-001"},
                )
            ]

            results = fnm.run_batch(subjects)
            result = results[0]

            # Get t-map and p-map
            tmap_data = result.results["FunctionalNetworkMapping"]["tmap"].get_data().get_fdata()
            pmap_data = result.results["FunctionalNetworkMapping"]["pmap"].get_data().get_fdata()

            # Find non-zero voxels
            nonzero_mask = tmap_data != 0
            t_values = tmap_data[nonzero_mask]
            p_values = pmap_data[nonzero_mask]

            # Compute expected p-values using scipy
            df = 20 - 1  # n_subjects - 1
            expected_p = 2 * stats.t.sf(np.abs(t_values), df)

            # Compare (allow small numerical differences)
            np.testing.assert_allclose(p_values, expected_p, rtol=1e-5)

        finally:
            unregister_functional_connectome("test_batch_pmap_correct")

    def test_batch_mode_fdr_stats_in_summary(self, valid_connectome, valid_mask_img):
        """Test that batch mode summary includes FDR statistics."""
        connectome_path, mask_shape, affine = valid_connectome

        register_functional_connectome(
            name="test_batch_fdr_summary",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=connectome_path,
            n_subjects=20,
        )
        try:
            fnm = FunctionalNetworkMapping(
                connectome_name="test_batch_fdr_summary",
                compute_p_map=True,
                fdr_alpha=0.05,
                return_in_input_space=False,
                verbose=False,
            )

            subjects = [
                SubjectData(
                    mask_img=valid_mask_img,
                    space="MNI152NLin6Asym",
                    resolution=2.0,
                    metadata={"subject_id": "sub-001"},
                )
            ]

            results = fnm.run_batch(subjects)
            result = results[0]

            # Get summary statistics
            summary = result.results["FunctionalNetworkMapping"]["summarystatistics"]
            summary_data = summary.get_data()

            # Check p-value stats are present
            assert "p_min" in summary_data
            assert "p_max" in summary_data

            # Check FDR stats are present
            assert "n_significant_fdr" in summary_data
            assert "pct_significant_fdr" in summary_data
            assert "fdr_alpha" in summary_data
            assert summary_data["fdr_alpha"] == 0.05

        finally:
            unregister_functional_connectome("test_batch_fdr_summary")
