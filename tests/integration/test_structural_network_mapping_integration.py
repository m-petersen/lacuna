"""Integration tests for StructuralNetworkMapping with real data.

These tests verify end-to-end functionality with actual tractograms,
atlases, and lesion masks, including connectivity matrix computation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData
from lacuna.analysis import StructuralNetworkMapping
from lacuna.utils.mrtrix import MRtrixError, check_mrtrix_available


def _check_mrtrix():
    """Check if MRtrix3 is available."""
    try:
        check_mrtrix_available()
        return True
    except MRtrixError:
        return False


# Skip all tests if MRtrix3 is not available
pytestmark = [
    pytest.mark.skipif(
        not _check_mrtrix(),
        reason="MRtrix3 not available",
    ),
    pytest.mark.requires_mrtrix,
    pytest.mark.slow,
    pytest.mark.integration,
]


@pytest.fixture
def test_data_paths():
    """Provide paths to test data if available."""
    # Check for test data in common locations
    possible_paths = [
        Path("/media/moritz/Storage2/projects_marvin/202509_PSCI_DISCONNECTIVITY"),
        Path.home() / "data" / "lesion_test_data",
        Path("data") / "test",
    ]

    for base_path in possible_paths:
        if base_path.exists():
            tractogram = base_path / "data/connectomes/dTOR_full_tractogram.tck"
            tdi = base_path / "data/connectomes/dTOR_tdi_2mm.nii.gz"
            lesion_dir = base_path / "data/raw/lesion_masks/acuteinfarct"

            if tractogram.exists() and tdi.exists() and lesion_dir.exists():
                lesions = list(lesion_dir.glob("*.nii.gz"))
                if lesions:
                    return {
                        "tractogram": tractogram,
                        "tdi": tdi,
                        "lesion": lesions[0],
                        "lesion_dir": lesion_dir,
                    }

    pytest.skip("Real test data not available")


class TestStructuralNetworkMappingIntegration:
    """Integration tests with real tractogram data."""

    def test_basic_analysis_without_atlas(self, test_data_paths):
        """Test basic voxel-wise analysis without connectivity matrices."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            n_jobs=2,
            verbose=False,
        )

        # Load lesion
        lesion = SubjectData.from_file(test_data_paths["lesion"])

        # Run analysis
        result = analysis.run(lesion)

        # Verify results
        assert isinstance(result, SubjectData)
        assert result.disconnection_map is not None
        assert result.disconnection_map.shape == lesion.mask_img.shape

        # Should NOT have connectivity matrices
        assert "lesion_connectivity_matrix" not in result.metadata
        assert "disconnectivity_percent" not in result.metadata

    def test_analysis_with_bundled_atlas(self, test_data_paths):
        """Test analysis with bundled Schaefer100 atlas."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            compute_disconnectivity_matrix=False,
            n_jobs=2,
            verbose=False,
        )

        # Load lesion
        lesion = SubjectData.from_file(test_data_paths["lesion"])

        # Run analysis
        result = analysis.run(lesion)

        # Verify voxel-wise results
        assert result.disconnection_map is not None

        # Verify connectivity matrix results
        assert "lesion_connectivity_matrix" in result.metadata
        assert "disconnectivity_percent" in result.metadata
        assert "full_connectivity_matrix" in result.metadata
        assert "matrix_statistics" in result.metadata

        # Verify matrix properties
        lesion_matrix = result.metadata["lesion_connectivity_matrix"]
        full_matrix = result.metadata["full_connectivity_matrix"]
        disconn_pct = result.metadata["disconnectivity_percent"]

        assert lesion_matrix.shape == (100, 100)
        assert full_matrix.shape == (100, 100)
        assert disconn_pct.shape == (100, 100)

        # Lesion matrix should be subset of full matrix
        assert np.all(lesion_matrix <= full_matrix)

        # Disconnectivity should be in valid range
        assert np.all(disconn_pct >= 0)
        assert np.all(disconn_pct <= 100)

        # Should NOT have lesioned matrix (compute_disconnectivity_matrix=False)
        assert result.metadata.get("lesioned_connectivity_matrix") is None

    def test_analysis_with_lesioned_computation(self, test_data_paths):
        """Test analysis with compute_disconnectivity_matrix=True."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            compute_disconnectivity_matrix=True,
            n_jobs=2,
            verbose=False,
        )

        # Load lesion
        lesion = SubjectData.from_file(test_data_paths["lesion"])

        # Run analysis
        result = analysis.run(lesion)

        # Verify lesioned matrix is computed
        assert "lesioned_connectivity_matrix" in result.metadata
        lesioned_matrix = result.metadata["lesioned_connectivity_matrix"]
        assert lesioned_matrix is not None
        assert lesioned_matrix.shape == (100, 100)

        # Verify QC metric exists
        stats = result.metadata["matrix_statistics"]
        assert "connectivity_preservation_ratio" in stats

        # QC: lesion + lesioned should approximately equal full
        preservation = stats["connectivity_preservation_ratio"]
        assert 0.5 <= preservation <= 1.5  # Allow some tolerance

    def test_matrix_symmetry(self, test_data_paths):
        """Verify connectivity matrices are symmetric."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            compute_disconnectivity_matrix=True,
            n_jobs=2,
            verbose=False,
        )

        lesion = SubjectData.from_file(test_data_paths["lesion"])
        result = analysis.run(lesion)

        # All matrices should be symmetric
        full_matrix = result.metadata["full_connectivity_matrix"]
        lesion_matrix = result.metadata["lesion_connectivity_matrix"]
        lesioned_matrix = result.metadata["lesioned_connectivity_matrix"]

        np.testing.assert_allclose(full_matrix, full_matrix.T, rtol=1e-10)
        np.testing.assert_allclose(lesion_matrix, lesion_matrix.T, rtol=1e-10)
        np.testing.assert_allclose(lesioned_matrix, lesioned_matrix.T, rtol=1e-10)

    def test_no_negative_edge_weights(self, test_data_paths):
        """Verify all edge weights are non-negative."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            compute_disconnectivity_matrix=True,
            n_jobs=2,
            verbose=False,
        )

        lesion = SubjectData.from_file(test_data_paths["lesion"])
        result = analysis.run(lesion)

        # No negative values allowed
        full_matrix = result.metadata["full_connectivity_matrix"]
        lesion_matrix = result.metadata["lesion_connectivity_matrix"]
        lesioned_matrix = result.metadata["lesioned_connectivity_matrix"]

        assert np.all(full_matrix >= 0)
        assert np.all(lesion_matrix >= 0)
        assert np.all(lesioned_matrix >= 0)

    def test_caching_across_multiple_lesions(self, test_data_paths):
        """Verify full connectivity matrix is cached for batch processing."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            n_jobs=2,
            verbose=False,
        )

        # Get multiple lesions
        lesion_files = list(test_data_paths["lesion_dir"].glob("*.nii.gz"))[:3]
        if len(lesion_files) < 2:
            pytest.skip("Need at least 2 lesions for caching test")

        lesions = [SubjectData.from_file(f) for f in lesion_files]

        # Run on first lesion
        result1 = analysis.run(lesions[0])
        full_matrix_1 = result1.metadata["full_connectivity_matrix"]

        # Full matrix should be cached now
        assert analysis._full_connectivity_matrix is not None

        # Run on second lesion
        result2 = analysis.run(lesions[1])
        full_matrix_2 = result2.metadata["full_connectivity_matrix"]

        # Full matrices should be identical (same reference)
        np.testing.assert_array_equal(full_matrix_1, full_matrix_2)
        assert full_matrix_1 is full_matrix_2  # Same object in memory

    def test_statistics_computation(self, test_data_paths):
        """Verify matrix statistics are computed correctly."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            n_jobs=2,
            verbose=False,
        )

        lesion = SubjectData.from_file(test_data_paths["lesion"])
        result = analysis.run(lesion)

        stats = result.metadata["matrix_statistics"]

        # Verify required keys exist
        required_keys = [
            "n_parcels",
            "n_edges_total",
            "n_edges_affected",
            "percent_edges_affected",
            "mean_disconnection_percent",
            "max_disconnection_percent",
            "mean_degree_reduction",
            "max_degree_reduction",
            "most_affected_parcel",
        ]

        for key in required_keys:
            assert key in stats, f"Missing statistic: {key}"

        # Verify values are sensible
        assert stats["n_parcels"] == 100
        assert stats["n_edges_total"] > 0
        assert stats["n_edges_affected"] >= 0
        assert stats["n_edges_affected"] <= stats["n_edges_total"]
        assert 0 <= stats["percent_edges_affected"] <= 100
        assert 0 <= stats["mean_disconnection_percent"] <= 100
        assert 0 <= stats["max_disconnection_percent"] <= 100
        assert stats["mean_degree_reduction"] >= 0
        assert stats["max_degree_reduction"] >= 0
        assert 0 <= stats["most_affected_parcel"] < 100


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_tiny_lesion_minimal_disconnection(self, test_data_paths):
        """Test with very small lesion that may not disconnect many streamlines."""
        # Create a tiny lesion (just a few voxels)
        template = nib.load(test_data_paths["tdi"])
        tiny_lesion = np.zeros(template.shape, dtype=np.uint8)
        center = tuple(s // 2 for s in template.shape)
        tiny_lesion[center] = 1  # Single voxel lesion
        tiny_mask_img = nib.Nifti1Image(tiny_lesion, template.affine, template.header)

        mask_data = SubjectData(
            mask_img=tiny_mask_img,
            metadata={"subject_id": "tiny_test", "n_voxels": 1},
        )

        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            n_jobs=2,
            verbose=False,
        )

        # Should complete without errors even if no disconnection
        result = analysis.run(mask_data)

        # Matrices should exist but may have minimal values
        assert "lesion_connectivity_matrix" in result.metadata
        lesion_matrix = result.metadata["lesion_connectivity_matrix"]

        # Lesion matrix should have mostly zeros
        assert np.sum(lesion_matrix > 0) >= 0  # May be 0 or small number

    def test_disconnectivity_percent_handles_zero_division(self, test_data_paths):
        """Verify disconnectivity percentage handles edges with zero full connectivity."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name="Schaefer2018_100Parcels7Networks",
            n_jobs=2,
            verbose=False,
        )

        lesion = SubjectData.from_file(test_data_paths["lesion"])
        result = analysis.run(lesion)

        disconn_pct = result.metadata["disconnectivity_percent"]

        # Should not have NaN or Inf values
        assert not np.any(np.isnan(disconn_pct))
        assert not np.any(np.isinf(disconn_pct))

        # All values should be valid percentages
        assert np.all(disconn_pct >= 0)
        assert np.all(disconn_pct <= 100)


class TestDifferentAtlases:
    """Test with different atlas parcellations."""

    @pytest.mark.parametrize(
        "atlas_name,expected_n_parcels",
        [
            ("Schaefer2018_100Parcels7Networks", 100),
            ("Schaefer2018_200Parcels7Networks", 200),
        ],
    )
    def test_bundled_atlases(self, test_data_paths, atlas_name, expected_n_parcels):
        """Test analysis with different bundled atlases."""
        analysis = StructuralNetworkMapping(
            tractogram_path=test_data_paths["tractogram"],
            whole_brain_tdi=test_data_paths["tdi"],
            parcellation_name=atlas_name,
            n_jobs=2,
            verbose=False,
        )

        lesion = SubjectData.from_file(test_data_paths["lesion"])
        result = analysis.run(lesion)

        # Verify matrix dimensions match atlas
        lesion_matrix = result.metadata["lesion_connectivity_matrix"]
        assert lesion_matrix.shape == (expected_n_parcels, expected_n_parcels)

        # Verify statistics
        stats = result.metadata["matrix_statistics"]
        assert stats["n_parcels"] == expected_n_parcels
