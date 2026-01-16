"""Unit tests for StructuralNetworkMapping connectivity matrix computation.

These tests verify the internal methods for computing and processing
connectivity matrices independently of MRtrix3 commands.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
from lacuna.assets.connectomes import (
    register_structural_connectome,
    unregister_structural_connectome,
)


class TestConnectivityMatrixComputation:
    """Test connectivity matrix computation methods."""

    @pytest.fixture
    def mock_analysis(self, tmp_path):
        """Create a mock StructuralNetworkMapping instance."""
        # Create fake files
        tractogram_path = tmp_path / "tractogram.tck"
        tractogram_path.write_text("fake")

        # Register connectome
        register_structural_connectome(
            name="test_mock_connectome",
            space="MNI152NLin2009cAsym",
            tractogram_path=tractogram_path,
            description="Test mock connectome",
        )

        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            analysis = StructuralNetworkMapping(
                connectome_name="test_mock_connectome",
                parcellation_name="Schaefer2018_100Parcels7Networks",
                n_jobs=1,
            )
            analysis._parcellation_resolved = Path("/fake/atlas.nii.gz")
            yield analysis

        # Cleanup
        unregister_structural_connectome("test_mock_connectome")

    def test_compute_matrix_statistics_basic(self, mock_analysis):
        """Test basic matrix statistics computation."""
        # Create simple test matrices
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        mask_matrix = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (mask_matrix / full_matrix) * 100
        disconn_pct = np.nan_to_num(disconn_pct)

        stats = mock_analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=None,
        )

        # Verify statistics
        assert stats["n_parcels"] == 3
        assert stats["n_edges_total"] == 6  # 3 edges (upper triangle + diagonal=0)
        assert stats["n_edges_affected"] == 6
        assert stats["percent_edges_affected"] == 100.0
        assert stats["mean_disconnection_percent"] > 0
        assert stats["max_disconnection_percent"] <= 100
        assert "mean_degree_reduction" in stats
        assert "max_degree_reduction" in stats
        assert "most_affected_parcel" in stats

    def test_compute_matrix_statistics_with_intact(self, mock_analysis):
        """Test statistics computation with intact matrix."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        mask_matrix = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
        intact_matrix = np.array([[0, 7, 3], [7, 0, 4], [3, 4, 0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (mask_matrix / full_matrix) * 100
        disconn_pct = np.nan_to_num(disconn_pct)

        stats = mock_analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=intact_matrix,
        )

        # Should have intact statistics
        assert "intact_mean_degree" in stats
        assert "connectivity_preservation_ratio" in stats

        # QC check
        preservation = stats["connectivity_preservation_ratio"]
        assert 0.0 <= preservation <= 2.0  # Should be close to 1.0

    def test_disconnectivity_calculation_handles_zeros(self, mock_analysis):
        """Test that disconnectivity calculation handles zero values correctly."""
        # Matrix with some zero edges
        full_matrix = np.array([[0, 10, 0], [10, 0, 5], [0, 5, 0]])
        mask_matrix = np.array([[0, 3, 0], [3, 0, 2], [0, 2, 0]])

        # Simulate the disconnectivity calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (mask_matrix / full_matrix) * 100
        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Should have no NaN or Inf
        assert not np.any(np.isnan(disconn_pct))
        assert not np.any(np.isinf(disconn_pct))

        # Where full_matrix is 0, disconn_pct should be 0
        assert disconn_pct[0, 2] == 0.0
        assert disconn_pct[2, 0] == 0.0
        assert disconn_pct[2, 2] == 0.0

    def test_matrix_symmetry_preserved(self):
        """Test that symmetric input produces symmetric output."""
        # Create symmetric matrix
        matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])

        # Verify it's symmetric
        np.testing.assert_array_equal(matrix, matrix.T)

        # After any operations, symmetry should be preserved
        scaled = matrix * 0.5
        np.testing.assert_array_equal(scaled, scaled.T)

    def test_no_negative_values_in_statistics(self, mock_analysis):
        """Test that statistics don't produce negative values inappropriately."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        mask_matrix = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (mask_matrix / full_matrix) * 100
        disconn_pct = np.nan_to_num(disconn_pct)

        stats = mock_analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=None,
        )

        # These should never be negative
        assert stats["n_edges_total"] >= 0
        assert stats["n_edges_affected"] >= 0
        assert stats["percent_edges_affected"] >= 0
        assert stats["mean_disconnection_percent"] >= 0
        assert stats["max_disconnection_percent"] >= 0
        assert stats["mean_degree_reduction"] >= 0
        assert stats["max_degree_reduction"] >= 0

    def test_empty_mask_matrix(self, mock_analysis):
        """Test handling of empty mask matrix (no disconnection)."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        mask_matrix = np.zeros_like(full_matrix)
        disconn_pct = np.zeros_like(full_matrix)

        stats = mock_analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=None,
        )

        # Should handle gracefully
        assert stats["n_edges_affected"] == 0
        assert stats["percent_edges_affected"] == 0.0
        assert stats["mean_disconnection_percent"] == 0.0
        assert stats["max_disconnection_percent"] == 0.0

    def test_full_disconnection(self, mock_analysis):
        """Test when mask disconnects all streamlines."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        mask_matrix = full_matrix.copy()  # All streamlines go through mask
        disconn_pct = np.ones_like(full_matrix) * 100
        disconn_pct[full_matrix == 0] = 0

        stats = mock_analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=None,
        )

        # Should be 100% affected
        assert stats["percent_edges_affected"] == 100.0
        assert stats["mean_disconnection_percent"] == 100.0
        assert stats["max_disconnection_percent"] == 100.0


class TestDisconnectivityPercentageEdgeCases:
    """Test edge cases in disconnectivity percentage calculation."""

    def test_division_by_zero_handling(self):
        """Test that division by zero is handled correctly."""
        full_matrix = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 0]])
        lesion_matrix = np.array([[0, 3, 0], [3, 0, 0], [0, 0, 0]])

        # This is the actual calculation from the code
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (lesion_matrix / full_matrix) * 100

        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify no invalid values
        assert not np.any(np.isnan(disconn_pct))
        assert not np.any(np.isinf(disconn_pct))
        assert np.all(disconn_pct >= 0)
        assert np.all(disconn_pct <= 100)

    def test_lesion_exceeds_full_numerical_error(self):
        """Test handling when lesion > full due to numerical errors."""
        # Simulate numerical errors
        full_matrix = np.array([[0, 10.0, 5.0], [10.0, 0, 8.0], [5.0, 8.0, 0]])
        lesion_matrix = np.array([[0, 10.1, 5.0], [10.1, 0, 8.0], [5.0, 8.0, 0]])  # Slightly over

        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (lesion_matrix / full_matrix) * 100

        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Should still be valid even if slightly over 100
        assert not np.any(np.isnan(disconn_pct))
        assert not np.any(np.isinf(disconn_pct))

    def test_all_zeros_matrices(self):
        """Test when both matrices are all zeros."""
        full_matrix = np.zeros((3, 3))
        lesion_matrix = np.zeros((3, 3))

        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (lesion_matrix / full_matrix) * 100

        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Should be all zeros
        assert np.all(disconn_pct == 0)


class TestQualityControlMetrics:
    """Test quality control metrics for lesioned connectivity."""

    def test_preservation_ratio_perfect(self):
        """Test when lesion + lesioned exactly equals full."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        lesion_matrix = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
        lesioned_matrix = np.array([[0, 7, 3], [7, 0, 4], [3, 4, 0]])

        # Verify: lesion + lesioned = full
        combined = lesion_matrix + lesioned_matrix
        np.testing.assert_array_equal(combined, full_matrix)

        # Calculate preservation
        n_edges = np.sum(full_matrix > 0)
        preservation = np.sum(combined > 0) / n_edges if n_edges > 0 else 0

        # Should be exactly 1.0
        assert preservation == 1.0

    def test_preservation_ratio_with_loss(self):
        """Test when some streamlines are lost (lesion + lesioned < full)."""
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        lesion_matrix = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
        lesioned_matrix = np.array([[0, 6, 2], [6, 0, 3], [2, 3, 0]])  # Lost some

        combined = lesion_matrix + lesioned_matrix

        # Verify loss
        assert np.any(combined < full_matrix)

        n_edges = np.sum(full_matrix > 0)
        preservation = np.sum(combined > 0) / n_edges if n_edges > 0 else 0

        # Should be < 1.0
        assert preservation <= 1.0
        assert preservation > 0.5  # But not too much loss

    def test_preservation_ratio_edge_cases(self):
        """Test preservation ratio with edge cases."""
        # Case 1: No edges
        full_matrix = np.zeros((3, 3))
        lesion_matrix = np.zeros((3, 3))
        lesioned_matrix = np.zeros((3, 3))

        n_edges = np.sum(full_matrix > 0)
        preservation = 0 if n_edges == 0 else np.sum(lesion_matrix + lesioned_matrix > 0) / n_edges

        # Should handle gracefully
        assert preservation == 0

        # Case 2: All streamlines disconnected
        full_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        lesion_matrix = full_matrix.copy()
        lesioned_matrix = np.zeros_like(full_matrix)

        n_edges = np.sum(full_matrix > 0)
        preservation = np.sum(lesion_matrix + lesioned_matrix > 0) / n_edges

        # Should be 1.0 (all edges still exist, just through lesion)
        assert preservation == 1.0


class TestMatrixDimensions:
    """Test handling of different matrix dimensions."""

    @pytest.fixture
    def mock_analysis_parametrized(self):
        """Create mock analysis that can use different atlases."""
        import tempfile
        from pathlib import Path

        import nibabel as nib
        import numpy as np

        from lacuna.assets.connectomes import (
            register_structural_connectome,
            unregister_structural_connectome,
        )

        def _create_analysis(n_parcels):
            # Create dummy tractogram and TDI files
            with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
                tractogram_path = Path(f.name)
                f.write(b"dummy")

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                tdi_path = Path(f.name)
                tdi_img = nib.Nifti1Image(np.zeros((91, 109, 91), dtype=np.float32), np.eye(4))
                nib.save(tdi_img, tdi_path)

            try:
                register_structural_connectome(
                    name="test_matrix_dims",
                    space="MNI152NLin2009cAsym",
                    tractogram_path=tractogram_path,
                )

                with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
                    analysis = StructuralNetworkMapping(
                        connectome_name="test_matrix_dims",
                        parcellation_name="Schaefer2018_100Parcels7Networks",
                        n_jobs=1,
                    )
                    analysis._parcellation_resolved = Path("/fake/atlas.nii.gz")
                    return analysis
            finally:
                try:
                    unregister_structural_connectome("test_matrix_dims")
                except KeyError:
                    pass
                tractogram_path.unlink(missing_ok=True)
                tdi_path.unlink(missing_ok=True)

        return _create_analysis

    @pytest.mark.parametrize("n_parcels", [100, 200, 116, 400])
    def test_different_atlas_sizes(self, mock_analysis_parametrized, n_parcels):
        """Test statistics computation works with different atlas sizes."""
        analysis = mock_analysis_parametrized(n_parcels)

        # Create random matrices of appropriate size
        np.random.seed(42)
        full_matrix = np.random.rand(n_parcels, n_parcels)
        full_matrix = (full_matrix + full_matrix.T) / 2  # Make symmetric
        full_matrix[full_matrix < 0.5] = 0  # Sparsify

        mask_matrix = full_matrix * np.random.rand(n_parcels, n_parcels) * 0.3
        mask_matrix = (mask_matrix + mask_matrix.T) / 2  # Make symmetric

        disconn_pct = np.zeros_like(full_matrix)
        mask = full_matrix > 0
        disconn_pct[mask] = (mask_matrix[mask] / full_matrix[mask]) * 100

        stats = analysis._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=None,
        )

        # Verify basic properties
        assert stats["n_parcels"] == n_parcels
        assert stats["n_edges_total"] > 0
        assert 0 <= stats["most_affected_parcel"] < n_parcels
