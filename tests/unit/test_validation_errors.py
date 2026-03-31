"""
Unit tests for input validation across CLI and API.

Tests that invalid inputs are caught early with clear error messages,
rather than silently ignored or caught after expensive computation.
"""

import tempfile
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_connectome():
    """Register a temporary structural connectome (no MRtrix needed)."""
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    name = f"test_snm_validation_{uuid.uuid4().hex[:8]}"
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)

    register_structural_connectome(
        name=name,
        space="MNI152NLin2009cAsym",
        tractogram_path=temp_tck,
        description="Test connectome for validation tests",
    )
    yield name
    unregister_structural_connectome(name)
    temp_tck.unlink(missing_ok=True)


@pytest.fixture
def temp_functional_connectome(tmp_path):
    """Register a temporary functional connectome."""
    import h5py
    import numpy as np

    from lacuna.assets.connectomes import (
        register_functional_connectome,
        unregister_functional_connectome,
    )

    name = f"test_fnm_validation_{uuid.uuid4().hex[:8]}"

    # Create minimal HDF5 file
    h5_path = tmp_path / "test_connectome.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("timeseries", data=np.zeros((10, 5)))
        f.create_dataset("mask_indices", data=np.array([[0, 0, 0], [1, 1, 1]]))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = [3, 3, 3]

    register_functional_connectome(
        name=name,
        space="MNI152NLin6Asym",
        data_path=h5_path,
        resolution=2,
        description="Test connectome for validation tests",
    )
    yield name
    unregister_functional_connectome(name)


# ---------------------------------------------------------------------------
# SNM: compute flags require parcellation
# ---------------------------------------------------------------------------


class TestSNMComputeFlagValidation:
    """Test that SNM compute flags require parcellation_name."""

    def test_compute_disconnectivity_matrix_without_parcellation_raises(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="parcellation_name"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                compute_disconnectivity_matrix=True,
                check_dependencies=False,
            )

    def test_compute_roi_disconnection_without_parcellation_raises(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="parcellation_name"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                compute_roi_disconnection=True,
                check_dependencies=False,
            )

    def test_both_compute_flags_without_parcellation_raises(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="parcellation_name"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                compute_disconnectivity_matrix=True,
                compute_roi_disconnection=True,
                check_dependencies=False,
            )


# ---------------------------------------------------------------------------
# SNM: early parcellation name validation
# ---------------------------------------------------------------------------


class TestSNMParcellationNameValidation:
    """Test that SNM validates parcellation name early in __init__."""

    def test_invalid_parcellation_name_raises_immediately(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="not found in registry"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                parcellation_name="NonexistentAtlas_999Parcels",
                compute_disconnectivity_matrix=True,
                check_dependencies=False,
            )

    def test_invalid_parcellation_error_lists_available(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="Available parcellations"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                parcellation_name="NotAnAtlas",
                compute_disconnectivity_matrix=True,
                check_dependencies=False,
            )


# ---------------------------------------------------------------------------
# SNM: n_jobs validation
# ---------------------------------------------------------------------------


class TestSNMNJobsValidation:
    """Test that SNM validates n_jobs parameter."""

    def test_n_jobs_zero_raises(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="n_jobs"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                n_jobs=0,
                check_dependencies=False,
            )

    def test_n_jobs_negative_two_raises(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        with pytest.raises(ValueError, match="n_jobs"):
            StructuralNetworkMapping(
                connectome_name=temp_connectome,
                n_jobs=-2,
                check_dependencies=False,
            )

    def test_n_jobs_minus_one_accepted(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        analysis = StructuralNetworkMapping(
            connectome_name=temp_connectome,
            n_jobs=-1,
            check_dependencies=False,
        )
        assert analysis.n_jobs == -1

    def test_n_jobs_positive_accepted(self, temp_connectome):
        from lacuna.analysis.structural_network_mapping import (
            StructuralNetworkMapping,
        )

        analysis = StructuralNetworkMapping(
            connectome_name=temp_connectome,
            n_jobs=4,
            check_dependencies=False,
        )
        assert analysis.n_jobs == 4


# ---------------------------------------------------------------------------
# FNM: range validation
# ---------------------------------------------------------------------------


class TestFNMRangeValidation:
    """Test that FNM validates pini_percentile and fdr_alpha ranges."""

    def test_pini_percentile_below_1_raises(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        with pytest.raises(ValueError, match="pini_percentile"):
            FunctionalNetworkMapping(
                connectome_name=temp_functional_connectome,
                pini_percentile=0,
            )

    def test_pini_percentile_above_100_raises(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        with pytest.raises(ValueError, match="pini_percentile"):
            FunctionalNetworkMapping(
                connectome_name=temp_functional_connectome,
                pini_percentile=101,
            )

    def test_fdr_alpha_zero_raises(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        with pytest.raises(ValueError, match="fdr_alpha"):
            FunctionalNetworkMapping(
                connectome_name=temp_functional_connectome,
                fdr_alpha=0.0,
            )

    def test_fdr_alpha_negative_raises(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        with pytest.raises(ValueError, match="fdr_alpha"):
            FunctionalNetworkMapping(
                connectome_name=temp_functional_connectome,
                fdr_alpha=-0.1,
            )

    def test_fdr_alpha_above_1_raises(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        with pytest.raises(ValueError, match="fdr_alpha"):
            FunctionalNetworkMapping(
                connectome_name=temp_functional_connectome,
                fdr_alpha=1.5,
            )

    def test_fdr_alpha_valid_accepted(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        analysis = FunctionalNetworkMapping(
            connectome_name=temp_functional_connectome,
            fdr_alpha=0.05,
        )
        assert analysis.fdr_alpha == 0.05

    def test_fdr_alpha_none_accepted(self, temp_functional_connectome):
        from lacuna.analysis.functional_network_mapping import (
            FunctionalNetworkMapping,
        )

        analysis = FunctionalNetworkMapping(
            connectome_name=temp_functional_connectome,
            fdr_alpha=None,
        )
        assert analysis.fdr_alpha is None


# ---------------------------------------------------------------------------
# CLI: atlas name validation for RD/FNM
# ---------------------------------------------------------------------------


class TestCLIAtlasNameValidation:
    """Test that CLI validates atlas names for all analysis types."""

    def test_rd_invalid_atlas_raises(self, tmp_path):
        from lacuna.cli.main import RunConfig

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = RunConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis="rd",
            analysis_options={"parcel_names": ["NonexistentAtlas"]},
        )
        with pytest.raises(ValueError, match="not found"):
            config.validate()

    def test_fnm_invalid_atlas_raises(self, tmp_path):
        from lacuna.cli.main import RunConfig

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = RunConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis="fnm",
            analysis_options={"parcel_names": ["NonexistentAtlas"]},
        )
        with pytest.raises(ValueError, match="not found"):
            config.validate()

    def test_rd_valid_atlas_passes(self, tmp_path):
        from lacuna.assets.parcellations import list_parcellations
        from lacuna.cli.main import RunConfig

        available = [a.name for a in list_parcellations()]
        if not available:
            pytest.skip("No parcellations registered")

        slug = available[0]
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = RunConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis="rd",
            analysis_options={"parcel_names": [slug]},
        )
        # Should not raise — slug is a valid registry name
        config.validate()
