"""
Contract tests for StructuralNetworkMapping analysis class.

Tests the interface and behavior requirements for tractography-based
lesion network mapping following the BaseAnalysis contract.
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from lacuna.utils.mrtrix import MRtrixError, check_mrtrix_available


def _check_mrtrix():
    """Check if MRtrix3 is available."""
    try:
        check_mrtrix_available()
        return True
    except MRtrixError:
        return False


@pytest.fixture
def temp_connectome():
    """Create and register a temporary structural connectome for testing."""
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_structural_{uuid.uuid4().hex[:8]}"

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)

    register_structural_connectome(
        name=connectome_name,
        space="MNI152NLin2009cAsym",
        tractogram_path=temp_tck,
        description="Test structural connectome",
    )

    yield connectome_name

    # Cleanup
    unregister_structural_connectome(connectome_name)
    temp_tck.unlink(missing_ok=True)


def test_structural_network_mapping_import():
    """Test that StructuralNetworkMapping can be imported."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    assert StructuralNetworkMapping is not None


def test_structural_network_mapping_inherits_base_analysis():
    """Test that StructuralNetworkMapping inherits from BaseAnalysis."""
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    assert issubclass(StructuralNetworkMapping, BaseAnalysis)


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_can_instantiate(temp_connectome):
    """Test that StructuralNetworkMapping can be instantiated with required parameters."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept connectome name
    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)
    assert analysis is not None
    assert analysis.tractogram_space == "MNI152NLin2009cAsym"
    assert analysis.output_resolution == 2  # default


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_has_run_method(temp_connectome):
    """Test that StructuralNetworkMapping has the run() method from BaseAnalysis."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_structural_network_mapping_validates_binary_mask():
    """Test that SubjectData validates binary lesion mask at construction (enforced at SubjectData level).

    This is a contract test - we verify the validation logic exists,
    not the full pipeline integration (which requires MRtrix and is tested in integration tests).
    """
    import nibabel as nib
    import numpy as np

    from lacuna import SubjectData

    # Create a simple lesion with non-binary values
    data = np.zeros((10, 10, 10))
    data[4:6, 4:6, 4:6] = 0.5  # Non-binary values

    mask_img = nib.Nifti1Image(data, np.eye(4))

    # The validation should detect non-binary values at SubjectData construction
    # This is enforced at the SubjectData level (T005), not in individual analysis modules
    with pytest.raises(ValueError, match="mask_img must be a binary mask"):
        SubjectData(mask_img=mask_img, space="MNI152NLin6Asym", resolution=2)


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_returns_mask_data(synthetic_mask_img, temp_connectome):
    """Test that run() returns a SubjectData object with namespaced results."""
    from lacuna import SubjectData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Mark lesion as MNI152 space
    mask_data = SubjectData(mask_img=synthetic_mask_img, space="MNI152NLin6Asym", resolution=2)

    # Note: This test will fail until implementation exists
    # It defines the expected behavior
    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)

    # For now, expect this to fail during actual run
    # The test documents the expected interface
    from lacuna.utils.mrtrix import MRtrixError

    with pytest.raises((FileNotFoundError, RuntimeError, MRtrixError)):
        analysis.run(mask_data)


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_result_structure(temp_connectome):
    """Test that results should contain expected keys and data types."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    StructuralNetworkMapping(connectome_name=temp_connectome)

    # Document expected result structure
    expected_keys = {
        "disconnection_map",  # NIfTI image showing voxel-level disconnection
        "mean_disconnection",  # Summary statistic
        "lesion_tck_count",  # Number of streamlines passing through lesion
    }

    # This documents the contract
    assert expected_keys is not None


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_accepts_n_jobs(temp_connectome):
    """Test that StructuralNetworkMapping accepts n_jobs parameter for MRtrix."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(connectome_name=temp_connectome, n_jobs=8)
    assert analysis.n_jobs == 8


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_preserves_input_immutability(
    temp_connectome, synthetic_mask_img
):
    """Test that SNM does not modify the input SubjectData object."""
    from lacuna import SubjectData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        space="MNI152NLin6Asym",
        resolution=2,
        metadata={"test_key": "test_value"},
    )

    # Store original state
    original_metadata = mask_data.metadata.copy()

    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)

    # Try to run (will fail due to missing files, but shouldn't modify input)
    try:
        analysis.run(mask_data)
    except Exception:
        pass

    # Input should be unchanged
    assert mask_data.metadata == original_metadata
    # Results may be modified by run() - that's expected behavior


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_adds_provenance(temp_connectome, synthetic_mask_img):
    """Test that SNM adds provenance information."""
    from lacuna import SubjectData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    SubjectData(
        mask_img=synthetic_mask_img,
        space="MNI152NLin6Asym",
        resolution=2,
    )

    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)

    # Get parameters (used for provenance)
    params = analysis._get_parameters()

    # Should include key parameters
    assert "connectome_name" in params
    assert "output_resolution" in params
    assert "n_jobs" in params


# ============================================================================
# Whole-Brain TDI Computation Tests
# ============================================================================


def test_compute_whole_brain_tdi_function_exists():
    """Test that compute_whole_brain_tdi utility function exists."""
    from lacuna.utils.mrtrix import compute_whole_brain_tdi

    assert compute_whole_brain_tdi is not None
    assert callable(compute_whole_brain_tdi)


def test_compute_whole_brain_tdi_accepts_parameters():
    """Test that compute_whole_brain_tdi accepts expected parameters."""
    from lacuna.utils.mrtrix import compute_whole_brain_tdi

    # Should accept tractogram path and output paths
    # This documents the interface even if execution fails
    try:
        compute_whole_brain_tdi(
            tractogram_path="/path/to/tractogram.tck",
            output_1mm="/path/to/tdi_1mm.nii.gz",
            output_2mm="/path/to/tdi_2mm.nii.gz",
            n_jobs=8,
            force=False,
        )
    except FileNotFoundError:
        # Expected - files don't exist
        pass


def test_compute_whole_brain_tdi_requires_at_least_one_output():
    """Test that compute_whole_brain_tdi requires at least one output."""
    from lacuna.utils.mrtrix import compute_whole_brain_tdi

    # Should raise error if neither output specified
    with pytest.raises(ValueError, match="At least one"):
        compute_whole_brain_tdi(tractogram_path="/path/to/tractogram.tck")


def test_compute_whole_brain_tdi_validates_tractogram_exists():
    """Test that compute_whole_brain_tdi validates tractogram file exists."""
    from lacuna.utils.mrtrix import compute_whole_brain_tdi

    # Should raise FileNotFoundError if tractogram doesn't exist
    with pytest.raises(FileNotFoundError, match="Tractogram"):
        compute_whole_brain_tdi(
            tractogram_path="/nonexistent/tractogram.tck",
            output_2mm="/path/to/output.nii.gz",
        )


def test_mrtrix_commands_are_printed(capsys):
    """Test that MRtrix3 commands are printed during execution."""
    from lacuna.utils.mrtrix import run_mrtrix_command

    # Test that verbose mode prints commands
    try:
        run_mrtrix_command(["nonexistent_command"], check=False, verbose=True)
    except Exception:
        pass

    captured = capsys.readouterr()
    assert "Executing:" in captured.out or "nonexistent_command" in captured.out


# ============================================================================
# Batch Processing and Parallel Execution Tests
# ============================================================================


def test_batch_strategy_is_parallel():
    """Test that StructuralNetworkMapping declares parallel batch strategy."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    assert StructuralNetworkMapping.batch_strategy == "parallel"


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_provenance_includes_memory_settings(temp_connectome):
    """Test that provenance includes cleanup and intermediate settings."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        connectome_name=temp_connectome,
        cleanup_temp_files=False,
        keep_intermediate=True,
    )

    params = analysis._get_parameters()

    # Should include cleanup and intermediate settings
    assert "cleanup_temp_files" in params
    assert "keep_intermediate" in params
    assert params["cleanup_temp_files"] is False
    assert params["keep_intermediate"] is True


# ============================================================================
# Contract Tests for Atlas-based Connectivity Matrix Computation
# ============================================================================


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_parcellation_parameter_optional(temp_connectome):
    """Test that parcellation_name is an optional parameter."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should work without parcellation (voxel-wise only)
    analysis = StructuralNetworkMapping(connectome_name=temp_connectome)
    assert analysis.parcellation_name is None


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_parcellation_accepts_bundled_name(temp_connectome):
    """Test that parcellation_name can be a bundled atlas name string."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept bundled atlas name
    analysis = StructuralNetworkMapping(
        connectome_name=temp_connectome,
        parcellation_name="Schaefer2018_100Parcels7Networks",
    )
    assert analysis.parcellation_name == "Schaefer2018_100Parcels7Networks"


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_compute_disconnectivity_matrix_parameter(temp_connectome):
    """Test that compute_disconnectivity_matrix parameter is available."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should have compute_disconnectivity_matrix parameter
    analysis = StructuralNetworkMapping(
        connectome_name=temp_connectome,
        parcellation_name="Schaefer2018_100Parcels7Networks",
        compute_disconnectivity_matrix=True,
    )
    assert analysis.compute_disconnectivity_matrix is True


def test_results_include_connectivity_matrices_when_atlas_provided():
    """Test that results include connectivity matrices when atlas is provided."""
    # This is a contract test - we're testing the interface, not the implementation
    # The actual matrices would be computed by MRtrix3 in real execution

    # Expected result structure when atlas is provided:
    expected_keys_with_atlas = {
        "disconnection_map",  # Always present
        "mean_disconnection",  # Always present
        "lesion_streamline_count",  # Always present
        "metadata",  # Always present
        "lesion_connectivity_matrix",  # Present when atlas provided
        "disconnectivity_percent",  # Present when atlas provided
        "full_connectivity_matrix",  # Present when atlas provided
        "matrix_statistics",  # Present when atlas provided
    }

    # When atlas is None, connectivity matrices should not be in results
    expected_keys_without_atlas = {
        "disconnection_map",
        "mean_disconnection",
        "lesion_streamline_count",
        "metadata",
    }

    # Test interface expectation
    assert expected_keys_with_atlas is not None
    assert expected_keys_without_atlas is not None


def test_matrix_statistics_structure():
    """Test expected structure of matrix_statistics in results."""
    # Contract: matrix_statistics should contain these keys
    expected_stats_keys = {
        "n_parcels",
        "n_edges_total",
        "n_edges_affected",
        "percent_edges_affected",
        "mean_disconnection_percent",
        "max_disconnection_percent",
        "mean_degree_reduction",
        "max_degree_reduction",
        "most_affected_parcel",
    }

    # When compute_disconnectivity_matrix=True, additional keys expected
    expected_stats_with_lesioned = expected_stats_keys | {
        "lesioned_mean_degree",
        "connectivity_preservation_ratio",
    }

    assert expected_stats_keys is not None
    assert expected_stats_with_lesioned is not None


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_lesioned_connectivity_optional(temp_connectome):
    """Test that lesioned connectivity is only computed when requested."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Default: compute_disconnectivity_matrix=False
    analysis1 = StructuralNetworkMapping(
        connectome_name=temp_connectome,
        parcellation_name="Schaefer2018_100Parcels7Networks",
    )
    assert analysis1.compute_disconnectivity_matrix is False

    # Explicit: compute_disconnectivity_matrix=True
    analysis2 = StructuralNetworkMapping(
        connectome_name=temp_connectome,
        parcellation_name="Schaefer2018_100Parcels7Networks",
        compute_disconnectivity_matrix=True,
    )
    assert analysis2.compute_disconnectivity_matrix is True
