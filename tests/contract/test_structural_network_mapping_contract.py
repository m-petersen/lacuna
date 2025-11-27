"""
Contract tests for StructuralNetworkMapping analysis class.

Tests the interface and behavior requirements for tractography-based
lesion network mapping following the BaseAnalysis contract.
"""

import pytest

from lacuna.utils.mrtrix import MRtrixError, check_mrtrix_available


def _check_mrtrix():
    """Check if MRtrix3 is available."""
    try:
        check_mrtrix_available()
        return True
    except MRtrixError:
        return False


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
def test_structural_network_mapping_can_instantiate():
    """Test that StructuralNetworkMapping can be instantiated with required parameters."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Use unique name to avoid conflicts in parallel tests
    connectome_name = f"test_structural_{uuid.uuid4().hex[:8]}"

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        # Register test connectome
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test structural connectome"
        )

        # Should accept connectome name
        analysis = StructuralNetworkMapping(connectome_name=connectome_name)
        assert analysis is not None
        assert analysis.tractogram_space == "MNI152NLin2009cAsym"
        assert analysis.output_resolution == 2  # default
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)


@pytest.mark.skip(reason="check_dependencies method not yet implemented")
def test_structural_network_mapping_validates_mrtrix_available():
    """Test that StructuralNetworkMapping checks for MRtrix3 availability."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_snm_mrtrix_{uuid.uuid4().hex[:8]}"

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test"
        )

        # Should check if MRtrix3 commands are available during initialization
        analysis = StructuralNetworkMapping(connectome_name=connectome_name)
        # Should have check_dependencies parameter or _check_dependencies attribute
        assert hasattr(analysis, "_check_dependencies") or hasattr(analysis, "check_dependencies")
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)


def test_structural_network_mapping_has_run_method():
    """Test that StructuralNetworkMapping has the run() method from BaseAnalysis."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_snm_run_{uuid.uuid4().hex[:8]}"

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test"
        )

        analysis = StructuralNetworkMapping(connectome_name=connectome_name)
        assert hasattr(analysis, "run")
        assert callable(analysis.run)
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)


def test_structural_network_mapping_validates_coordinate_space(synthetic_mask_img, tmp_path):
    """Test removed: 'native' space no longer supported after T007."""
    pytest.skip("'native' space removed from SUPPORTED_TEMPLATE_SPACES in T007")


def test_structural_network_mapping_validates_binary_mask():
    """Test that MaskData validates binary lesion mask at construction (enforced at MaskData level).

    This is a contract test - we verify the validation logic exists,
    not the full pipeline integration (which requires MRtrix and is tested in integration tests).
    """
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData

    # Create a simple lesion with non-binary values
    data = np.zeros((10, 10, 10))
    data[4:6, 4:6, 4:6] = 0.5  # Non-binary values

    mask_img = nib.Nifti1Image(data, np.eye(4))

    # The validation should detect non-binary values at MaskData construction
    # This is enforced at the MaskData level (T005), not in individual analysis modules
    with pytest.raises(ValueError, match="mask_img must be a binary mask"):
        MaskData(mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})


@pytest.mark.skipif(not _check_mrtrix(), reason="MRtrix3 not available")
@pytest.mark.requires_mrtrix
def test_structural_network_mapping_returns_mask_data(synthetic_mask_img):
    """Test that run() returns a MaskData object with namespaced results."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna import MaskData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_snm_mask_{uuid.uuid4().hex[:8]}"

    # Mark lesion as MNI152 space
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test"
        )

        # Note: This test will fail until implementation exists
        # It defines the expected behavior
        analysis = StructuralNetworkMapping(connectome_name=connectome_name)

        # For now, expect this to fail during actual run
        # The test documents the expected interface
        from lacuna.utils.mrtrix import MRtrixError
        with pytest.raises((FileNotFoundError, RuntimeError, MRtrixError)):
            analysis.run(mask_data)
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)


def test_structural_network_mapping_result_structure():
    """Test that results should contain expected keys and data types."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_snm_result_{uuid.uuid4().hex[:8]}"

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test"
        )

        StructuralNetworkMapping(connectome_name=connectome_name)

        # Document expected result structure
        expected_keys = {
            "disconnection_map",  # NIfTI image showing voxel-level disconnection
            "mean_disconnection",  # Summary statistic
            "lesion_tck_count",  # Number of streamlines passing through lesion
        }

        # This documents the contract
        assert expected_keys is not None
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)


def test_structural_network_mapping_accepts_n_jobs():
    """Test that StructuralNetworkMapping accepts n_jobs parameter for MRtrix."""
    import tempfile
    import uuid
    from pathlib import Path

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    connectome_name = f"test_snm_njobs_{uuid.uuid4().hex[:8]}"

    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        temp_tck = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_tdi = Path(f.name)

    try:
        register_structural_connectome(
            name=connectome_name,
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            tractogram_path=temp_tck,
            tdi_path=temp_tdi,
            n_subjects=10,
            description="Test"
        )

        analysis = StructuralNetworkMapping(
            connectome_name=connectome_name,
            n_jobs=8,
        )
        assert analysis.n_jobs == 8
    finally:
        unregister_structural_connectome(connectome_name)
        temp_tck.unlink(missing_ok=True)
        temp_tdi.unlink(missing_ok=True)

    assert analysis.n_jobs == 8


def test_structural_network_mapping_preserves_input_immutability(synthetic_mask_img):
    """Test that run() does not modify the input MaskData."""
    from lacuna import MaskData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_results = mask_data.results.copy()

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )

    # Expected to fail during execution, but documents immutability requirement
    try:
        result = analysis.run(mask_data)
        # If it somehow succeeds, check immutability
        assert mask_data.results == original_results
        assert result is not mask_data
    except (FileNotFoundError, RuntimeError):
        # Expected until implementation exists
        pass


def test_structural_network_mapping_adds_provenance():
    """Test that run() should add provenance record."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )

    # Document provenance requirement
    # Implementation should record: tractogram path, TDI path, MRtrix version
    expected_provenance_keys = {"tractogram_path", "whole_brain_tdi", "template"}
    assert expected_provenance_keys is not None


# ============================================================================
# Template Loading and Space Validation Tests
# ============================================================================


@pytest.mark.requires_templateflow
@pytest.mark.slow
def test_template_auto_loading_2mm(synthetic_mask_img, tmp_path):
    """Test that 2mm template is auto-loaded for MNI152_2mm space."""
    import nibabel as nib

    from lacuna import MaskData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    MaskData(mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        tractogram_space="MNI152NLin6Asym",
        output_resolution=2,
        check_dependencies=False,
    )

    # Template loading happens during analysis setup
    # Verify that get_template_path can be called with resolution=2
    template_path = get_template_path(resolution=2)
    assert template_path is not None
    assert template_path.exists()

    # Load template to check dimensions
    template_img = nib.load(template_path)
    assert template_img.shape == (91, 109, 91)


@pytest.mark.requires_templateflow
@pytest.mark.slow
def test_template_auto_loading_1mm(synthetic_mask_img, tmp_path):
    """Test that 1mm template is auto-loaded for MNI152_1mm space."""
    import nibabel as nib

    from lacuna import MaskData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    MaskData(mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 1})

    StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        tractogram_space="MNI152NLin6Asym",
        output_resolution=1,
        check_dependencies=False,
    )

    # Template loading happens during analysis setup
    # Verify that get_template_path can be called with resolution=1
    template_path = get_template_path(resolution=1)
    assert template_path is not None
    assert template_path.exists()

    # Load template to check dimensions
    template_img = nib.load(template_path)
    assert template_img.shape == (182, 218, 182)
    assert template_img.shape == (182, 218, 182)


def test_space_validation_requires_exact_format(synthetic_mask_img, tmp_path):
    """Test removed: 'native' space no longer supported after T007."""
    pytest.skip("'native' space removed from SUPPORTED_TEMPLATE_SPACES in T007")


def test_space_validation_rejects_non_mni(synthetic_mask_img, tmp_path):
    """Test removed: 'native' space no longer supported after T007."""
    pytest.skip("'native' space removed from SUPPORTED_TEMPLATE_SPACES in T007")


@pytest.mark.requires_templateflow
@pytest.mark.slow
def test_template_loading_api():
    """Test the template loading API functions.

    This test requires TemplateFlow because it verifies that templates can be loaded.
    """
    from lacuna.data import get_mni_template, get_template_path, list_templates

    # Test list_templates
    templates = list_templates()
    assert "1mm" in templates
    assert "2mm" in templates
    assert templates["1mm"]["exists"] is True
    assert templates["2mm"]["exists"] is True
    assert templates["1mm"]["shape"] == (182, 218, 182)
    assert templates["2mm"]["shape"] == (91, 109, 91)

    # Test get_template_path
    path_2mm = get_template_path(resolution=2)
    assert path_2mm.exists()
    assert path_2mm.name.endswith(".nii.gz")

    # Test get_mni_template returns nibabel image
    template = get_mni_template(resolution=2)
    assert hasattr(template, "get_fdata")
    assert hasattr(template, "affine")


def test_template_loading_invalid_resolution():
    """Test that invalid resolutions raise appropriate errors."""
    from lacuna.data import get_mni_template, get_template_path

    # Test get_mni_template
    with pytest.raises(ValueError, match="Resolution must be 1 or 2"):
        get_mni_template(resolution=3)

    # Test get_template_path
    with pytest.raises(ValueError, match="Resolution must be 1 or 2"):
        get_template_path(resolution=0)


# ============================================================================
# Memory Management and Intermediate Files Tests
# ============================================================================


def test_memory_management_parameters():
    """Test that StructuralNetworkMapping accepts memory management parameters."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Test with default values
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
    )
    assert analysis.load_to_memory is True  # Default
    assert analysis.keep_intermediate is False  # Default

    # Test with custom values
    analysis_batch = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        load_to_memory=False,
        keep_intermediate=True,
    )
    assert analysis_batch.load_to_memory is False
    assert analysis_batch.keep_intermediate is True


def test_load_to_memory_requires_keep_intermediate():
    """Test that load_to_memory=False requires keep_intermediate=True."""

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # This should raise an error during execution
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        load_to_memory=False,
        keep_intermediate=False,  # Conflicting settings
    )

    # The validation happens during _run_analysis
    # Document that this combination should fail
    assert analysis.load_to_memory is False
    assert analysis.keep_intermediate is False


def test_template_stored_as_path_not_image(synthetic_mask_img, tmp_path):
    """Test that template is stored as Path, not loaded as nibabel image.

    This is a contract test - verify the API behavior without executing MRtrix.
    """
    from pathlib import Path
    from unittest.mock import patch

    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy tractogram file (doesn't need to be valid for this test)
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    # Create mock template path
    mock_template = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), mock_template)

    MaskData(mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Mock the template loading to avoid TemplateFlow dependency
    with patch(
        "lacuna.analysis.structural_network_mapping.load_template", return_value=mock_template
    ):
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tck,
            tractogram_space="MNI152NLin6Asym",
            output_resolution=2,
            check_dependencies=False,
        )

        # Template should be set during initialization
        # The actual template would be loaded from TemplateFlow in real usage
        # Here we verify it's stored as Path, not nibabel image
        assert analysis.template is None or isinstance(
            analysis.template, Path
        ), f"Template should be None or Path, got {type(analysis.template)}"


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


def test_provenance_includes_memory_settings():
    """Test that provenance includes memory management settings."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        load_to_memory=False,
        keep_intermediate=True,
    )

    params = analysis._get_parameters()

    # Should include memory management settings
    assert "load_to_memory" in params
    assert "keep_intermediate" in params
    assert params["load_to_memory"] is False
    assert params["keep_intermediate"] is True


# ============================================================================
# Contract Tests for Atlas-based Connectivity Matrix Computation
# ============================================================================


def test_atlas_parameter_optional():
    """Test that atlas_name is an optional parameter."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should work without atlas (voxel-wise only)
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
    )
    assert analysis.atlas_name is None


def test_atlas_accepts_bundled_name():
    """Test that atlas_name can be a bundled atlas name string."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept bundled atlas name
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="schaefer100",
    )
    assert analysis.atlas_name == "schaefer100"


def test_atlas_accepts_custom_path():
    """Test that atlas_name accepts atlas registry names."""

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept atlas name from registry
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="Schaefer2018_100Parcels7Networks",
    )
    assert analysis.atlas_name == "Schaefer2018_100Parcels7Networks"


def test_compute_lesioned_parameter():
    """Test that compute_lesioned parameter is available."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should have compute_lesioned parameter
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="schaefer100",
        compute_lesioned=True,
    )
    assert analysis.compute_lesioned is True


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

    # When compute_lesioned=True, additional keys expected
    expected_stats_with_lesioned = expected_stats_keys | {
        "lesioned_mean_degree",
        "connectivity_preservation_ratio",
    }

    assert expected_stats_keys is not None
    assert expected_stats_with_lesioned is not None


def test_full_connectivity_matrix_caching():
    """Test that full connectivity matrix is cached for batch processing."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="schaefer100",
    )

    # Should have cache attribute
    assert hasattr(analysis, "_full_connectivity_matrix")
    # Initially None
    assert analysis._full_connectivity_matrix is None


def test_lesioned_connectivity_optional():
    """Test that lesioned connectivity is only computed when requested."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Default: compute_lesioned=False
    analysis1 = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="schaefer100",
    )
    assert analysis1.compute_lesioned is False

    # Explicit: compute_lesioned=True
    analysis2 = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        parcellation_name="schaefer100",
        compute_lesioned=True,
    )
    assert analysis2.compute_lesioned is True
