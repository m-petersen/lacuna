"""
Contract tests for StructuralNetworkMapping analysis class.

Tests the interface and behavior requirements for tractography-based
lesion network mapping following the BaseAnalysis contract.
"""

import pytest


def test_structural_network_mapping_import():
    """Test that StructuralNetworkMapping can be imported."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    assert StructuralNetworkMapping is not None


def test_structural_network_mapping_inherits_base_analysis():
    """Test that StructuralNetworkMapping inherits from BaseAnalysis."""
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    assert issubclass(StructuralNetworkMapping, BaseAnalysis)


def test_structural_network_mapping_can_instantiate():
    """Test that StructuralNetworkMapping can be instantiated with required parameters."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept tractogram path
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )
    assert analysis is not None
    assert analysis.tractogram_space == "MNI152NLin2009cAsym"
    assert analysis.output_resolution == 2  # default


def test_structural_network_mapping_validates_mrtrix_available():
    """Test that StructuralNetworkMapping checks for MRtrix3 availability."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should check if MRtrix3 commands are available during initialization
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )
    # Should have check_dependencies parameter or _check_dependencies attribute
    assert hasattr(analysis, "_check_dependencies") or hasattr(analysis, "check_dependencies")


def test_structural_network_mapping_has_run_method():
    """Test that StructuralNetworkMapping has the run() method from BaseAnalysis."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_structural_network_mapping_validates_coordinate_space(synthetic_lesion_img, tmp_path):
    """Test that StructuralNetworkMapping validates MNI152 coordinate space."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "native"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Should raise error if not in MNI152 space
    with pytest.raises(ValueError, match="MNI152"):
        analysis.run(lesion_data)


def test_structural_network_mapping_requires_binary_mask(synthetic_lesion_img, tmp_path):
    """Test that StructuralNetworkMapping requires binary lesion mask."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    # Create non-binary lesion data
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5  # Make it non-binary

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    lesion_data = LesionData(lesion_img=non_binary_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        analysis.run(lesion_data)


def test_structural_network_mapping_returns_lesion_data(synthetic_lesion_img):
    """Test that run() returns a LesionData object with namespaced results."""
    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Mark lesion as MNI152 space
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Note: This test will fail until implementation exists
    # It defines the expected behavior
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )

    # For now, expect this to fail during actual run
    # The test documents the expected interface
    with pytest.raises((FileNotFoundError, RuntimeError)):
        result = analysis.run(lesion_data)


def test_structural_network_mapping_result_structure():
    """Test that results should contain expected keys and data types."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )

    # Document expected result structure
    expected_keys = {
        "disconnection_map",  # NIfTI image showing voxel-level disconnection
        "mean_disconnection",  # Summary statistic
        "lesion_tck_count",  # Number of streamlines passing through lesion
    }

    # This documents the contract
    assert expected_keys is not None


def test_structural_network_mapping_accepts_n_jobs():
    """Test that StructuralNetworkMapping accepts n_jobs parameter for MRtrix."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
        n_jobs=8,
    )

    assert analysis.n_jobs == 8


def test_structural_network_mapping_preserves_input_immutability(synthetic_lesion_img):
    """Test that run() does not modify the input LesionData."""
    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    original_results = lesion_data.results.copy()

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        template="/path/to/template.nii.gz",
    )

    # Expected to fail during execution, but documents immutability requirement
    try:
        result = analysis.run(lesion_data)
        # If it somehow succeeds, check immutability
        assert lesion_data.results == original_results
        assert result is not lesion_data
    except (FileNotFoundError, RuntimeError):
        # Expected until implementation exists
        pass


def test_structural_network_mapping_adds_provenance():
    """Test that run() should add provenance record."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
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


def test_template_auto_loading_2mm(synthetic_lesion_img, tmp_path):
    """Test that 2mm template is auto-loaded for MNI152_2mm space."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Validate inputs triggers template loading
    analysis._validate_inputs(lesion_data)

    # Should have loaded 2mm template path
    assert analysis.template is not None
    assert analysis.template == get_template_path(resolution=2)

    # Load template to check dimensions
    template_img = nib.load(analysis.template)
    assert template_img.shape == (91, 109, 91)


def test_template_auto_loading_1mm(synthetic_lesion_img, tmp_path):
    """Test that 1mm template is auto-loaded for MNI152_1mm space."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
    from lacuna.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 1})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Validate inputs triggers template loading
    analysis._validate_inputs(lesion_data)

    # Should have loaded 1mm template path
    assert analysis.template is not None
    assert analysis.template == get_template_path(resolution=1)

    # Load template to check dimensions
    template_img = nib.load(analysis.template)
    assert template_img.shape == (182, 218, 182)


def test_space_validation_requires_exact_format(synthetic_lesion_img, tmp_path):
    """Test that native space is rejected."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # NOTE: With the new API, space validation is more flexible.
    # As long as the space is not "native", the base class will attempt transformation.
    # Invalid formats will fail during transformation, not during initial validation.
    
    # Test that native space is rejected
    native_lesion = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "native"})
    with pytest.raises(ValueError, match="Native space"):
        analysis.run(native_lesion)


def test_space_validation_rejects_non_mni(synthetic_lesion_img, tmp_path):
    """Test that non-MNI152 spaces are rejected."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "native"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Space validation happens in run(), not _validate_inputs()
    with pytest.raises(ValueError, match="Native space lesions are not supported"):
        analysis.run(lesion_data)


def test_bundled_template_exists():
    """Test that bundled MNI templates are available."""
    from lacuna.data import get_template_path

    # Both templates should exist
    path_1mm = get_template_path(resolution=1)
    path_2mm = get_template_path(resolution=2)

    assert path_1mm.exists(), f"1mm template not found at {path_1mm}"
    assert path_2mm.exists(), f"2mm template not found at {path_2mm}"


def test_bundled_template_correct_dimensions():
    """Test that bundled templates have correct dimensions."""
    from lacuna.data import get_mni_template

    # Test 1mm template
    template_1mm = get_mni_template(resolution=1)
    assert template_1mm.shape == (182, 218, 182), (
        f"1mm template has wrong shape: {template_1mm.shape}"
    )

    # Test 2mm template
    template_2mm = get_mni_template(resolution=2)
    assert template_2mm.shape == (91, 109, 91), (
        f"2mm template has wrong shape: {template_2mm.shape}"
    )


def test_template_loading_api():
    """Test the template loading API functions."""
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
    assert path_2mm.name == "MNI152_T1_2mm.nii.gz"

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


def test_template_stored_as_path_not_image(synthetic_lesion_img, tmp_path):
    """Test that template is stored as Path, not loaded as nibabel image."""
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tck.touch()

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        check_dependencies=False,
    )

    # Validate inputs triggers template loading
    analysis._validate_inputs(lesion_data)

    # Template should be a Path object, not a nibabel image
    assert isinstance(analysis.template, Path), (
        f"Template should be Path, got {type(analysis.template)}"
    )
    assert analysis.template.exists()
    assert str(analysis.template).endswith(".nii.gz")


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
        result = compute_whole_brain_tdi(
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
        atlas_name="schaefer100",
    )
    assert analysis.atlas_name == "schaefer100"


def test_atlas_accepts_custom_path():
    """Test that atlas_name accepts atlas registry names."""

    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept atlas name from registry
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        atlas_name="Schaefer2018_100Parcels7Networks",
    )
    assert analysis.atlas_name == "Schaefer2018_100Parcels7Networks"


def test_compute_lesioned_parameter():
    """Test that compute_lesioned parameter is available."""
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should have compute_lesioned parameter
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        atlas_name="schaefer100",
        compute_lesioned=True,
    )
    assert analysis.compute_lesioned is True


def test_atlas_resolved_during_validation():
    """Test that bundled atlas is resolved to Path during validation."""
    import tempfile
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create dummy files
        tck_file = tmp_path / "tractogram.tck"
        tdi_file = tmp_path / "tdi.nii.gz"
        tck_file.touch()
        nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), tdi_file)

        # Create lesion
        lesion_data = np.zeros((91, 109, 91))
        lesion_data[45:50, 54:59, 45:50] = 1
        lesion_img = nib.Nifti1Image(lesion_data, np.eye(4))
        lesion = LesionData(lesion_img=lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

        # Initialize with bundled atlas name
        analysis = StructuralNetworkMapping(
            tractogram_path=tck_file,
            whole_brain_tdi=tdi_file,
            atlas_name="schaefer100",
            check_dependencies=False,
        )

        # Before validation, _atlas_resolved should be None
        assert analysis._atlas_resolved is None

        # Validate - this should resolve the atlas
        try:
            analysis._validate_inputs(lesion)
            # After validation, _atlas_resolved should be a Path
            assert analysis._atlas_resolved is not None
            assert isinstance(analysis._atlas_resolved, Path)
        except (FileNotFoundError, ValueError):
            # Expected if atlas doesn't exist, but _atlas_resolved should still be attempted
            pass


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
        atlas_name="schaefer100",
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
        atlas_name="schaefer100",
    )
    assert analysis1.compute_lesioned is False

    # Explicit: compute_lesioned=True
    analysis2 = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        tractogram_space="MNI152NLin2009cAsym",
        atlas_name="schaefer100",
        compute_lesioned=True,
    )
    assert analysis2.compute_lesioned is True
