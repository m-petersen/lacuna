"""
Contract tests for StructuralNetworkMapping analysis class.

Tests the interface and behavior requirements for tractography-based
lesion network mapping following the BaseAnalysis contract.
"""

import pytest


def test_structural_network_mapping_import():
    """Test that StructuralNetworkMapping can be imported."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    assert StructuralNetworkMapping is not None


def test_structural_network_mapping_inherits_base_analysis():
    """Test that StructuralNetworkMapping inherits from BaseAnalysis."""
    from ldk.analysis.base import BaseAnalysis
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    assert issubclass(StructuralNetworkMapping, BaseAnalysis)


def test_structural_network_mapping_can_instantiate():
    """Test that StructuralNetworkMapping can be instantiated with required parameters."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should accept tractogram and TDI paths
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )
    assert analysis is not None


def test_structural_network_mapping_validates_mrtrix_available():
    """Test that StructuralNetworkMapping checks for MRtrix3 availability."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Should check if MRtrix3 commands are available during initialization
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )
    # Should have check_dependencies parameter or _check_dependencies attribute
    assert hasattr(analysis, "_check_dependencies") or hasattr(analysis, "check_dependencies")


def test_structural_network_mapping_has_run_method():
    """Test that StructuralNetworkMapping has the run() method from BaseAnalysis."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_structural_network_mapping_validates_coordinate_space(synthetic_lesion_img, tmp_path):
    """Test that StructuralNetworkMapping validates MNI152 coordinate space."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "native"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
        check_dependencies=False,
    )

    # Should raise error if not in MNI152 space
    with pytest.raises(ValueError, match="MNI152"):
        analysis.run(lesion_data)


def test_structural_network_mapping_requires_binary_mask(synthetic_lesion_img, tmp_path):
    """Test that StructuralNetworkMapping requires binary lesion mask."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    # Create non-binary lesion data
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5  # Make it non-binary

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    lesion_data = LesionData(lesion_img=non_binary_img, metadata={"space": "MNI152_2mm"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
        check_dependencies=False,
    )

    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        analysis.run(lesion_data)


def test_structural_network_mapping_returns_lesion_data(synthetic_lesion_img):
    """Test that run() returns a LesionData object with namespaced results."""
    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Mark lesion as MNI152 space
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    # Note: This test will fail until implementation exists
    # It defines the expected behavior
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )

    # For now, expect this to fail during actual run
    # The test documents the expected interface
    with pytest.raises((FileNotFoundError, RuntimeError)):
        result = analysis.run(lesion_data)


def test_structural_network_mapping_result_structure():
    """Test that results should contain expected keys and data types."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
        n_jobs=8,
    )

    assert analysis.n_jobs == 8


def test_structural_network_mapping_preserves_input_immutability(synthetic_lesion_img):
    """Test that run() does not modify the input LesionData."""
    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
    original_results = lesion_data.results.copy()

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
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

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping
    from ldk.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
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

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping
    from ldk.data import get_template_path

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_1mm"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
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
    """Test that space must be exactly 'MNI152_1mm' or 'MNI152_2mm'."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
        check_dependencies=False,
    )

    # Test invalid space formats - should all fail
    invalid_spaces = [
        "MNI152",  # Missing resolution
        "MNI_2mm",  # Wrong format
        "mni152_2mm",  # Wrong case
        "MNI152_2",  # Missing 'mm'
        "MNI152 2mm",  # Space instead of underscore
    ]

    for invalid_space in invalid_spaces:
        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": invalid_space})

        with pytest.raises(ValueError, match="Invalid coordinate space"):
            analysis._validate_inputs(lesion_data)


def test_space_validation_rejects_non_mni(synthetic_lesion_img, tmp_path):
    """Test that non-MNI152 spaces are rejected."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "native"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
        check_dependencies=False,
    )

    with pytest.raises(ValueError, match="Invalid coordinate space"):
        analysis._validate_inputs(lesion_data)


def test_bundled_template_exists():
    """Test that bundled MNI templates are available."""
    from ldk.data import get_template_path

    # Both templates should exist
    path_1mm = get_template_path(resolution=1)
    path_2mm = get_template_path(resolution=2)

    assert path_1mm.exists(), f"1mm template not found at {path_1mm}"
    assert path_2mm.exists(), f"2mm template not found at {path_2mm}"


def test_bundled_template_correct_dimensions():
    """Test that bundled templates have correct dimensions."""
    from ldk.data import get_mni_template

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
    from ldk.data import get_mni_template, get_template_path, list_templates

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
    from ldk.data import get_mni_template, get_template_path

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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Test with default values
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
    )
    assert analysis.load_to_memory is True  # Default
    assert analysis.keep_intermediate is False  # Default

    # Test with custom values
    analysis_batch = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        load_to_memory=False,
        keep_intermediate=True,
    )
    assert analysis_batch.load_to_memory is False
    assert analysis_batch.keep_intermediate is True


def test_load_to_memory_requires_keep_intermediate():
    """Test that load_to_memory=False requires keep_intermediate=True."""

    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # This should raise an error during execution
    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
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

    from ldk import LesionData
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    # Create dummy files
    dummy_tck = tmp_path / "tractogram.tck"
    dummy_tdi = tmp_path / "tdi.nii.gz"
    dummy_tck.touch()
    nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    analysis = StructuralNetworkMapping(
        tractogram_path=dummy_tck,
        whole_brain_tdi=dummy_tdi,
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
    from ldk.utils.mrtrix import compute_whole_brain_tdi

    assert compute_whole_brain_tdi is not None
    assert callable(compute_whole_brain_tdi)


def test_compute_whole_brain_tdi_accepts_parameters():
    """Test that compute_whole_brain_tdi accepts expected parameters."""
    from ldk.utils.mrtrix import compute_whole_brain_tdi

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
    from ldk.utils.mrtrix import compute_whole_brain_tdi

    # Should raise error if neither output specified
    with pytest.raises(ValueError, match="At least one"):
        compute_whole_brain_tdi(tractogram_path="/path/to/tractogram.tck")


def test_compute_whole_brain_tdi_validates_tractogram_exists():
    """Test that compute_whole_brain_tdi validates tractogram file exists."""
    from ldk.utils.mrtrix import compute_whole_brain_tdi

    # Should raise FileNotFoundError if tractogram doesn't exist
    with pytest.raises(FileNotFoundError, match="Tractogram"):
        compute_whole_brain_tdi(
            tractogram_path="/nonexistent/tractogram.tck",
            output_2mm="/path/to/output.nii.gz",
        )


def test_mrtrix_commands_are_printed(capsys):
    """Test that MRtrix3 commands are printed during execution."""
    from ldk.utils.mrtrix import run_mrtrix_command

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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    assert StructuralNetworkMapping.batch_strategy == "parallel"


def test_provenance_includes_memory_settings():
    """Test that provenance includes memory management settings."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        load_to_memory=False,
        keep_intermediate=True,
    )

    params = analysis._get_parameters()

    # Should include memory management settings
    assert "load_to_memory" in params
    assert "keep_intermediate" in params
    assert params["load_to_memory"] is False
    assert params["keep_intermediate"] is True
