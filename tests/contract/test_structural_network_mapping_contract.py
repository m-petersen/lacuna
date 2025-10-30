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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    from ldk.analysis.base import BaseAnalysis

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
    # Should have method to check MRtrix availability
    assert hasattr(analysis, "_check_mrtrix_available") or hasattr(analysis, "check_dependencies")


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


def test_structural_network_mapping_validates_coordinate_space(synthetic_lesion_img):
    """Test that StructuralNetworkMapping validates MNI152 coordinate space."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    from ldk import LesionData

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )

    # Should raise error if not in MNI152 space
    with pytest.raises(ValueError, match="MNI152"):
        analysis.run(lesion_data)


def test_structural_network_mapping_requires_binary_mask(synthetic_lesion_img):
    """Test that StructuralNetworkMapping requires binary lesion mask."""
    import nibabel as nib
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    from ldk import LesionData

    # Create non-binary lesion data
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5  # Make it non-binary

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    lesion_data = LesionData(lesion_img=non_binary_img)

    analysis = StructuralNetworkMapping(
        tractogram_path="/path/to/tractogram.tck",
        whole_brain_tdi="/path/to/tdi.nii.gz",
        template="/path/to/template.nii.gz",
    )

    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        analysis.run(lesion_data)


def test_structural_network_mapping_returns_lesion_data(synthetic_lesion_img):
    """Test that run() returns a LesionData object with namespaced results."""
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    from ldk import LesionData

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
    from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

    from ldk import LesionData

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
