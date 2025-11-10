"""
Contract tests for FunctionalNetworkMapping analysis class.

Tests the interface and behavior requirements for functional connectivity-based
lesion network mapping following the BaseAnalysis contract.
"""

import pytest


def test_functional_network_mapping_import():
    """Test that FunctionalNetworkMapping can be imported."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    assert FunctionalNetworkMapping is not None


def test_functional_network_mapping_inherits_base_analysis():
    """Test that FunctionalNetworkMapping inherits from BaseAnalysis."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna.analysis.base import BaseAnalysis

    assert issubclass(FunctionalNetworkMapping, BaseAnalysis)


def test_functional_network_mapping_can_instantiate():
    """Test that FunctionalNetworkMapping can be instantiated with required parameters."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    # Should accept connectome path
    analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5")
    assert analysis is not None


def test_functional_network_mapping_has_method_parameter():
    """Test that FunctionalNetworkMapping accepts method parameter (boes/pini)."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    # Test BOES method
    analysis_boes = FunctionalNetworkMapping(
        connectome_path="/path/to/connectome.h5", method="boes"
    )
    assert analysis_boes.method == "boes"

    # Test PINI method
    analysis_pini = FunctionalNetworkMapping(
        connectome_path="/path/to/connectome.h5", method="pini"
    )
    assert analysis_pini.method == "pini"


def test_functional_network_mapping_validates_method():
    """Test that invalid method raises ValueError."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    with pytest.raises(ValueError, match="method must be 'boes' or 'pini'"):
        FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5", method="invalid")


def test_functional_network_mapping_has_run_method():
    """Test that FunctionalNetworkMapping has the run() method from BaseAnalysis."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5")
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_functional_network_mapping_validates_coordinate_space(synthetic_lesion_img):
    """Test that FunctionalNetworkMapping validates MNI152 coordinate space."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create lesion data with non-MNI space
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5")

    # Should raise error if not in MNI152 space
    with pytest.raises(ValueError, match="MNI152"):
        analysis.run(lesion_data)


def test_functional_network_mapping_requires_binary_mask(synthetic_lesion_img):
    """Test that FunctionalNetworkMapping requires binary lesion mask."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create non-binary lesion data
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5  # Make it non-binary
    import nibabel as nib

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    lesion_data = LesionData(lesion_img=non_binary_img, metadata={"space": "MNI152_2mm"})

    analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5")

    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        analysis.run(lesion_data)


def test_functional_network_mapping_returns_lesion_data(synthetic_lesion_img, tmp_path):
    """Test that run() returns a LesionData object with namespaced results."""
    import h5py
    import numpy as np
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create minimal mock connectome file
    connectome_path = tmp_path / "mock_connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        # Mock timeseries data: (subjects, timepoints, voxels)
        f.create_dataset("timeseries", data=np.random.randn(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.array([[30, 30, 30], [31, 31, 31]]))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = (64, 64, 64)

    # Mark lesion as MNI152 space (add metadata)
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    analysis = FunctionalNetworkMapping(connectome_path=str(connectome_path))
    result = analysis.run(lesion_data)

    # Should return LesionData
    assert isinstance(result, LesionData)

    # Should have namespaced results
    assert "FunctionalNetworkMapping" in result.results


def test_functional_network_mapping_result_structure(synthetic_lesion_img, tmp_path):
    """Test that results contain expected keys and data types."""
    import h5py
    import numpy as np
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create minimal mock connectome
    connectome_path = tmp_path / "mock_connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.array([[30, 30, 30], [31, 31, 31]]))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = (64, 64, 64)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

    analysis = FunctionalNetworkMapping(connectome_path=str(connectome_path))
    result = analysis.run(lesion_data)

    results_dict = result.results["FunctionalNetworkMapping"]

    # Should contain correlation map (r-values)
    assert "correlation_map" in results_dict or "network_map" in results_dict

    # Should contain z-transformed map
    assert "z_map" in results_dict

    # Should contain summary statistics
    assert "mean_correlation" in results_dict or "summary_statistics" in results_dict


def test_functional_network_mapping_accepts_pini_percentile():
    """Test that PINI method accepts percentile parameter."""
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    analysis = FunctionalNetworkMapping(
        connectome_path="/path/to/connectome.h5", method="pini", pini_percentile=20
    )

    assert analysis.pini_percentile == 20


def test_functional_network_mapping_preserves_input_immutability(synthetic_lesion_img, tmp_path):
    """Test that run() does not modify the input LesionData."""
    import h5py
    import numpy as np
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create mock connectome
    connectome_path = tmp_path / "mock_connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.array([[30, 30, 30]]))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = (64, 64, 64)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
    original_results = lesion_data.results.copy()

    analysis = FunctionalNetworkMapping(connectome_path=str(connectome_path))
    result = analysis.run(lesion_data)

    # Input should not be modified
    assert lesion_data.results == original_results
    assert "FunctionalNetworkMapping" not in lesion_data.results

    # Result should be different object
    assert result is not lesion_data


def test_functional_network_mapping_adds_provenance(synthetic_lesion_img, tmp_path):
    """Test that run() adds provenance record."""
    import h5py
    import numpy as np
    from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

    from lacuna import LesionData

    # Create mock connectome
    connectome_path = tmp_path / "mock_connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.array([[30, 30, 30]]))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = (64, 64, 64)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
    original_prov_len = len(lesion_data.provenance)

    analysis = FunctionalNetworkMapping(connectome_path=str(connectome_path))
    result = analysis.run(lesion_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "FunctionalNetworkMapping" in latest_prov["function"]
