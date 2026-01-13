"""
Test for template resolution bug fix in StructuralNetworkMapping.

This test ensures that the template is resolved BEFORE TDI computation,
preventing the TypeError: template must be str, Path, or nibabel.Nifti1Image, got <class 'NoneType'>
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
from lacuna.assets.connectomes import (
    register_structural_connectome,
    unregister_structural_connectome,
)
from lacuna.core.subject_data import SubjectData


def test_template_resolved_before_tdi_computation(tmp_path):
    """Test that template is resolved before TDI computation is attempted.

    This is a regression test for the bug where template=None caused
    compute_tdi_map to fail with TypeError.
    """
    # Create dummy tractogram file
    tractogram_path = tmp_path / "tractogram.tck"
    tractogram_path.write_text("dummy tractogram")

    registered = False
    try:
        # Register connectome (TDI will be computed on-the-fly)
        register_structural_connectome(
            name="test_template_fix",
            space="MNI152NLin2009cAsym",
            tractogram_path=tractogram_path,
        )
        registered = True

        # Create dummy lesion
        mask_data = np.zeros((10, 10, 10), dtype=np.float32)
        mask_data[4:6, 4:6, 4:6] = 1.0
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(mask_data, affine)
        lesion = SubjectData(
            mask_img=mask_img,
            metadata={
                "subject_id": "test",
                "space": "MNI152NLin2009cAsym",
                "resolution": 2,
            },
        )

        # Mock dependencies
        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            with patch(
                "lacuna.analysis.structural_network_mapping.load_template"
            ) as mock_load_template:
                # Mock template loading
                template_path = tmp_path / "template.nii.gz"
                template_data = np.zeros((91, 109, 91), dtype=np.float32)
                template_img = nib.Nifti1Image(template_data, np.eye(4))
                nib.save(template_img, template_path)
                mock_load_template.return_value = template_path

                with patch(
                    "lacuna.analysis.structural_network_mapping.compute_tdi_map"
                ) as mock_compute_tdi:
                    # Mock TDI computation - this is where the bug would have occurred
                    # TDI is now computed on-the-fly, so we just mock the function
                    def save_tdi(*args, **kwargs):
                        output_path = kwargs.get("output_path")
                        tdi_img = nib.Nifti1Image(template_data, np.eye(4))
                        nib.save(tdi_img, output_path)
                        return output_path

                    mock_compute_tdi.side_effect = save_tdi

                    # Initialize analysis using registered connectome
                    analysis = StructuralNetworkMapping(
                        connectome_name="test_template_fix",
                        cache_tdi=False,  # Don't use cache for this test
                        check_dependencies=False,
                    )

                    # Call _validate_inputs directly to test the fix
                    # This is where template resolution happens before TDI computation
                    analysis._validate_inputs(lesion)

                    # Verify template was loaded BEFORE compute_tdi_map was called
                    assert mock_load_template.called, "Template should be loaded"
                    assert mock_compute_tdi.called, "TDI computation should be called"

                    # Verify compute_tdi_map was called with a valid template (not None)
                    call_kwargs = mock_compute_tdi.call_args[1]
                    template_arg = call_kwargs.get("template")
                    assert (
                        template_arg is not None
                    ), "Template should not be None when compute_tdi_map is called"
                    assert isinstance(
                        template_arg, Path
                    ), f"Template should be Path, got {type(template_arg)}"

                    # Most importantly: verify the analysis.template attribute was set
                    assert (
                        analysis.template is not None
                    ), "analysis.template should be set after validation"
                    assert (
                        analysis.template == template_path
                    ), "analysis.template should match loaded template"

    except TypeError as e:
        if "template must be" in str(e):
            pytest.fail(f"Template resolution bug not fixed: {e}")
        else:
            raise
    finally:
        if registered:
            unregister_structural_connectome("test_template_fix")


def test_cache_directory_uses_unified_location():
    """Test that TDI cache uses the unified cache directory system."""
    from lacuna.utils.cache import get_tdi_cache_dir

    # Create dummy tractogram file
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as tmp:
        tractogram_path = Path(tmp.name)
        tmp.write(b"dummy")

    registered = False
    try:
        register_structural_connectome(
            name="test_cache_location",
            space="MNI152NLin2009cAsym",
            tractogram_path=tractogram_path,
        )
        registered = True

        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            analysis = StructuralNetworkMapping(
                connectome_name="test_cache_location",
                cache_tdi=True,
                check_dependencies=False,
            )

            cache_path = analysis._get_tdi_cache_path()
            expected_cache_dir = get_tdi_cache_dir()

            # Verify cache path is in unified cache directory
            assert (
                cache_path.parent == expected_cache_dir
            ), f"Cache should use unified directory {expected_cache_dir}, got {cache_path.parent}"
    finally:
        if registered:
            unregister_structural_connectome("test_cache_location")
        tractogram_path.unlink(missing_ok=True)


def test_cache_directory_respects_env_variable():
    """Test that cache directory can be configured via LACUNA_CACHE_DIR."""
    from lacuna.utils.cache import get_cache_dir

    custom_cache = "/tmp/my_custom_lacuna_cache"

    # Set environment variable
    old_env = os.environ.get("LACUNA_CACHE_DIR")
    os.environ["LACUNA_CACHE_DIR"] = custom_cache

    try:
        cache_dir = get_cache_dir()
        assert (
            str(cache_dir) == custom_cache
        ), f"Cache directory should respect LACUNA_CACHE_DIR, got {cache_dir}"
    finally:
        # Restore original environment
        if old_env is not None:
            os.environ["LACUNA_CACHE_DIR"] = old_env
        else:
            os.environ.pop("LACUNA_CACHE_DIR", None)
