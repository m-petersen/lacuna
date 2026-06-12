"""Contract tests for template registry.

Tests the interface and behavior requirements for template asset management.
"""

import pytest


def test_template_registry_can_import():
    """Test that template registry can be imported."""
    from lacuna.assets.templates import list_templates, load_template

    assert list_templates is not None
    assert load_template is not None


def test_list_templates_returns_list():
    """Test that list_templates returns a list of metadata."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()

    assert isinstance(templates, list)
    assert len(templates) > 0


def test_list_templates_includes_expected_spaces():
    """Test that list_templates includes MNI152 templates."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()

    # Should have both NLin6Asym and NLin2009cAsym templates
    spaces = {t.space for t in templates}
    assert "MNI152NLin6Asym" in spaces
    assert "MNI152NLin2009cAsym" in spaces


def test_list_templates_includes_common_resolutions():
    """Test that list_templates includes 1mm and 2mm resolutions."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()

    resolutions = {t.resolution for t in templates}
    assert 1.0 in resolutions
    assert 2.0 in resolutions


def test_list_templates_filter_by_space():
    """Test that list_templates can filter by space."""
    from lacuna.assets.templates import list_templates

    # Filter by NLin6Asym
    nlin6_templates = list_templates(space="MNI152NLin6Asym")
    assert all(t.space == "MNI152NLin6Asym" for t in nlin6_templates)

    # Filter by NLin2009cAsym
    nlin2009c_templates = list_templates(space="MNI152NLin2009cAsym")
    assert all(t.space == "MNI152NLin2009cAsym" for t in nlin2009c_templates)


def test_list_templates_filter_by_resolution():
    """Test that list_templates can filter by resolution."""
    from lacuna.assets.templates import list_templates

    # Filter by 1mm
    templates_1mm = list_templates(resolution=1.0)
    assert all(t.resolution == 1.0 for t in templates_1mm)

    # Filter by 2mm
    templates_2mm = list_templates(resolution=2.0)
    assert all(t.resolution == 2.0 for t in templates_2mm)


def test_list_templates_filter_by_modality():
    """Test that list_templates can filter by modality."""
    from lacuna.assets.templates import list_templates

    # Filter by T1w
    t1w_templates = list_templates(modality="T1w")
    assert all(t.modality == "T1w" for t in t1w_templates)


def test_list_templates_combined_filters():
    """Test that list_templates can apply multiple filters."""
    from lacuna.assets.templates import list_templates

    # Get NLin2009cAsym at 1mm
    templates = list_templates(space="MNI152NLin2009cAsym", resolution=1.0)

    assert len(templates) > 0
    assert all(t.space == "MNI152NLin2009cAsym" and t.resolution == 1.0 for t in templates)


def test_template_metadata_has_required_fields():
    """Test that TemplateMetadata has required fields."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()
    template = templates[0]

    # Check required fields
    assert hasattr(template, "name")
    assert hasattr(template, "space")
    assert hasattr(template, "resolution")
    assert hasattr(template, "description")
    assert hasattr(template, "modality")

    assert isinstance(template.name, str)
    assert isinstance(template.space, str)
    assert isinstance(template.resolution, (int, float))
    assert isinstance(template.description, str)
    assert isinstance(template.modality, str)


def test_load_template_returns_path():
    """load_template returns a Path to the bundled grid-only reference (no network)."""
    from pathlib import Path

    from lacuna.assets.templates import load_template

    result = load_template("MNI152NLin2009cAsym_res-1")
    assert isinstance(result, Path)
    assert result.exists()


def test_load_template_raises_on_invalid_name():
    """Test that load_template raises KeyError for invalid template name."""
    from lacuna.assets.templates import load_template

    with pytest.raises(KeyError, match="Template.*not found"):
        load_template("NonexistentTemplate12345")


def test_is_template_cached_returns_bool():
    """is_template_cached returns a boolean."""
    from lacuna.assets.templates.loader import is_template_cached

    assert isinstance(is_template_cached("MNI152NLin2009cAsym_res-1"), bool)


def test_template_returns_bundled_grid_reference():
    """load_template returns a bundled grid-only reference, not a TemplateFlow download.

    Lacuna uses templates only as a resampling grid, so load_template must return
    a zero-filled NIfTI bundled with the package, on the correct canonical grid,
    without requiring TemplateFlow.
    """
    import nibabel as nib
    import numpy as np

    from lacuna.assets.templates.loader import load_template
    from lacuna.core.spaces import REFERENCE_AFFINES

    path = load_template("MNI152NLin2009cAsym_res-1")

    # Comes from the bundled package data, not a network cache
    assert path.exists()
    assert path.parent.name == "templates"
    assert "data" in path.parts

    img = nib.load(path)
    # Grid-only: zero intensity, but correct canonical geometry
    assert float(np.asarray(img.dataobj).max()) == 0.0
    np.testing.assert_allclose(img.affine, REFERENCE_AFFINES[("MNI152NLin2009cAsym", 1)], atol=1e-3)


def test_template_space_equivalence_maps_2009b_to_2009c():
    """A 2009b request resolves to the canonical 2009c bundled grid."""
    from lacuna.assets.templates.loader import load_template

    path = load_template("MNI152NLin2009bAsym_res-2")
    assert path.name == "MNI152NLin2009cAsym_res-2.nii.gz"


def test_template_load_is_deterministic():
    """Loading the same template twice returns the same bundled path."""
    from lacuna.assets.templates import load_template

    result1 = load_template("MNI152NLin2009cAsym_res-1")
    result2 = load_template("MNI152NLin2009cAsym_res-1")
    assert result1 == result2
    assert result1.exists()


def test_template_names_follow_convention():
    """Test that template names follow expected naming convention."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()

    # Names should include space and resolution
    for template in templates:
        assert "MNI152" in template.name
        assert "res-" in template.name or any(str(r) in template.name for r in [1, 2])
