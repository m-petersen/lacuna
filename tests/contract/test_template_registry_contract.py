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


def test_load_template_returns_path(tmp_path, monkeypatch):
    """Test that load_template returns a Path object."""
    import sys
    from pathlib import Path

    # Mock templateflow to return test path
    def mock_tflow_get(*args, **kwargs):
        test_file = tmp_path / "test_template.nii.gz"
        test_file.touch()
        return str(test_file)

    # Create mock templateflow.api module
    mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()
    monkeypatch.setitem(sys.modules, "templateflow.api", mock_tflow_module)

    # Must import after mocking
    from lacuna.assets.templates import load_template

    result = load_template("MNI152NLin2009cAsym_res-1")

    assert isinstance(result, Path)
    assert result.exists()


def test_load_template_raises_on_invalid_name():
    """Test that load_template raises KeyError for invalid template name."""
    from lacuna.assets.templates import load_template

    with pytest.raises(KeyError, match="Template.*not found"):
        load_template("NonexistentTemplate12345")


def test_is_template_cached_returns_bool(tmp_path, monkeypatch):
    """Test that is_template_cached returns boolean."""
    import sys

    # Mock the cache check
    def mock_tflow_get(*args, **kwargs):
        test_file = tmp_path / "cached_template.nii.gz"
        test_file.touch()
        return str(test_file)

    # Create mock templateflow.api module
    mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()
    monkeypatch.setitem(sys.modules, "templateflow.api", mock_tflow_module)

    # Must import after mocking
    from lacuna.assets.templates import is_template_cached

    result = is_template_cached("MNI152NLin2009cAsym_res-1")

    assert isinstance(result, bool)


def test_template_integration_with_templateflow(tmp_path, monkeypatch):
    """Test that templates integrate correctly with TemplateFlow API."""
    import sys

    # Track TemplateFlow API calls
    calls = []

    def mock_tflow_get(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        test_file = tmp_path / "template.nii.gz"
        test_file.touch()
        return str(test_file)

    # Create mock templateflow.api module
    mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()
    monkeypatch.setitem(sys.modules, "templateflow.api", mock_tflow_module)

    # Must import after mocking
    from lacuna.assets.templates import load_template

    # Load template
    load_template("MNI152NLin2009cAsym_res-1")

    # Should have called TemplateFlow API
    assert len(calls) == 1
    call = calls[0]

    # Check that space, resolution, and modality were passed
    assert "MNI152NLin2009cAsym" in str(call)


def test_template_caching_avoids_redownload(tmp_path, monkeypatch):
    """Test that loading the same template twice doesn't re-download."""
    import sys

    call_count = [0]

    def mock_tflow_get(*args, **kwargs):
        call_count[0] += 1
        test_file = tmp_path / f"template_{call_count[0]}.nii.gz"
        test_file.touch()
        return str(test_file)

    # Create mock templateflow.api module
    mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()
    monkeypatch.setitem(sys.modules, "templateflow.api", mock_tflow_module)

    # Must import after mocking
    from lacuna.assets.templates import load_template

    # Load twice
    result1 = load_template("MNI152NLin2009cAsym_res-1")
    result2 = load_template("MNI152NLin2009cAsym_res-1")

    # Should be same path (cached)
    assert result1 == result2


def test_template_names_follow_convention():
    """Test that template names follow expected naming convention."""
    from lacuna.assets.templates import list_templates

    templates = list_templates()

    # Names should include space and resolution
    for template in templates:
        assert "MNI152" in template.name
        assert "res-" in template.name or any(str(r) in template.name for r in [1, 2])
