"""
Tests for bundled atlas functionality.

Verifies that bundled atlases can be discovered, loaded, and used in analyses.
"""

from pathlib import Path

import pytest

from ldk.data import (
    get_atlas_citation,
    get_bundled_atlas,
    get_bundled_atlas_dir,
    list_bundled_atlases,
)


def test_get_bundled_atlas_dir():
    """Test that bundled atlas directory exists."""
    atlas_dir = get_bundled_atlas_dir()

    assert isinstance(atlas_dir, Path)
    assert atlas_dir.exists()
    assert atlas_dir.is_dir()
    assert atlas_dir.name == "atlases"


def test_list_bundled_atlases_returns_list():
    """Test that list_bundled_atlases returns a list."""
    atlases = list_bundled_atlases()

    assert isinstance(atlases, list)
    # Should be sorted
    if len(atlases) > 1:
        assert atlases == sorted(atlases)


def test_list_bundled_atlases_finds_atlases():
    """Test that bundled atlases are discovered (if any exist)."""
    atlases = list_bundled_atlases()
    atlas_dir = get_bundled_atlas_dir()

    # Count .nii.gz files
    nifti_files = list(atlas_dir.glob("*.nii.gz"))

    # Should match the number of atlases found
    assert len(atlases) == len(nifti_files)


def test_get_bundled_atlas_with_valid_name():
    """Test getting a specific bundled atlas (if any exist)."""
    atlases = list_bundled_atlases()

    if not atlases:
        pytest.skip("No bundled atlases available yet")

    # Get the first available atlas
    atlas_name = atlases[0]
    img_path, labels_path = get_bundled_atlas(atlas_name)

    assert isinstance(img_path, Path)
    assert isinstance(labels_path, Path)
    assert img_path.exists()
    assert labels_path.exists()
    assert img_path.suffix == ".gz"
    assert labels_path.suffix == ".txt"


def test_get_bundled_atlas_with_invalid_name():
    """Test that get_bundled_atlas raises error for invalid name."""
    with pytest.raises(ValueError, match="not found"):
        get_bundled_atlas("nonexistent_atlas_xyz123")


def test_get_atlas_citation_returns_string():
    """Test that get_atlas_citation returns a string."""
    # Test with known atlas names
    known_atlases = [
        "aal3",
        "harvard-oxford-cortical",
        "schaefer2018-100parcels-7networks",
    ]

    for atlas_name in known_atlases:
        citation = get_atlas_citation(atlas_name)
        assert isinstance(citation, str)
        assert len(citation) > 0


def test_get_atlas_citation_with_unknown_name():
    """Test that get_atlas_citation handles unknown names gracefully."""
    citation = get_atlas_citation("unknown_atlas_xyz")

    assert isinstance(citation, str)
    assert "No citation available" in citation


def test_bundled_atlases_have_required_files():
    """Test that each bundled atlas has both image and labels files."""
    atlases = list_bundled_atlases()
    atlas_dir = get_bundled_atlas_dir()

    for atlas_name in atlases:
        # Check image file exists
        img_path = atlas_dir / f"{atlas_name}.nii.gz"
        assert img_path.exists(), f"Missing image for {atlas_name}"

        # Check labels file exists (either _labels.txt or .txt)
        labels_candidates = [
            atlas_dir / f"{atlas_name}_labels.txt",
            atlas_dir / f"{atlas_name}.txt",
        ]
        has_labels = any(p.exists() for p in labels_candidates)
        assert has_labels, f"Missing labels file for {atlas_name}"


def test_bundled_atlas_directory_has_readme():
    """Test that the atlases directory has a README."""
    atlas_dir = get_bundled_atlas_dir()
    readme_path = atlas_dir / "README.md"

    # README should exist and have content
    assert readme_path.exists()
    assert readme_path.stat().st_size > 0


def test_atlas_names_are_valid_identifiers():
    """Test that atlas names don't contain problematic characters."""
    atlases = list_bundled_atlases()

    for atlas_name in atlases:
        # Should not contain spaces or special chars that cause issues
        assert " " not in atlas_name
        assert "/" not in atlas_name
        assert "\\" not in atlas_name
        # Should be a reasonable length
        assert len(atlas_name) > 0
        assert len(atlas_name) < 200
