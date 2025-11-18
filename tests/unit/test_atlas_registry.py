"""Unit tests for atlas registry functionality.

Tests atlas metadata, registry operations, and atlas listing/filtering.
"""

import pytest

from lacuna.assets.atlases import (
    ATLAS_REGISTRY,
    Atlas,
    AtlasMetadata,
    list_atlases,
    register_atlas,
)


def test_atlas_metadata_creation():
    """Test creating AtlasMetadata with required fields."""
    metadata = AtlasMetadata(
        name='TestAtlas',
        full_name='Test Atlas Full Name',
        space='MNI152NLin6Asym',
        resolution=2.0,
        atlas_type='deterministic',
        n_regions=50,
        description='Test atlas for unit testing'
    )
    
    assert metadata.name == 'TestAtlas'
    assert metadata.full_name == 'Test Atlas Full Name'
    assert metadata.space == 'MNI152NLin6Asym'
    assert metadata.resolution == 2.0
    assert metadata.atlas_type == 'deterministic'
    assert metadata.n_regions == 50
    assert metadata.bundled is False  # Default
    assert metadata.filename is None  # Default


def test_atlas_metadata_bundled():
    """Test creating bundled atlas metadata."""
    metadata = AtlasMetadata(
        name='BundledAtlas',
        full_name='Bundled Test Atlas',
        space='MNI152NLin6Asym',
        resolution=1.0,
        atlas_type='deterministic',
        n_regions=100,
        description='Bundled atlas',
        bundled=True,
        filename='test_atlas_file'
    )
    
    assert metadata.bundled is True
    assert metadata.filename == 'test_atlas_file'


def test_atlas_registry_has_bundled_atlases():
    """Test that registry contains expected bundled atlases."""
    # Check for Schaefer atlases
    assert 'Schaefer100' in ATLAS_REGISTRY
    assert 'Schaefer200' in ATLAS_REGISTRY
    assert 'Schaefer400' in ATLAS_REGISTRY
    assert 'Schaefer1000' in ATLAS_REGISTRY
    
    # Check for Tian atlases
    assert 'TianS1' in ATLAS_REGISTRY
    assert 'TianS2' in ATLAS_REGISTRY
    assert 'TianS3' in ATLAS_REGISTRY
    
    # Check for HCP atlas
    assert 'HCP1065' in ATLAS_REGISTRY


def test_atlas_registry_metadata_validity():
    """Test that all registered atlases have valid metadata."""
    for name, metadata in ATLAS_REGISTRY.items():
        assert isinstance(metadata, AtlasMetadata)
        assert metadata.name == name
        assert len(metadata.description) > 0
        assert len(metadata.space) > 0
        assert metadata.resolution > 0
        assert metadata.n_regions is None or metadata.n_regions > 0
        assert isinstance(metadata.is_4d, bool)


def test_list_atlases_all():
    """Test listing all atlases without filters."""
    atlases = list_atlases()
    
    assert len(atlases) >= 8  # At least 8 bundled atlases
    assert all(isinstance(a, AtlasMetadata) for a in atlases)
    
    # Check they're sorted by name
    names = [a.name for a in atlases]
    assert names == sorted(names)


def test_list_atlases_filter_by_space():
    """Test filtering atlases by coordinate space."""
    nlin6_atlases = list_atlases(space='MNI152NLin6Asym')
    
    assert len(nlin6_atlases) > 0
    assert all(a.space == 'MNI152NLin6Asym' for a in nlin6_atlases)
    
    # Should include Schaefer atlases
    names = [a.name for a in nlin6_atlases]
    assert 'Schaefer100' in names


def test_list_atlases_filter_by_type():
    """Test filtering atlases by type."""
    deterministic = list_atlases(atlas_type='deterministic')
    probabilistic = list_atlases(atlas_type='probabilistic')
    
    assert len(deterministic) > 0
    assert len(probabilistic) > 0
    
    assert all(a.atlas_type == 'deterministic' for a in deterministic)
    assert all(a.atlas_type == 'probabilistic' for a in probabilistic)
    
    # HCP1065 should be probabilistic
    prob_names = [a.name for a in probabilistic]
    assert 'HCP1065' in prob_names


def test_list_atlases_filter_by_region_count():
    """Test filtering atlases by region count range."""
    # Small atlases (< 100 regions)
    small = list_atlases(max_regions=99)
    assert all(a.n_regions <= 99 for a in small)
    
    # Medium atlases (100-500 regions)
    medium = list_atlases(min_regions=100, max_regions=500)
    assert all(100 <= a.n_regions <= 500 for a in medium)
    
    # Large atlases (> 500 regions)
    large = list_atlases(min_regions=501)
    assert all(a.n_regions >= 501 for a in large)


def test_list_atlases_combined_filters():
    """Test combining multiple filters."""
    # Deterministic atlases in NLin6 space with 100-500 regions
    filtered = list_atlases(
        space='MNI152NLin6Asym',
        atlas_type='deterministic',
        min_regions=100,
        max_regions=500
    )
    
    assert all(
        a.space == 'MNI152NLin6Asym' and
        a.atlas_type == 'deterministic' and
        100 <= a.n_regions <= 500
        for a in filtered
    )
    
    # Should include Schaefer100, Schaefer200, Schaefer400
    names = {a.name for a in filtered}
    assert 'Schaefer100' in names
    assert 'Schaefer200' in names
    assert 'Schaefer400' in names


def test_register_atlas():
    """Test registering a custom atlas."""
    # Create custom atlas metadata
    custom = AtlasMetadata(
        name='CustomTestAtlas',
        full_name='Custom Test Atlas',
        space='MNI152NLin6Asym',
        resolution=2.0,
        atlas_type='deterministic',
        n_regions=25,
        description='Custom atlas for testing'
    )
    
    # Register it
    register_atlas(custom)
    
    # Verify it's in the registry
    assert 'CustomTestAtlas' in ATLAS_REGISTRY
    assert ATLAS_REGISTRY['CustomTestAtlas'] == custom
    
    # Verify it appears in listings
    atlases = list_atlases()
    names = [a.name for a in atlases]
    assert 'CustomTestAtlas' in names
    
    # Clean up
    del ATLAS_REGISTRY['CustomTestAtlas']


def test_register_atlas_overwrites_with_warning():
    """Test that registering an existing atlas name shows a warning."""
    # Create atlas with same name as existing one
    duplicate = AtlasMetadata(
        name='Schaefer100',
        full_name='Duplicate Schaefer',
        space='MNI152NLin6Asym',
        resolution=2.0,
        atlas_type='deterministic',
        n_regions=100,
        description='Duplicate for testing'
    )
    
    original = ATLAS_REGISTRY['Schaefer100']
    
    # Register should emit warning
    with pytest.warns(UserWarning, match="already exists"):
        register_atlas(duplicate)
    
    # Should be overwritten
    assert ATLAS_REGISTRY['Schaefer100'] == duplicate
    
    # Restore original
    ATLAS_REGISTRY['Schaefer100'] = original


def test_schaefer_atlas_metadata():
    """Test specific Schaefer atlas metadata."""
    schaefer400 = ATLAS_REGISTRY['Schaefer400']
    
    assert schaefer400.name == 'Schaefer400'
    assert 'Schaefer' in schaefer400.full_name
    assert '400' in schaefer400.full_name
    assert schaefer400.space == 'MNI152NLin6Asym'
    assert schaefer400.resolution == 1.0
    assert schaefer400.atlas_type == 'deterministic'
    assert schaefer400.n_regions == 400
    assert schaefer400.bundled is True
    assert schaefer400.filename is not None
    assert 'Schaefer2018' in schaefer400.filename


def test_tian_atlas_metadata():
    """Test specific Tian atlas metadata."""
    tian_s2 = ATLAS_REGISTRY['TianS2']
    
    assert tian_s2.name == 'TianS2'
    assert 'Tian' in tian_s2.full_name
    assert 'Scale 2' in tian_s2.full_name
    assert tian_s2.space == 'MNI152NLin2009cAsym'
    assert tian_s2.resolution == 1.0
    assert tian_s2.atlas_type == 'deterministic'
    assert tian_s2.n_regions == 32
    assert tian_s2.bundled is True


def test_hcp_atlas_metadata():
    """Test HCP1065 atlas metadata."""
    hcp = ATLAS_REGISTRY['HCP1065']
    
    assert hcp.name == 'HCP1065'
    assert 'HCP' in hcp.full_name or 'White Matter' in hcp.full_name
    assert hcp.space == 'MNI152NLin2009aAsym'
    assert hcp.resolution == 1.0
    assert hcp.atlas_type == 'probabilistic'  # White matter is probabilistic
    assert hcp.n_regions == 72
    assert hcp.bundled is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
