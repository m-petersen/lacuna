"""Unit tests for analysis object repr and str methods."""

from lacuna.analysis import (
    AtlasAggregation,
    FunctionalNetworkMapping,
    RegionalDamage,
    StructuralNetworkMapping,
)


class TestFunctionalNetworkMappingRepr:
    """Tests for FunctionalNetworkMapping __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5", method="boes")

        repr_str = repr(analysis)

        assert "FunctionalNetworkMapping(" in repr_str
        assert "connectome_path='/path/to/connectome.h5'" in repr_str
        assert "method='boes'" in repr_str
        assert repr_str.endswith(")")

    def test_repr_all_parameters(self):
        """Test __repr__ includes all parameters."""
        analysis = FunctionalNetworkMapping(
            connectome_path="/path/to/data.h5",
            method="pini",
            pini_percentile=30,
            compute_t_map=True,
            t_threshold=2.5,
            verbose=True,
        )

        repr_str = repr(analysis)

        assert "method='pini'" in repr_str
        assert "pini_percentile=30" in repr_str
        assert "compute_t_map=True" in repr_str
        assert "t_threshold=2.5" in repr_str
        assert "verbose=True" in repr_str

    def test_repr_long_path_truncated(self):
        """Test that very long paths are truncated in repr."""
        long_path = "/very/long/path/" + "x" * 100 + "/connectome.h5"
        analysis = FunctionalNetworkMapping(connectome_path=long_path, method="boes")

        repr_str = repr(analysis)

        # Should be truncated to 50 chars with "..."
        assert len(repr_str) < len(long_path) + 100
        assert "..." in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = FunctionalNetworkMapping(
            connectome_path="/path/to/connectome.h5",
            method="boes",
            compute_t_map=True,
            t_threshold=2.0,
        )

        str_output = str(analysis)

        assert "FunctionalNetworkMapping Analysis" in str_output
        assert "Configuration:" in str_output
        assert "connectome_path: /path/to/connectome.h5" in str_output
        assert "method: boes" in str_output
        assert "compute_t_map: True" in str_output
        assert "t_threshold: 2.0" in str_output

    def test_str_multiline(self):
        """Test that __str__ produces multiple lines."""
        analysis = FunctionalNetworkMapping(connectome_path="/path/to/connectome.h5", method="boes")

        str_output = str(analysis)
        lines = str_output.split("\n")

        assert len(lines) > 5  # Should have multiple lines
        assert all(line.strip() for line in lines if line)  # No empty content


class TestStructuralNetworkMappingRepr:
    """Tests for StructuralNetworkMapping __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        analysis = StructuralNetworkMapping(
            tractogram_path="/path/to/tracts.tck",
            check_dependencies=False,
        )

        repr_str = repr(analysis)

        assert "StructuralNetworkMapping(" in repr_str
        assert "tractogram_path='/path/to/tracts.tck'" in repr_str

    def test_repr_with_atlas(self):
        """Test __repr__ includes atlas parameters."""
        analysis = StructuralNetworkMapping(
            tractogram_path="/path/to/tracts.tck",
            atlas_name="schaefer100",
            compute_lesioned=True,
            check_dependencies=False,
        )

        repr_str = repr(analysis)

        assert "atlas_name='schaefer100'" in repr_str
        assert "compute_lesioned=True" in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = StructuralNetworkMapping(
            tractogram_path="/path/to/tracts.tck",
            atlas_name="schaefer100",
            n_jobs=4,
            verbose=False,
            check_dependencies=False,
        )

        str_output = str(analysis)

        assert "StructuralNetworkMapping Analysis" in str_output
        assert "Configuration:" in str_output
        assert "tractogram_path: /path/to/tracts.tck" in str_output
        assert "atlas_name: schaefer100" in str_output
        assert "n_jobs: 4" in str_output


class TestRegionalDamageRepr:
    """Tests for RegionalDamage __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        analysis = RegionalDamage(threshold=0.5)

        repr_str = repr(analysis)

        assert "RegionalDamage(" in repr_str
        assert "threshold=" in repr_str
        assert "analysis_type='RegionalDamage'" in repr_str

    def test_repr_with_atlas_names(self):
        """Test __repr__ includes atlas names."""
        analysis = RegionalDamage(threshold=0.3, atlas_names=["AAL3", "Schaefer2018"])

        repr_str = repr(analysis)

        assert "threshold=" in repr_str
        assert "atlas_names=" in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = RegionalDamage(threshold=0.5)

        str_output = str(analysis)

        assert "RegionalDamage Analysis" in str_output
        assert "Configuration:" in str_output
        assert "threshold: 0.5" in str_output
        assert "analysis_type: RegionalDamage" in str_output


class TestAtlasAggregationRepr:
    """Tests for AtlasAggregation __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        analysis = AtlasAggregation(source="lesion_img", aggregation="mean", threshold=0.3)

        repr_str = repr(analysis)

        assert "AtlasAggregation(" in repr_str
        assert "source='lesion_img'" in repr_str
        assert "aggregation='mean'" in repr_str
        assert "threshold=0.3" in repr_str

    def test_repr_with_atlas_names(self):
        """Test __repr__ includes atlas names."""
        analysis = AtlasAggregation(
            atlas_names=["AAL3", "Schaefer2018"],
            source="lesion_img",
            aggregation="percent",
        )

        repr_str = repr(analysis)

        assert "atlas_names=" in repr_str
        assert "source='lesion_img'" in repr_str
        assert "aggregation='percent'" in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = AtlasAggregation(source="lesion_img", aggregation="mean")

        str_output = str(analysis)

        assert "AtlasAggregation Analysis" in str_output
        assert "Configuration:" in str_output
        assert "source: lesion_img" in str_output
        assert "aggregation: mean" in str_output


class TestReprConsistency:
    """Tests for consistency across all analysis classes."""

    def test_all_analyses_have_repr(self):
        """Test that all analysis classes implement __repr__."""
        analyses = [
            FunctionalNetworkMapping("/path/data.h5", "boes"),
            StructuralNetworkMapping(
                "/path/tracts.tck", check_dependencies=False
            ),
            RegionalDamage(threshold=0.5),
            AtlasAggregation(source="lesion_img", aggregation="mean"),
        ]

        for analysis in analyses:
            repr_str = repr(analysis)
            # Check basic format: ClassName(param=value, ...)
            assert "(" in repr_str
            assert ")" in repr_str
            assert "=" in repr_str or repr_str.endswith("()")

    def test_all_analyses_have_str(self):
        """Test that all analysis classes implement __str__."""
        analyses = [
            FunctionalNetworkMapping("/path/data.h5", "boes"),
            StructuralNetworkMapping(
                "/path/tracts.tck", "/path/tdi.nii.gz", check_dependencies=False
            ),
            RegionalDamage(threshold=0.5),
            AtlasAggregation(source="lesion_img", aggregation="mean"),
        ]

        for analysis in analyses:
            str_output = str(analysis)
            # Check basic format: ClassName Analysis\nConfiguration:\n- param: value
            assert "Analysis" in str_output
            assert "Configuration:" in str_output
            assert "\n" in str_output  # Should be multiline

    def test_repr_str_different(self):
        """Test that __repr__ and __str__ produce different output."""
        analysis = FunctionalNetworkMapping("/path/data.h5", "boes")

        repr_str = repr(analysis)
        str_output = str(analysis)

        assert repr_str != str_output
        # repr should be more compact
        assert len(repr_str) < len(str_output)
