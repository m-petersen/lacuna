"""Unit tests for analysis object repr and str methods."""

import tempfile
from pathlib import Path

from lacuna.analysis import (
    FunctionalNetworkMapping,
    ParcelAggregation,
    RegionalDamage,
    StructuralNetworkMapping,
)
from lacuna.assets.connectomes import (
    register_functional_connectome,
    register_structural_connectome,
    unregister_functional_connectome,
    unregister_structural_connectome,
)


class TestFunctionalNetworkMappingRepr:
    """Tests for FunctionalNetworkMapping __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        registered = False
        try:
            register_functional_connectome(
                name="test_repr_connectome",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping(
                connectome_name="test_repr_connectome", method="boes"
            )

            repr_str = repr(analysis)

            assert "FunctionalNetworkMapping(" in repr_str
            assert "connectome_name='test_repr_connectome'" in repr_str
            assert "method='boes'" in repr_str
            assert repr_str.endswith(")")
        finally:
            if registered:
                unregister_functional_connectome("test_repr_connectome")
            temp_h5.unlink(missing_ok=True)

    def test_repr_all_parameters(self):
        """Test __repr__ includes all parameters."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        registered = False
        try:
            register_functional_connectome(
                name="test_repr_all_connectome",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping(
                connectome_name="test_repr_all_connectome",
                method="pini",
                pini_percentile=30,
                compute_t_map=True,
                t_threshold=2.5,
            )

            repr_str = repr(analysis)

            assert "method='pini'" in repr_str
            assert "pini_percentile=30" in repr_str
            assert "compute_t_map=True" in repr_str
            assert "t_threshold=2.5" in repr_str
        finally:
            if registered:
                unregister_functional_connectome("test_repr_all_connectome")
            temp_h5.unlink(missing_ok=True)

    def test_repr_long_path_truncated(self):
        """Test that very long paths are truncated in repr."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        long_name = "test_" + "x" * 100 + "_connectome"
        registered = False
        try:
            register_functional_connectome(
                name=long_name,
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping(connectome_name=long_name, method="boes")

            repr_str = repr(analysis)

            # Should have truncation indicator "..."
            # The long connectome name (116 chars) should be truncated
            assert "..." in repr_str
            # Repr should be shorter than having the full 116 char name
            # (full name would make repr ~300+ chars)
            assert len(repr_str) < 300
        finally:
            if registered:
                unregister_functional_connectome(long_name)
            temp_h5.unlink(missing_ok=True)

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        registered = False
        try:
            register_functional_connectome(
                name="test_str_connectome",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping(
                connectome_name="test_str_connectome",
                method="boes",
                compute_t_map=True,
                t_threshold=2.0,
            )

            str_output = str(analysis)

            assert "FunctionalNetworkMapping Analysis" in str_output
            assert "Configuration:" in str_output
            assert "connectome_name: test_str_connectome" in str_output
            assert "method: boes" in str_output
            assert "compute_t_map: True" in str_output
            assert "t_threshold: 2.0" in str_output
        finally:
            if registered:
                unregister_functional_connectome("test_str_connectome")
            temp_h5.unlink(missing_ok=True)

    def test_str_multiline(self):
        """Test that __str__ produces multiple lines."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        registered = False
        try:
            register_functional_connectome(
                name="test_multiline_connectome",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping(
                connectome_name="test_multiline_connectome", method="boes"
            )

            str_output = str(analysis)
            lines = str_output.split("\n")

            assert len(lines) > 5  # Should have multiple lines
            assert all(line.strip() for line in lines if line)  # No empty content
        finally:
            if registered:
                unregister_functional_connectome("test_multiline_connectome")
            temp_h5.unlink(missing_ok=True)


class TestStructuralNetworkMappingRepr:
    """Tests for StructuralNetworkMapping __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
            tractogram_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tdi_path = Path(f.name)

        try:
            register_structural_connectome(
                name="test_struct_repr",
                space="MNI152NLin2009cAsym",
                tractogram_path=tractogram_path,
                tdi_path=tdi_path,
                n_subjects=1000,
                description="Test",
            )

            analysis = StructuralNetworkMapping(
                connectome_name="test_struct_repr",
                check_dependencies=False,
            )

            repr_str = repr(analysis)

            assert "StructuralNetworkMapping(" in repr_str
            assert "connectome_name='test_struct_repr'" in repr_str
        finally:
            unregister_structural_connectome("test_struct_repr")
            tractogram_path.unlink(missing_ok=True)
            tdi_path.unlink(missing_ok=True)

    def test_repr_with_atlas(self):
        """Test __repr__ includes atlas parameters."""
        with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
            tractogram_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tdi_path = Path(f.name)

        try:
            register_structural_connectome(
                name="test_struct_atlas",
                space="MNI152NLin2009cAsym",
                tractogram_path=tractogram_path,
                tdi_path=tdi_path,
                n_subjects=1000,
                description="Test",
            )

            analysis = StructuralNetworkMapping(
                connectome_name="test_struct_atlas",
                parcellation_name="schaefer100",
                compute_lesioned_matrix=True,
                check_dependencies=False,
            )

            repr_str = repr(analysis)

            assert "parcellation_name='schaefer100'" in repr_str
            assert "compute_lesioned_matrix=True" in repr_str
        finally:
            unregister_structural_connectome("test_struct_atlas")
            tractogram_path.unlink(missing_ok=True)
            tdi_path.unlink(missing_ok=True)

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
            tractogram_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tdi_path = Path(f.name)

        try:
            register_structural_connectome(
                name="test_struct_str",
                space="MNI152NLin2009cAsym",
                tractogram_path=tractogram_path,
                tdi_path=tdi_path,
                n_subjects=1000,
                description="Test",
            )

            analysis = StructuralNetworkMapping(
                connectome_name="test_struct_str",
                parcellation_name="schaefer100",
                n_jobs=4,
                verbose=False,
                check_dependencies=False,
            )

            str_output = str(analysis)

            assert "StructuralNetworkMapping Analysis" in str_output
            assert "Configuration:" in str_output
            assert "connectome_name: test_struct_str" in str_output
            assert "parcellation_name: schaefer100" in str_output
            assert "n_jobs: 4" in str_output
        finally:
            unregister_structural_connectome("test_struct_str")
            tractogram_path.unlink(missing_ok=True)
            tdi_path.unlink(missing_ok=True)


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
        analysis = RegionalDamage(
            threshold=0.3, parcel_names=["TianSubcortex_3TS1", "Schaefer2018_100Parcels7Networks"]
        )

        repr_str = repr(analysis)

        assert "threshold=" in repr_str
        assert "parcel_names=" in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = RegionalDamage(threshold=0.5)

        str_output = str(analysis)

        assert "RegionalDamage Analysis" in str_output
        assert "Configuration:" in str_output
        assert "threshold: 0.5" in str_output
        assert "analysis_type: RegionalDamage" in str_output


class TestParcelAggregationRepr:
    """Tests for ParcelAggregation __repr__ and __str__."""

    def test_repr_basic(self):
        """Test __repr__ with basic parameters."""
        analysis = ParcelAggregation(source="maskimg", aggregation="mean", threshold=0.3)

        repr_str = repr(analysis)

        assert "ParcelAggregation(" in repr_str
        assert "source='maskimg'" in repr_str
        assert "aggregation='mean'" in repr_str
        assert "threshold=0.3" in repr_str

    def test_repr_with_atlas_names(self):
        """Test __repr__ includes atlas names."""
        analysis = ParcelAggregation(
            parcel_names=["TianSubcortex_3TS1", "Schaefer2018_100Parcels7Networks"],
            source="maskimg",
            aggregation="percent",
        )

        repr_str = repr(analysis)

        assert "parcel_names=" in repr_str
        assert "source='maskimg'" in repr_str
        assert "aggregation='percent'" in repr_str

    def test_str_formatting(self):
        """Test __str__ provides human-readable output."""
        analysis = ParcelAggregation(source="maskimg", aggregation="mean")

        str_output = str(analysis)

        assert "ParcelAggregation Analysis" in str_output
        assert "Configuration:" in str_output
        assert "source: maskimg" in str_output
        assert "aggregation: mean" in str_output


class TestReprConsistency:
    """Tests for consistency across all analysis classes."""

    def test_all_analyses_have_repr(self):
        """Test that all analysis classes implement __repr__."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
            tractogram_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tdi_path = Path(f.name)

        registered_func = False
        registered_struct = False
        try:
            register_functional_connectome(
                name="test_all_repr_func",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered_func = True

            register_structural_connectome(
                name="test_all_repr_struct",
                space="MNI152NLin2009cAsym",
                tractogram_path=tractogram_path,
                tdi_path=tdi_path,
                n_subjects=1000,
                description="Test",
            )
            registered_struct = True

            analyses = [
                FunctionalNetworkMapping("test_all_repr_func", "boes"),
                StructuralNetworkMapping("test_all_repr_struct", check_dependencies=False),
                RegionalDamage(threshold=0.5),
                ParcelAggregation(source="maskimg", aggregation="mean"),
            ]

            for analysis in analyses:
                repr_str = repr(analysis)
                # Check basic format: ClassName(param=value, ...)
                assert "(" in repr_str
                assert ")" in repr_str
                assert "=" in repr_str or repr_str.endswith("()")
        finally:
            if registered_func:
                unregister_functional_connectome("test_all_repr_func")
            if registered_struct:
                unregister_structural_connectome("test_all_repr_struct")
            temp_h5.unlink(missing_ok=True)
            tractogram_path.unlink(missing_ok=True)
            tdi_path.unlink(missing_ok=True)

    def test_all_analyses_have_str(self):
        """Test that all analysis classes implement __str__."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
            tractogram_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tdi_path = Path(f.name)

        registered_func = False
        registered_struct = False
        try:
            register_functional_connectome(
                name="test_all_str_func",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered_func = True

            register_structural_connectome(
                name="test_all_str_struct",
                space="MNI152NLin2009cAsym",
                tractogram_path=tractogram_path,
                tdi_path=tdi_path,
                n_subjects=1000,
                description="Test",
            )
            registered_struct = True

            analyses = [
                FunctionalNetworkMapping("test_all_str_func", "boes"),
                StructuralNetworkMapping("test_all_str_struct", check_dependencies=False),
                RegionalDamage(threshold=0.5),
                ParcelAggregation(source="maskimg", aggregation="mean"),
            ]

            for analysis in analyses:
                str_output = str(analysis)
                # Check basic format: ClassName Analysis\nConfiguration:\n- param: value
                assert "Analysis" in str_output
                assert "Configuration:" in str_output
                assert "\n" in str_output  # Should be multiline
        finally:
            if registered_func:
                unregister_functional_connectome("test_all_str_func")
            if registered_struct:
                unregister_structural_connectome("test_all_str_struct")
            temp_h5.unlink(missing_ok=True)
            tractogram_path.unlink(missing_ok=True)
            tdi_path.unlink(missing_ok=True)

    def test_repr_str_different(self):
        """Test that __repr__ and __str__ produce different output."""
        import h5py
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5 = Path(f.name)

        # Create minimal valid H5 file
        with h5py.File(temp_h5, "w") as hf:
            hf.create_dataset("timeseries", data=np.random.randn(10, 100, 1000).astype(np.float32))
            hf.create_dataset("mask_indices", data=np.array([range(1000)] * 3))
            hf.create_dataset("mask_affine", data=np.eye(4))
            hf.attrs["mask_shape"] = (91, 109, 91)

        registered = False
        try:
            register_functional_connectome(
                name="test_repr_str_diff",
                space="MNI152NLin6Asym",
                resolution=2.0,
                data_path=temp_h5,
                n_subjects=10,
                description="Test",
            )
            registered = True

            analysis = FunctionalNetworkMapping("test_repr_str_diff", "boes")

            repr_str = repr(analysis)
            str_output = str(analysis)

            assert repr_str != str_output
            # repr should be more compact
            assert len(repr_str) < len(str_output)
        finally:
            if registered:
                unregister_functional_connectome("test_repr_str_diff")
            temp_h5.unlink(missing_ok=True)
