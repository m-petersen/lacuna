"""
Unit tests for BaseAnalysis functionality.

Tests automatic result namespacing, immutability enforcement, and provenance tracking.
"""


class TestResultNamespacing:
    """Test automatic result namespacing in BaseAnalysis.run()."""

    def test_results_namespaced_by_class_name(self, synthetic_mask_img):
        """Test that results are automatically namespaced under class name."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class VolumeAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"volume_mm3": 123.45}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        result = VolumeAnalysis().run(mask_data)

        # Results should be under "VolumeAnalysis" key
        assert "VolumeAnalysis" in result.results
        assert result.results["VolumeAnalysis"]["volume_mm3"] == 123.45

    def test_multiple_analyses_separate_namespaces(self, synthetic_mask_img):
        """Test that different analyses have separate namespaces."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class Analysis1(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "first"}

        class Analysis2(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "second"}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        result = Analysis1().run(mask_data)
        result = Analysis2().run(result)

        # Both namespaces should exist and be separate
        assert "Analysis1" in result.results
        assert "Analysis2" in result.results
        assert result.results["Analysis1"]["result"] == "first"
        assert result.results["Analysis2"]["result"] == "second"

    def test_namespace_collision_overwrites(self, synthetic_mask_img):
        """Test that running same analysis twice overwrites previous results."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"value": self.value}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Run first time
        result1 = TestAnalysis(value=1).run(mask_data)
        assert result1.results["TestAnalysis"]["value"] == 1

        # Run second time on same data
        result2 = TestAnalysis(value=2).run(result1)

        # Second run should overwrite first
        assert result2.results["TestAnalysis"]["value"] == 2

    def test_namespace_preserves_other_results(self, synthetic_mask_img):
        """Test that new analysis preserves results from other analyses."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class AnalysisA(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"a": 1}

        class AnalysisB(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"b": 2}

        class AnalysisC(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"c": 3}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Run three analyses in sequence
        result = AnalysisA().run(mask_data)
        result = AnalysisB().run(result)
        result = AnalysisC().run(result)

        # All three should be present
        assert len(result.results) == 3
        assert result.results["AnalysisA"]["a"] == 1
        assert result.results["AnalysisB"]["b"] == 2
        assert result.results["AnalysisC"]["c"] == 3

    def test_namespace_with_complex_results(self, synthetic_mask_img):
        """Test namespacing with complex nested result dictionaries."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class ComplexAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {
                    "metrics": {"mean": 10.5, "std": 2.3, "max": 15.2},
                    "regions": ["frontal", "temporal", "parietal"],
                    "network_scores": [0.1, 0.5, 0.8, 0.3],
                    "metadata": {"method": "correlation", "threshold": 0.05},
                }

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        result = ComplexAnalysis().run(mask_data)

        # Complex structure should be preserved under namespace
        assert "ComplexAnalysis" in result.results
        assert result.results["ComplexAnalysis"]["metrics"]["mean"] == 10.5
        assert "frontal" in result.results["ComplexAnalysis"]["regions"]
        assert len(result.results["ComplexAnalysis"]["network_scores"]) == 4


class TestImmutability:
    """Test that BaseAnalysis.run() enforces immutability."""

    def test_input_mask_data_not_modified(self, synthetic_mask_img):
        """Test that input SubjectData object is never modified."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "test"}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        original_results_keys = set(mask_data.results.keys())

        # Run analysis
        result = TestAnalysis().run(mask_data)

        # Input should be unchanged
        assert set(mask_data.results.keys()) == original_results_keys
        assert "TestAnalysis" not in mask_data.results

        # Result should be different object
        assert result is not mask_data

    def test_input_results_dict_not_modified(self, synthetic_mask_img):
        """Test that the input results dictionary is not modified."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class Analysis1(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"value": 1}

        class Analysis2(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"value": 2}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Run first analysis
        result1 = Analysis1().run(mask_data)

        # Store keys from result1
        result1_keys = set(result1.results.keys())

        # Run second analysis on result1
        Analysis2().run(result1)

        # result1 should not be modified (same keys, no new Analysis2)
        assert set(result1.results.keys()) == result1_keys
        assert "Analysis2" not in result1.results
        # The dict object may be different (it's a copy), but contents should match
        assert result1.results["Analysis1"]["value"] == 1

    def test_mask_img_not_modified(self, synthetic_mask_img):
        """Test that running analysis doesn't affect original SubjectData image reference."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                # The analysis can access data but shouldn't modify original
                return {"modified": True}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        original_img_id = id(mask_data.mask_img)

        # Run analysis
        TestAnalysis().run(mask_data)

        # Original SubjectData should still reference the same image object
        assert id(mask_data.mask_img) == original_img_id

    def test_metadata_not_modified(self, synthetic_mask_img):
        """Test that metadata is not modified."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "test"}

        metadata = {"subject_id": "sub-001", "age": 45, "space": "MNI152NLin6Asym", "resolution": 2}
        mask_data = SubjectData(mask_img=synthetic_mask_img, metadata=metadata)

        # Run analysis
        TestAnalysis().run(mask_data)

        # Original metadata should be unchanged (except space and resolution are required)
        assert mask_data.metadata["subject_id"] == "sub-001"
        assert mask_data.metadata["age"] == 45
        assert mask_data.space == "MNI152NLin6Asym"
        assert mask_data.resolution == 2

    def test_chained_analyses_preserve_immutability(self, synthetic_mask_img):
        """Test that chaining multiple analyses maintains immutability."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class A1(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"n": 1}

        class A2(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"n": 2}

        class A3(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"n": 3}

        # Start with clean data
        ld0 = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Chain analyses
        ld1 = A1().run(ld0)
        ld2 = A2().run(ld1)
        ld3 = A3().run(ld2)

        # Each step should be independent
        assert len(ld0.results) == 0
        assert len(ld1.results) == 1
        assert len(ld2.results) == 2
        assert len(ld3.results) == 3

        # Each object should be distinct
        assert ld0 is not ld1 is not ld2 is not ld3


class TestProvenanceTracking:
    """Test provenance recording in BaseAnalysis.run()."""

    def test_provenance_added_after_analysis(self, synthetic_mask_img):
        """Test that provenance is recorded after running analysis."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "test"}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        original_prov_len = len(mask_data.provenance)

        result = TestAnalysis().run(mask_data)

        # Provenance should have been added
        assert len(result.provenance) == original_prov_len + 1

    def test_provenance_contains_analysis_name(self, synthetic_mask_img):
        """Test that provenance records the analysis class name."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class MyCustomAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "test"}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        result = MyCustomAnalysis().run(mask_data)

        # Latest provenance should reference the analysis
        latest_prov = result.provenance[-1]
        assert "MyCustomAnalysis" in latest_prov["function"]

    def test_provenance_records_parameters(self, synthetic_mask_img):
        """Test that analysis parameters are recorded in provenance."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis

        class ParameterizedAnalysis(BaseAnalysis):
            def __init__(self, threshold=0.5, method="default"):
                super().__init__()
                self.threshold = threshold
                self.method = method

            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                return {"result": "test"}

            def _get_parameters(self):
                return {"threshold": self.threshold, "method": self.method}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        result = ParameterizedAnalysis(threshold=0.8, method="advanced").run(mask_data)

        # Provenance should contain parameters
        latest_prov = result.provenance[-1]
        assert "parameters" in latest_prov
        assert latest_prov["parameters"]["threshold"] == 0.8
        assert latest_prov["parameters"]["method"] == "advanced"


# T020: Unit test for result key generation with source context
class TestResultKeyGeneration:
    """Test that BaseAnalysis generates descriptive result keys with source context."""

    def test_result_key_includes_source_name(self, synthetic_mask_img):
        """Test that result keys include source information (e.g., atlas name)."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis
        from lacuna.core.data_types import ParcelData

        class MockAtlasAnalysis(BaseAnalysis):
            def __init__(self, atlas_name):
                super().__init__()
                self.atlas_name = atlas_name

            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                # Should generate key like "atlas_DKT"
                result = ParcelData(
                    name=self.atlas_name,
                    data={"region1": 0.5},
                )
                return {f"atlas_{self.atlas_name}": result}

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        # Run analysis
        result = MockAtlasAnalysis(atlas_name="DKT").run(mask_data)

        # Result should be in dict with descriptive key
        assert "MockAtlasAnalysis" in result.results
        analysis_results = result.results["MockAtlasAnalysis"]
        assert isinstance(analysis_results, dict)
        assert "atlas_DKT" in analysis_results
        assert isinstance(analysis_results["atlas_DKT"], ParcelData)

    def test_result_key_multiple_sources(self, synthetic_mask_img):
        """Test that multiple source-specific results are stored separately."""
        from lacuna import SubjectData
        from lacuna.analysis.base import BaseAnalysis
        from lacuna.core.data_types import ParcelData

        class MultiAtlasAnalysis(BaseAnalysis):
            def _validate_inputs(self, mask_data):
                pass

            def _run_analysis(self, mask_data):
                # Generate results for multiple atlases
                return {
                    "atlas_DKT": ParcelData(name="DKT", data={"r1": 0.1}),
                    "atlas_Schaefer": ParcelData(name="Schaefer", data={"r1": 0.2}),
                    "atlas_HarvardOxford": ParcelData(name="HarvardOxford", data={"r1": 0.3}),
                }

        mask_data = SubjectData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        result = MultiAtlasAnalysis().run(mask_data)

        # All three atlas results should be accessible
        analysis_results = result.results["MultiAtlasAnalysis"]
        assert len(analysis_results) == 3
        assert "atlas_DKT" in analysis_results
        assert "atlas_Schaefer" in analysis_results
        assert "atlas_HarvardOxford" in analysis_results
