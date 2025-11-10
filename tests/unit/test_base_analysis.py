"""
Unit tests for BaseAnalysis functionality.

Tests automatic result namespacing, immutability enforcement, and provenance tracking.
"""


class TestResultNamespacing:
    """Test automatic result namespacing in BaseAnalysis.run()."""

    def test_results_namespaced_by_class_name(self, synthetic_lesion_img):
        """Test that results are automatically namespaced under class name."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class VolumeAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"volume_mm3": 123.45}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        result = VolumeAnalysis().run(lesion_data)

        # Results should be under "VolumeAnalysis" key
        assert "VolumeAnalysis" in result.results
        assert result.results["VolumeAnalysis"]["volume_mm3"] == 123.45

    def test_multiple_analyses_separate_namespaces(self, synthetic_lesion_img):
        """Test that different analyses have separate namespaces."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class Analysis1(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "first"}

        class Analysis2(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "second"}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        result = Analysis1().run(lesion_data)
        result = Analysis2().run(result)

        # Both namespaces should exist and be separate
        assert "Analysis1" in result.results
        assert "Analysis2" in result.results
        assert result.results["Analysis1"]["result"] == "first"
        assert result.results["Analysis2"]["result"] == "second"

    def test_namespace_collision_overwrites(self, synthetic_lesion_img):
        """Test that running same analysis twice overwrites previous results."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"value": self.value}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

        # Run first time
        result1 = TestAnalysis(value=1).run(lesion_data)
        assert result1.results["TestAnalysis"]["value"] == 1

        # Run second time on same data
        result2 = TestAnalysis(value=2).run(result1)

        # Second run should overwrite first
        assert result2.results["TestAnalysis"]["value"] == 2

    def test_namespace_preserves_other_results(self, synthetic_lesion_img):
        """Test that new analysis preserves results from other analyses."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class AnalysisA(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"a": 1}

        class AnalysisB(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"b": 2}

        class AnalysisC(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"c": 3}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

        # Run three analyses in sequence
        result = AnalysisA().run(lesion_data)
        result = AnalysisB().run(result)
        result = AnalysisC().run(result)

        # All three should be present
        assert len(result.results) == 3
        assert result.results["AnalysisA"]["a"] == 1
        assert result.results["AnalysisB"]["b"] == 2
        assert result.results["AnalysisC"]["c"] == 3

    def test_namespace_with_complex_results(self, synthetic_lesion_img):
        """Test namespacing with complex nested result dictionaries."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class ComplexAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {
                    "metrics": {"mean": 10.5, "std": 2.3, "max": 15.2},
                    "regions": ["frontal", "temporal", "parietal"],
                    "network_scores": [0.1, 0.5, 0.8, 0.3],
                    "metadata": {"method": "correlation", "threshold": 0.05},
                }

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        result = ComplexAnalysis().run(lesion_data)

        # Complex structure should be preserved under namespace
        assert "ComplexAnalysis" in result.results
        assert result.results["ComplexAnalysis"]["metrics"]["mean"] == 10.5
        assert "frontal" in result.results["ComplexAnalysis"]["regions"]
        assert len(result.results["ComplexAnalysis"]["network_scores"]) == 4


class TestImmutability:
    """Test that BaseAnalysis.run() enforces immutability."""

    def test_input_lesion_data_not_modified(self, synthetic_lesion_img):
        """Test that input LesionData object is never modified."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "test"}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        original_results_keys = set(lesion_data.results.keys())

        # Run analysis
        result = TestAnalysis().run(lesion_data)

        # Input should be unchanged
        assert set(lesion_data.results.keys()) == original_results_keys
        assert "TestAnalysis" not in lesion_data.results

        # Result should be different object
        assert result is not lesion_data

    def test_input_results_dict_not_modified(self, synthetic_lesion_img):
        """Test that the input results dictionary is not modified."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class Analysis1(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"value": 1}

        class Analysis2(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"value": 2}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

        # Run first analysis
        result1 = Analysis1().run(lesion_data)

        # Store keys from result1
        result1_keys = set(result1.results.keys())

        # Run second analysis on result1
        result2 = Analysis2().run(result1)

        # result1 should not be modified (same keys, no new Analysis2)
        assert set(result1.results.keys()) == result1_keys
        assert "Analysis2" not in result1.results
        # The dict object may be different (it's a copy), but contents should match
        assert result1.results["Analysis1"]["value"] == 1

    def test_lesion_img_not_modified(self, synthetic_lesion_img):
        """Test that running analysis doesn't affect original LesionData image reference."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                # The analysis can access data but shouldn't modify original
                return {"modified": True}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        original_img_id = id(lesion_data.lesion_img)

        # Run analysis
        TestAnalysis().run(lesion_data)

        # Original LesionData should still reference the same image object
        assert id(lesion_data.lesion_img) == original_img_id

    def test_metadata_not_modified(self, synthetic_lesion_img):
        """Test that metadata is not modified."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "test"}

        metadata = {"subject_id": "sub-001", "age": 45}
        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata=metadata)

        # Run analysis
        result = TestAnalysis().run(lesion_data)

        # Original metadata should be unchanged
        assert lesion_data.metadata["subject_id"] == "sub-001"
        assert lesion_data.metadata["age"] == 45
        assert len(lesion_data.metadata) == 2

    def test_chained_analyses_preserve_immutability(self, synthetic_lesion_img):
        """Test that chaining multiple analyses maintains immutability."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class A1(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"n": 1}

        class A2(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"n": 2}

        class A3(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"n": 3}

        # Start with clean data
        ld0 = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})

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

    def test_provenance_added_after_analysis(self, synthetic_lesion_img):
        """Test that provenance is recorded after running analysis."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "test"}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        original_prov_len = len(lesion_data.provenance)

        result = TestAnalysis().run(lesion_data)

        # Provenance should have been added
        assert len(result.provenance) == original_prov_len + 1

    def test_provenance_contains_analysis_name(self, synthetic_lesion_img):
        """Test that provenance records the analysis class name."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class MyCustomAnalysis(BaseAnalysis):
            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "test"}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        result = MyCustomAnalysis().run(lesion_data)

        # Latest provenance should reference the analysis
        latest_prov = result.provenance[-1]
        assert "MyCustomAnalysis" in latest_prov["function"]

    def test_provenance_records_parameters(self, synthetic_lesion_img):
        """Test that analysis parameters are recorded in provenance."""
        from lacuna import LesionData
        from lacuna.analysis.base import BaseAnalysis

        class ParameterizedAnalysis(BaseAnalysis):
            def __init__(self, threshold=0.5, method="default"):
                super().__init__()
                self.threshold = threshold
                self.method = method

            def _validate_inputs(self, lesion_data):
                pass

            def _run_analysis(self, lesion_data):
                return {"result": "test"}

            def _get_parameters(self):
                return {"threshold": self.threshold, "method": self.method}

        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152_2mm"})
        result = ParameterizedAnalysis(threshold=0.8, method="advanced").run(lesion_data)

        # Provenance should contain parameters
        latest_prov = result.provenance[-1]
        assert "parameters" in latest_prov
        assert latest_prov["parameters"]["threshold"] == 0.8
        assert latest_prov["parameters"]["method"] == "advanced"
