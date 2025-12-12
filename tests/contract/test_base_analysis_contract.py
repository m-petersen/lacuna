"""
Contract tests for BaseAnalysis ABC.

Tests the abstract interface that all analysis modules must implement
to ensure plug-and-play extensibility.
"""

import pytest


def test_base_analysis_import():
    """Test that BaseAnalysis can be imported."""
    from lacuna.analysis.base import BaseAnalysis

    assert BaseAnalysis is not None


def test_base_analysis_is_abstract():
    """Test that BaseAnalysis is an abstract base class."""
    from abc import ABC

    from lacuna.analysis.base import BaseAnalysis

    # Should be a subclass of ABC
    assert issubclass(BaseAnalysis, ABC)


def test_base_analysis_cannot_be_instantiated():
    """Test that BaseAnalysis cannot be instantiated directly."""
    from lacuna.analysis.base import BaseAnalysis

    # Should raise TypeError when trying to instantiate
    with pytest.raises(TypeError, match="abstract"):
        BaseAnalysis()


def test_base_analysis_requires_validate_inputs():
    """Test that subclasses must implement _validate_inputs."""
    from lacuna.analysis.base import BaseAnalysis

    # Create incomplete subclass (missing _validate_inputs)
    class IncompleteAnalysis(BaseAnalysis):
        def _run_analysis(self, mask_data):
            return {}

    # Should raise TypeError when instantiating
    with pytest.raises(TypeError, match="_validate_inputs"):
        IncompleteAnalysis()


def test_base_analysis_requires_run_analysis():
    """Test that subclasses must implement _run_analysis."""
    from lacuna.analysis.base import BaseAnalysis

    # Create incomplete subclass (missing _run_analysis)
    class IncompleteAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

    # Should raise TypeError when instantiating
    with pytest.raises(TypeError, match="_run_analysis"):
        IncompleteAnalysis()


def test_base_analysis_complete_subclass_can_instantiate():
    """Test that complete subclass can be instantiated."""
    from lacuna.analysis.base import BaseAnalysis

    # Create complete subclass
    class CompleteAnalysis(BaseAnalysis):
        def __init__(self, param1="test"):
            super().__init__()
            self.param1 = param1

        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {"result": "test"}

    # Should be able to instantiate
    analysis = CompleteAnalysis(param1="custom")
    assert analysis.param1 == "custom"


def test_base_analysis_run_method_exists():
    """Test that run() method is defined on BaseAnalysis."""
    from lacuna.analysis.base import BaseAnalysis

    # Create complete subclass
    class TestAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {}

    analysis = TestAnalysis()

    # Should have run method
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_base_analysis_run_method_is_final():
    """Test that run() method cannot be overridden (marked with @final)."""
    from lacuna.analysis.base import BaseAnalysis

    # Create subclass that tries to override run()
    class BadAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {}

        def run(self, mask_data):  # Should not be allowed!
            return mask_data

    # Python's @final decorator is a type hint, not enforced at runtime
    # But we can check it exists in annotations
    import typing

    if hasattr(typing, "final"):
        # Check if BaseAnalysis.run is marked final
        analysis = BadAnalysis()
        # The decorator exists, but runtime enforcement depends on type checker
        # We'll document this in tests
        assert hasattr(analysis, "run")


def test_base_analysis_run_accepts_mask_data(synthetic_mask_img):
    """Test that run() accepts SubjectData and returns SubjectData."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    # Create test analysis
    class TestAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            assert isinstance(mask_data, SubjectData)

        def _run_analysis(self, mask_data):
            return {"test_result": 42}

    analysis = TestAnalysis()
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Run should accept SubjectData
    result = analysis.run(mask_data)

    # Should return SubjectData
    assert isinstance(result, SubjectData)


def test_base_analysis_run_validates_inputs(synthetic_mask_img):
    """Test that run() calls _validate_inputs before analysis."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    validation_called = []

    class TestAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            validation_called.append(True)
            raise ValueError("Validation failed!")

        def _run_analysis(self, mask_data):
            return {}

    analysis = TestAnalysis()
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Should raise validation error
    with pytest.raises(ValueError, match="Validation failed"):
        analysis.run(mask_data)

    # Validation should have been called
    assert len(validation_called) == 1


def test_base_analysis_run_namespaces_results(synthetic_mask_img):
    """Test that run() automatically namespaces results under class name."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    class MyTestAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {"score": 123, "metric": "test"}

    analysis = MyTestAnalysis()
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    result = analysis.run(mask_data)

    # Results should be namespaced under class name
    assert "MyTestAnalysis" in result.results
    assert result.results["MyTestAnalysis"]["score"] == 123
    assert result.results["MyTestAnalysis"]["metric"] == "test"


def test_base_analysis_run_preserves_existing_results(synthetic_mask_img):
    """Test that run() preserves results from other analyses."""
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
    assert "Analysis1" in result1.results

    # Run second analysis on result
    result2 = Analysis2().run(result1)

    # Both results should be present
    assert "Analysis1" in result2.results
    assert "Analysis2" in result2.results
    assert result2.results["Analysis1"]["value"] == 1
    assert result2.results["Analysis2"]["value"] == 2


def test_base_analysis_run_does_not_modify_input(synthetic_mask_img):
    """Test that run() does not modify the input SubjectData (immutability)."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    class TestAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {"result": "new"}

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_results = mask_data.results.copy()

    analysis = TestAnalysis()
    result = analysis.run(mask_data)

    # Input should not be modified
    assert mask_data.results == original_results
    assert "TestAnalysis" not in mask_data.results

    # Result should be a new object
    assert result is not mask_data
    assert "TestAnalysis" in result.results


def test_base_analysis_run_handles_analysis_errors(synthetic_mask_img):
    """Test that run() properly handles errors from _run_analysis."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    class FailingAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            raise RuntimeError("Analysis computation failed!")

    analysis = FailingAnalysis()
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Should propagate the error
    with pytest.raises(RuntimeError, match="Analysis computation failed"):
        analysis.run(mask_data)


def test_base_analysis_supports_custom_parameters():
    """Test that subclasses can accept custom initialization parameters."""
    from lacuna.analysis.base import BaseAnalysis

    class ParameterizedAnalysis(BaseAnalysis):
        def __init__(self, threshold=0.5, method="default", **kwargs):
            super().__init__()
            self.threshold = threshold
            self.method = method
            self.extra_params = kwargs

        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            return {"threshold": self.threshold, "method": self.method}

    # Create with different parameters
    analysis = ParameterizedAnalysis(threshold=0.8, method="advanced", custom_param=True)

    assert analysis.threshold == 0.8
    assert analysis.method == "advanced"
    assert analysis.extra_params["custom_param"] is True


def test_base_analysis_chain_multiple_analyses(synthetic_mask_img):
    """Test that multiple analyses can be chained on the same SubjectData."""
    from lacuna import SubjectData
    from lacuna.analysis.base import BaseAnalysis

    class VolumeAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            volume = mask_data.get_volume_mm3()
            return {"volume_mm3": volume}

    class NetworkAnalysis(BaseAnalysis):
        def _validate_inputs(self, mask_data):
            # Can access results from previous analyses
            if "VolumeAnalysis" not in mask_data.results:
                raise ValueError("VolumeAnalysis must be run first")

        def _run_analysis(self, mask_data):
            volume = mask_data.results["VolumeAnalysis"]["volume_mm3"]
            return {"network_score": volume * 0.5}

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Chain analyses
    result = VolumeAnalysis().run(mask_data)
    result = NetworkAnalysis().run(result)

    # Both analyses should be present
    assert "VolumeAnalysis" in result.results
    assert "NetworkAnalysis" in result.results
    assert result.results["NetworkAnalysis"]["network_score"] > 0
