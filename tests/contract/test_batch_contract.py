"""
Contract tests for batch processing infrastructure.

These tests define the expected behavior and API contracts for batch processing
functionality without depending on implementation details.
"""

import sys

import pytest

sys.path.insert(0, "/home/marvin/projects/lacuna/src")

from lacuna.analysis.base import BaseAnalysis
from lacuna.batch import batch_process
from lacuna.core.mask_data import MaskData


class MockAnalysis(BaseAnalysis):
    """Mock analysis for testing."""

    batch_strategy = "parallel"

    def __init__(self, should_fail=False):
        super().__init__()
        self.should_fail = should_fail
        self.call_count = 0

    def _validate_inputs(self, mask_data: MaskData) -> None:
        """No validation needed for tests."""
        pass

    def _run_analysis(self, mask_data: MaskData) -> dict:
        """Mock analysis that increments call count."""
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Mock analysis failure")
        return {"mock_result": self.call_count}


class TestBatchProcessAPI:
    """Test batch_process function API and contracts."""

    def test_batch_process_is_callable(self):
        """batch_process should be a callable function."""
        assert callable(batch_process)

    def test_batch_process_accepts_required_parameters(self, synthetic_mask_data):
        """batch_process should accept inputs and analysis parameters."""
        analysis = MockAnalysis()

        # Should not raise
        result = batch_process(
            inputs=[synthetic_mask_data],
            analysis=analysis,
            n_jobs=1,
            show_progress=False,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], MaskData)

    def test_batch_process_returns_list_of_mask_data(self, batch_mask_data_list):
        """batch_process should return a list of MaskData objects."""
        analysis = MockAnalysis()

        result = batch_process(
            inputs=batch_mask_data_list,
            analysis=analysis,
            n_jobs=1,
            show_progress=False,
        )

        assert isinstance(result, list)
        assert len(result) == len(batch_mask_data_list)
        for mask_data in result:
            assert isinstance(mask_data, MaskData)
            # Verify results were added
            assert "MockAnalysis" in mask_data.results

    def test_batch_process_raises_on_empty_list(self):
        """batch_process should raise ValueError if inputs is empty."""
        analysis = MockAnalysis()

        with pytest.raises(ValueError, match="cannot be empty"):
            batch_process(inputs=[], analysis=analysis)

    def test_batch_process_raises_on_invalid_analysis(self, synthetic_mask_data):
        """batch_process should raise ValueError if analysis is not BaseAnalysis."""
        with pytest.raises(ValueError, match="must be a BaseAnalysis instance"):
            batch_process(inputs=[synthetic_mask_data], analysis="not_an_analysis")

    def test_batch_process_accepts_n_jobs_parameter(self, synthetic_mask_data):
        """batch_process should accept n_jobs parameter for parallelization control."""
        analysis = MockAnalysis()

        # Test different n_jobs values
        for n_jobs in [-1, 1, 2, 4]:
            result = batch_process(
                inputs=[synthetic_mask_data],
                analysis=analysis,
                n_jobs=n_jobs,
                show_progress=False,
            )
            assert isinstance(result, list)
            assert len(result) == 1

    def test_batch_process_accepts_show_progress_parameter(self, synthetic_mask_data):
        """batch_process should accept show_progress parameter."""
        analysis = MockAnalysis()

        # Test with progress bar disabled
        result = batch_process(
            inputs=[synthetic_mask_data], analysis=analysis, show_progress=False
        )
        assert isinstance(result, list)

        # Test with progress bar enabled (should not raise)
        result = batch_process(
            inputs=[synthetic_mask_data], analysis=analysis, show_progress=True
        )
        assert isinstance(result, list)

    def test_batch_process_accepts_strategy_parameter(self, synthetic_mask_data):
        """batch_process should accept optional strategy parameter."""
        analysis = MockAnalysis()

        # Test with explicit strategy
        result = batch_process(
            inputs=[synthetic_mask_data],
            analysis=analysis,
            strategy="parallel",
            show_progress=False,
        )
        assert isinstance(result, list)

        # Test with None (auto-selection)
        result = batch_process(
            inputs=[synthetic_mask_data],
            analysis=analysis,
            strategy=None,
            show_progress=False,
        )
        assert isinstance(result, list)


class TestBatchStrategyContract:
    """Test BatchStrategy abstract base class contract."""

    def test_batch_strategy_is_abstract(self):
        """BatchStrategy should be an abstract base class."""
        from abc import ABC

        from lacuna.batch.strategies import BatchStrategy

        assert issubclass(BatchStrategy, ABC)

    def test_batch_strategy_has_execute_method(self):
        """BatchStrategy should define execute() abstract method."""
        from lacuna.batch.strategies import BatchStrategy

        assert hasattr(BatchStrategy, "execute")
        assert callable(BatchStrategy.execute)

    def test_batch_strategy_has_name_property(self):
        """BatchStrategy should define name property."""
        from lacuna.batch.strategies import BatchStrategy

        assert hasattr(BatchStrategy, "name")

    def test_parallel_strategy_is_batch_strategy(self):
        """ParallelStrategy should be a subclass of BatchStrategy."""
        from lacuna.batch.strategies import BatchStrategy, ParallelStrategy

        assert issubclass(ParallelStrategy, BatchStrategy)

    def test_parallel_strategy_has_name(self):
        """ParallelStrategy should have name property returning 'parallel'."""
        from lacuna.batch.strategies import ParallelStrategy

        strategy = ParallelStrategy(n_jobs=1)
        assert strategy.name == "parallel"


class TestBaseAnalysisBatchIntegration:
    """Test BaseAnalysis integration with batch processing."""

    def test_base_analysis_has_batch_strategy_attribute(self):
        """BaseAnalysis should have batch_strategy class attribute."""
        from lacuna.analysis.base import BaseAnalysis

        assert hasattr(BaseAnalysis, "batch_strategy")
        assert isinstance(BaseAnalysis.batch_strategy, str)
        assert BaseAnalysis.batch_strategy == "parallel"

    def test_regional_damage_has_batch_strategy(self):
        """RegionalDamage should declare batch_strategy."""
        from lacuna.analysis import RegionalDamage

        assert hasattr(RegionalDamage, "batch_strategy")
        assert RegionalDamage.batch_strategy == "parallel"

    def test_atlas_aggregation_has_batch_strategy(self):
        """ParcelAggregation should declare batch_strategy."""
        from lacuna.analysis import ParcelAggregation

        assert hasattr(ParcelAggregation, "batch_strategy")
        assert ParcelAggregation.batch_strategy == "parallel"


class TestBatchProcessBackend:
    """Test backend parameter functionality."""

    def test_batch_process_accepts_backend_parameter(self):
        """batch_process should accept backend parameter."""
        import inspect

        sig = inspect.signature(batch_process)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "loky"

    def test_parallel_strategy_accepts_backend_parameter(self):
        """ParallelStrategy should accept backend parameter in constructor."""
        import inspect

        from lacuna.batch.strategies import ParallelStrategy

        sig = inspect.signature(ParallelStrategy.__init__)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "loky"

    def test_parallel_strategy_stores_backend(self):
        """ParallelStrategy should store backend parameter."""
        from lacuna.batch.strategies import ParallelStrategy

        strategy = ParallelStrategy(n_jobs=2, backend="threading")
        assert hasattr(strategy, "backend")
        assert strategy.backend == "threading"

    def test_parallel_strategy_default_backend_is_loky(self):
        """ParallelStrategy should default to loky backend."""
        from lacuna.batch.strategies import ParallelStrategy

        strategy = ParallelStrategy(n_jobs=2)
        assert strategy.backend == "loky"

    def test_select_strategy_accepts_backend_parameter(self):
        """select_strategy should accept backend parameter."""
        import inspect

        from lacuna.batch.selection import select_strategy

        sig = inspect.signature(select_strategy)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "loky"

    def test_select_strategy_passes_backend_to_strategy(self):
        """select_strategy should pass backend to created strategy."""
        from lacuna.batch.selection import select_strategy

        analysis = MockAnalysis()
        strategy = select_strategy(analysis=analysis, n_subjects=10, n_jobs=2, backend="threading")
        assert strategy.backend == "threading"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
