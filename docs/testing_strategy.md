# Testing Strategy

## Test Organization

Tests are organized into three categories based on speed and scope:

### 1. **Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual functions/classes in isolation
- **Speed**: Fast (<1s per test typically)
- **Characteristics**:
  - No external dependencies (network, files)
  - Mock external components
  - Focus on single unit of code
  - Run constantly during development

### 2. **Contract Tests** (`tests/contract/`)
- **Purpose**: Validate public API contracts and interfaces
- **Speed**: Fast to Medium (1-5s per test)
- **Characteristics**:
  - Test public API behavior
  - Ensure backwards compatibility
  - Validate input/output contracts
  - Minimal external dependencies

### 3. **Integration Tests** (`tests/integration/`)
- **Purpose**: Test interactions between multiple components
- **Speed**: Medium to Slow (5s-30s per test)
- **Characteristics**:
  - Multiple components working together
  - May use real files/data
  - May download templates (TemplateFlow)
  - May require MRtrix3

## Pytest Markers

Use markers to categorize tests by characteristics:

```python
@pytest.mark.fast          # Quick tests (<1s), core logic
@pytest.mark.slow          # Slow tests (>5s), downloads, heavy computation
@pytest.mark.integration   # Integration tests (multiple components)
@pytest.mark.contract      # Contract/API tests
@pytest.mark.requires_mrtrix       # Requires MRtrix3 installation
@pytest.mark.requires_templateflow # Requires internet for template downloads
```

### Example Usage

```python
import pytest

@pytest.mark.fast
def test_coordinate_space_creation():
    """Fast unit test - no external dependencies."""
    space = CoordinateSpace("MNI152NLin6Asym", resolution=2.0)
    assert space.identifier == "MNI152NLin6Asym"

@pytest.mark.slow
@pytest.mark.requires_templateflow
def test_template_download():
    """Slow test - downloads from TemplateFlow."""
    template = get_template("MNI152NLin6Asym", resolution=2)
    assert template.exists()

@pytest.mark.integration
def test_full_analysis_pipeline():
    """Integration test - multiple components."""
    lesion = LesionData.from_nifti("test.nii.gz")
    analysis = RegionalDamage()
    result = analysis.run(lesion)
    assert "RegionalDamage" in result.results
```

## Development Workflow

### Inner Loop (Constant Use, ~30s)

Run fast tests during active development:

```bash
# Run only unit + contract tests (no slow/integration)
make test-fast

# Or run specific test type
make test-unit        # Unit tests only (~15s)
make test-contract    # Contract tests only (~15s)
```

### Middle Loop (Before Commit, ~2min)

Run all tests to ensure nothing breaks:

```bash
# Run all tests (unit + contract + integration)
make test
```

### Outer Loop (Before Push, ~30s native)

Run full CI validation:

```bash
# Native CI (linting, type checking, tests with coverage)
make ci-native
```

### Verification Loop (Before Push, ~90s Docker)

Verify in Docker environment (optional, only if native CI passes):

```bash
# Docker CI with act
make ci-act
```

## Makefile Commands

```bash
make help              # Show all available commands

# Test commands (from fastest to slowest)
make test-unit         # Unit tests only (~15s)
make test-contract     # Contract tests only (~15s)
make test-fast         # Unit + contract, no slow (~30s)
make test-integration  # Integration tests only (~1min)
make test              # All tests (~2min)
make test-slow         # Only slow/integration tests (~1min)
make test-coverage     # Tests with coverage report (~2min)

# Quality commands
make lint              # Linting checks (~5s)
make format            # Auto-format code (~5s)
make typecheck         # Type checking (~10s)

# CI commands
make ci-native         # Full native CI (~2min)
make ci-act            # Docker CI with act (~90s)
```

## When to Mark Tests as Slow

Mark a test as `@pytest.mark.slow` if it:

1. **Takes >5 seconds** to run
2. **Downloads data** from the internet (TemplateFlow, etc.)
3. **Performs heavy computation** (atlas transformation, tractography)
4. **Requires MRtrix3** operations (TDI computation, tractography)
5. **Processes large files** (>100MB neuroimaging data)

## When to Use Integration Tests

Create integration tests (`tests/integration/`) when:

1. **Testing multiple components** working together
2. **Testing end-to-end workflows** (load → process → save)
3. **Testing spatial transformations** between coordinate spaces
4. **Testing analysis pipelines** with real data
5. **Testing batch processing** workflows

## Benefits

### For Development
- **Faster feedback loop**: Run only relevant tests (~30s vs 2min)
- **Targeted testing**: Test specific components quickly
- **Better productivity**: Don't wait for slow tests during development

### For CI/CD
- **Parallel execution**: Run fast and slow tests in parallel
- **Fail fast**: Quick failures on fast tests, saving time
- **Resource optimization**: Only run heavy tests when needed

### For Contributors
- **Clear expectations**: Know which tests are fast vs slow
- **Easy to run**: Simple make commands for common workflows
- **Self-documenting**: Markers explain test characteristics

## Migration Guide

### Marking Existing Tests

1. **Identify slow tests**:
   ```bash
   pytest --durations=20  # Show 20 slowest tests
   ```

2. **Add markers**:
   ```python
   @pytest.mark.slow
   def test_heavy_computation():
       # Test that takes >5s
       pass
   ```

3. **Categorize by directory**:
   - Move pure unit tests → `tests/unit/`
   - Move contract tests → `tests/contract/`
   - Move integration tests → `tests/integration/`

### Running Specific Test Categories

```bash
# Only fast tests
pytest -m "not slow"

# Only slow tests
pytest -m "slow"

# Only integration tests
pytest tests/integration/

# Only tests that don't require MRtrix
pytest -m "not requires_mrtrix"

# Only tests that don't download anything
pytest -m "not requires_templateflow"
```

## Best Practices

1. **Keep unit tests fast**: Mock external dependencies
2. **Use fixtures wisely**: Shared fixtures for integration tests
3. **Avoid network calls in unit tests**: Use mocks or cached data
4. **Mark tests appropriately**: Help others understand test characteristics
5. **Run fast tests constantly**: Catch issues early
6. **Run all tests before commit**: Ensure full compatibility
7. **Use parallel execution**: `-n auto` for faster test runs

## Future Enhancements

Potential improvements to consider:

1. **Test sharding**: Split tests across multiple CI jobs
2. **Smart test selection**: Run only tests affected by changes
3. **Test timing database**: Track historical test durations
4. **Automated marking**: Auto-detect slow tests and mark them
5. **Coverage-aware testing**: Prioritize tests for changed code
