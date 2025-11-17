# Fix Summary: Template Resolution Bug and Cache Management

## Issues Fixed

### 1. **Template Resolution Order Bug**
**Problem**: In `StructuralNetworkMapping`, when `template=None` (default), the template was being resolved AFTER TDI computation was attempted, causing:
```
TypeError: template must be str, Path, or nibabel.Nifti1Image, got <class 'NoneType'>
```

**Root Cause**: In `_validate_inputs()`, the order of operations was:
1. Compute/load TDI (requires template)
2. Resolve template from registry

**Solution**: Reordered operations in `_validate_inputs()`:
1. Resolve template from registry first
2. Then compute/load TDI using the resolved template

**Files Changed**:
- `src/lacuna/analysis/structural_network_mapping.py`

### 2. **Unified Cache Directory System**
**Problem**: Cache locations were scattered across the codebase using different temporary directories.

**Solution**: Created unified cache management system:
- New module: `src/lacuna/utils/cache.py`
- Functions:
  - `get_cache_dir()`: Main cache directory
  - `get_tdi_cache_dir()`: TDI cache subdirectory
  - `get_transform_cache_dir()`: Transform cache subdirectory

**Cache Locations** (platform-aware):
- **Linux/macOS**: `~/.cache/lacuna/` (or `$XDG_CACHE_HOME/lacuna`)
- **Windows**: `%LOCALAPPDATA%\lacuna\cache`
- **Custom**: Set `LACUNA_CACHE_DIR` environment variable

**Environment Variable Configuration**:
```bash
# In terminal
export LACUNA_CACHE_DIR=/path/to/custom/cache

# In Python (before importing lacuna)
import os
os.environ['LACUNA_CACHE_DIR'] = '/path/to/custom/cache'

# In Jupyter
%env LACUNA_CACHE_DIR=/path/to/custom/cache
```

**Files Changed**:
- `src/lacuna/utils/cache.py` (new file)
- `src/lacuna/analysis/structural_network_mapping.py` (updated to use `get_tdi_cache_dir()`)

### 3. **Comprehensive Tests**
**Added**: `tests/unit/test_template_resolution_fix.py`

**Tests**:
1. `test_template_resolved_before_tdi_computation()`: Verifies template is resolved before TDI computation
2. `test_cache_directory_uses_unified_location()`: Verifies TDI cache uses unified directory
3. `test_cache_directory_respects_env_variable()`: Verifies `LACUNA_CACHE_DIR` is respected

**Test Results**: All 3 tests passing ✓

### 4. **Batch Processing Examples**
**Added**: Section 12 to `notebooks/comprehensive_package_test.ipynb`

**Examples Include**:
- **12.1 Regional Damage**: Parallel batch processing with CSV export
- **12.2 Structural Network Mapping**: Automatic TDI caching demonstration
- **12.3 Functional Network Mapping**: Sequential processing with method comparison (BOES vs PINI)
- **12.4 Cache Management**: Cache inspection and configuration

**Key Features Demonstrated**:
- Automatic TDI caching (computed once, reused across batch)
- Memory-efficient mode (`load_to_memory=False`)
- Custom cache location configuration
- Result export (CSV, NIfTI)
- Progress tracking

## API Impact

### No Breaking Changes
- All existing code continues to work
- New cache system is transparent to users
- Default behavior unchanged

### New Features
- Environment variable: `LACUNA_CACHE_DIR`
- Unified cache functions in `lacuna.utils.cache`
- Better cache management and cleanup

## Migration Guide

### For Users
No action required! The fix is backwards-compatible.

**Optional**: Configure custom cache location:
```python
import os
os.environ['LACUNA_CACHE_DIR'] = '/my/custom/cache'
from lacuna.analysis import StructuralNetworkMapping
```

### For Developers
If you were using temporary directories directly:
```python
# Old (don't do this)
import tempfile
cache_dir = Path(tempfile.gettempdir()) / "my_cache"

# New (recommended)
from lacuna.utils.cache import get_cache_dir
cache_dir = get_cache_dir() / "my_cache"
```

## Testing Checklist

- [x] Template resolution bug fixed
- [x] Unified cache system implemented
- [x] Environment variable support added
- [x] Tests added and passing
- [x] Batch processing examples added to notebook
- [x] Documentation updated
- [x] No breaking changes

## Files Changed Summary

### New Files
1. `src/lacuna/utils/cache.py` - Unified cache management
2. `tests/unit/test_template_resolution_fix.py` - Regression tests

### Modified Files
1. `src/lacuna/analysis/structural_network_mapping.py`:
   - Fixed template resolution order in `_validate_inputs()`
   - Updated to use `get_tdi_cache_dir()` instead of `tempfile.gettempdir()`
   - Added import: `from lacuna.utils.cache import get_tdi_cache_dir`

2. `notebooks/comprehensive_package_test.ipynb`:
   - Added Section 12: Batch Processing Examples
   - Updated summary to reflect new examples

## Performance Impact

**Positive**:
- TDI caching now uses proper cache directory (persistent across sessions)
- Batch processing more efficient (TDI computed once, reused)

**Neutral**:
- Cache location lookup adds negligible overhead (~microseconds)

## Security & Privacy

- Cache directory respects platform conventions (user-specific)
- No data leakage between users
- Cache can be moved to secure/encrypted locations via `LACUNA_CACHE_DIR`

## Future Work

**Optional Enhancements**:
1. Cache size management (auto-cleanup old files)
2. Cache statistics/monitoring
3. Cache compression
4. Parallel cache warming for batch jobs

## References

- Issue: Template resolution bug in StructuralNetworkMapping
- Related: TDI caching optimization
- Impacts: Batch processing workflows
