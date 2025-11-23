# Atlas → Parcellation Refactoring Plan

**Status**: In Progress  
**Date**: 2025-11-23  
**Branch**: 003-package-optimization

## Rationale

"Parcellation" is the more precise neuroimaging term for dividing the brain into regions. "Atlas" technically refers to the spatial reference/template. The package currently mixes these terms inconsistently.

## Scope

### Phase 1: Asset Module Renaming (T160-T163)
- **Directory**: `src/lacuna/assets/atlases/` → `src/lacuna/assets/parcellations/`
- **Module**: `loader.py`, `registry.py`, `__init__.py`
- **Classes**: `Atlas` → `Parcellation`, `AtlasMetadata` → `ParcellationMetadata`
- **Functions**: `load_atlas()` → `load_parcellation()`, `list_atlases()` → `list_parcellations()`, etc.
- **Constants**: `ATLAS_REGISTRY` → `PARCELLATION_REGISTRY`, `BUNDLED_ATLASES_DIR` → `BUNDLED_PARCELLATIONS_DIR`

### Phase 2: Analysis Module Updates (T164-T166)
- **ParcelAggregation**:
  - Keep class name (already uses "parcel" correctly)
  - Parameters: `parcel_names` ✓ (already correct), `atlas_names` → `parcellation_names` 
  - Internal: `atlas_dir`, `atlases`, `_load_atlases_from_registry()` → parcellation equivalents
  - Docstrings: Update all "atlas" references to "parcellation"
  
- **RegionalDamage**:
  - Parameter: `parcel_names` ✓ (already correct)
  - Add missing `log_level` parameter (T165)
  - Update docstrings

- **StructuralNetworkMapping** / **FunctionalNetworkMapping**:
  - Parameters: `atlas_name` → `parcellation_name`
  - Update internal references

### Phase 3: BaseAnalysis Enhancement (T167)
- Make `log_level` enforcement clearer in ABC
- Consider adding abstract property or validation
- Ensure all analyses accept `log_level` in `__init__`

### Phase 4: Testing (T168)
- Update all test files referencing atlases
- Fix imports in 460+ tests
- Ensure backward compatibility where needed
- Add deprecation warnings for old names

### Phase 5: Documentation (T169)
- Update notebooks
- Update README and migration guides
- Update API examples
- Update docstrings package-wide

## Implementation Strategy (TDD)

1. **Write tests first** for each component:
   - Test that parcellation functions exist and work
   - Test that old atlas names raise deprecation warnings
   - Test log_level parameter propagation

2. **Implement changes** to make tests pass:
   - Rename modules/classes/functions
   - Add deprecation wrappers for old names
   - Update all references

3. **Run full test suite** after each phase:
   - Ensure 460+ existing tests still pass
   - Fix any breakages immediately

## File Impact Analysis

### High Impact (Direct Changes)
```
src/lacuna/assets/atlases/                → parcellations/
src/lacuna/analysis/parcel_aggregation.py → Update internals
src/lacuna/analysis/regional_damage.py    → Add log_level
src/lacuna/analysis/structural_network_mapping.py → Update param names
src/lacuna/analysis/functional_network_mapping.py → Update param names
```

### Medium Impact (Import Updates)
```
tests/unit/test_atlas_*.py                → Update imports and assertions
tests/contract/test_*_contract.py         → Update parameter names
tests/integration/                        → Update test data
```

### Low Impact (Documentation Only)
```
docs/*.md
examples/*.ipynb
notebooks/*.ipynb
README.md
```

## Backward Compatibility

### Deprecation Strategy
- Keep old names as aliases with `DeprecationWarning`
- Maintain for 1-2 release cycles
- Clear migration path in docs

Example:
```python
# In parcellations/__init__.py
from .loader import Parcellation, load_parcellation

# Deprecated aliases
Atlas = Parcellation  # Deprecated
load_atlas = load_parcellation  # Deprecated

def __getattr__(name):
    if name == "Atlas":
        warnings.warn(
            "Atlas is deprecated, use Parcellation instead",
            DeprecationWarning, stacklevel=2
        )
        return Parcellation
    ...
```

## Task Breakdown

### T160: Plan and document refactoring ✅ (This file)

### T161: Write tests for parcellation naming (TDD)
- Test parcellation module imports
- Test function signatures
- Test parameter name validation
- Test deprecation warnings

### T162: Rename assets/atlases → assets/parcellations
- Move directory
- Update all internal references
- Add backward compatibility aliases

### T163: Update ParcelAggregation for parcellation terminology
- Rename internal variables/methods
- Update docstrings
- Keep `parcel_names` parameter name

### T164: Add log_level to RegionalDamage
- Write test verifying log_level propagation
- Add parameter to __init__
- Pass to parent ParcelAggregation

### T165: Update analysis parameter names
- `atlas_name` → `parcellation_name` in StructuralNetworkMapping
- `atlas_name` → `parcellation_name` in FunctionalNetworkMapping
- Update all tests

### T166: Enforce log_level in BaseAnalysis
- Document requirement in ABC docstring
- Add validation or abstract property if needed
- Verify all analyses comply

### T167: Update all tests for parcellation terminology
- Fix imports
- Update assertions
- Update test data references

### T168: Update documentation
- Notebooks
- README
- Migration guides
- API examples

### T169: Final integration testing
- Run full test suite (460+ tests)
- Fix any remaining issues
- Verify backward compatibility

## Success Criteria

✅ All references to "atlas" in code → "parcellation"  
✅ Backward compatibility maintained with deprecation warnings  
✅ All 460+ tests passing  
✅ RegionalDamage accepts log_level parameter  
✅ BaseAnalysis clearly documents log_level requirement  
✅ Documentation updated  
✅ No breaking changes for existing users (with warnings)

## Estimated Effort

- Planning & Testing: 2-3 hours
- Implementation: 3-4 hours  
- Testing & Fixes: 1-2 hours
- Documentation: 1 hour

**Total**: ~8 hours of focused work
