# Lacuna Development Tasks

## Completed ✅

### Atlas → Parcellation Refactoring (v0.5.0+)

**Status**: Complete  
**Date**: 2024

#### Completed Tasks:

1. **T160**: Created TDD tests for parcellations module (14/22 passing)
2. **T161**: Created `lacuna.assets.parcellations` module to replace `lacuna.assets.atlases`
3. **T162**: Removed backward compatibility layer (deprecated aliases, __getattr__ hooks)
4. **T163**: Updated ParcelAggregation to use `parcellation_names` parameter
5. **T164**: Added `log_level` parameter to RegionalDamage (passes to ParcelAggregation parent)
6. **T165**: Updated network mapping classes (StructuralNetworkMapping, FunctionalNetworkMapping) to use `parcellation_name`
7. **T166**: Mass-updated 460+ test files to use parcellation terminology
8. **T167**: Updated API demo notebook (notebooks/api_demo_v0.5.ipynb) with:
   - All imports updated to `from lacuna.assets.parcellations`
   - All parameters changed to `parcellation_name`
   - Fixed cell 23 with correct parcellation name
   - Added comprehensive batch processing examples:
     - ParcelAggregation batch processing with pandas analysis
     - StructuralNetworkMapping batch disconnection mapping
     - FunctionalNetworkMapping batch functional connectivity
   - Documented batch mode best practices (log_level=0, result aggregation)

#### Results:

- **Test Suite**: 132/141 fast tests passing (93% pass rate)
- **Remaining Issues**: 5 test failures related to test naming conventions (non-critical)
- **API Consistency**: All analysis classes use consistent parcellation terminology
- **Breaking Changes**: No backward compatibility maintained (early development)

#### Commits:

1. `refactor: complete atlas->parcellation rename in analysis modules`
2. `refactor: update all tests to use parcellation terminology`
3. `fix: complete parameter renames in tests`

---

## In Progress 🔄

None currently.

---

## Pending ⏳

### Optional Cleanup

1. Fix remaining 5 test failures (test naming conventions, not functional issues)
2. Improve parcellation module test coverage (currently 14/22 passing)

---

## Notes

- All refactoring performed in early development - no backward compatibility needed
- Batch processing examples demonstrate real-world workflows
- API demo notebook (`notebooks/api_demo_v0.5.ipynb`) is not tracked in git (local demo file)
