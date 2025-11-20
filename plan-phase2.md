# Phase 2 Implementation Plan - Output System Refactoring

## Summary
Based on user decisions and updated requirements from optimizations.txt

## User Decisions
1. **Result Structure:** Nested dict approach ✓
2. **Atlas Labels:** For all bundled atlases, auto-load labels.txt ✓
3. **ConnectivityMatrixResult:** Clean break (early development) ✓
4. **Multi-Atlas Storage:** Nested dict with atlas name as key ✓

---

## Task Groups

### Group A: Result Structure Refactoring (Priority: HIGH)
**Dependencies:** None  
**Estimated Impact:** All analysis classes, LesionData

#### Task A1: Implement Nested Dict Structure
- **File:** `src/lacuna/core/lesion_data.py`
- **Changes:**
  - Update `add_results()` signature to accept dict[str, AnalysisResult]
  - Change internal `_results` from `dict[str, List]` to `dict[str, dict[str, AnalysisResult]]`
  - Update docstrings and examples
- **Example:** `result.results['RegionalDamage']['Schaefer100']` instead of `[0]`

#### Task A2: Add Result Retrieval Helpers
- **Files:** `src/lacuna/core/lesion_data.py`, `src/lacuna/core/output.py`
- **New Methods:**
  - `LesionData.get_result(analysis: str, result_name: str) -> AnalysisResult`
  - `LesionData.list_analyses() -> List[str]`
  - `LesionData.get_analysis_results(analysis: str) -> dict[str, AnalysisResult]`
  - `LesionData.list_result_names(analysis: str) -> List[str]`

---

### Group B: Atlas Region Labels (Priority: HIGH)
**Dependencies:** None  
**Estimated Impact:** Atlas system, ConnectivityMatrixResult

#### Task B1: Extend AtlasMetadata
- **File:** `src/lacuna/assets/atlases/registry.py`
- **Changes:**
  - Add `region_labels: List[str] | None` field to AtlasMetadata
  - Update registry entries with labels for bundled atlases

#### Task B2: Create Label Files
- **Directory:** `src/lacuna/data/atlases/`
- **Action:** Create `{atlas_name}_labels.txt` for each .nii.gz
- **Format:** One label per line, matching ROI indices (0-indexed or 1-indexed?)
- **Priority Atlases:**
  - Schaefer2018_100Parcels7Networks
  - Schaefer2018_200Parcels7Networks
  - Schaefer2018_400Parcels7Networks
  - Any other bundled atlases

#### Task B3: Auto-Load Labels in Registration
- **File:** `src/lacuna/assets/atlases/loader.py`
- **Changes:**
  - Update `register_atlas()` to check for `{atlas_filename}_labels.txt`
  - Load labels automatically if file exists
  - Update `load_atlas()` to return labels (or access via metadata)
- **Note:** Labels optional (fallback to "region_001" if missing)

---

### Group C: ConnectivityMatrixResult Refactoring (Priority: HIGH)
**Dependencies:** B1-B3 (for region labels)  
**Estimated Impact:** StructuralNetworkMapping, FunctionalNetworkMapping

#### Task C1: Refactor ConnectivityMatrixResult (BREAKING)
- **File:** `src/lacuna/core/output.py`
- **Remove:**
  - `lesioned_matrix` attribute
  - `compute_disconnection()` method
- **Keep/Modify:**
  - `matrix: np.ndarray` (generic connectivity matrix)
  - `region_labels: List[str]` (from atlas labels, not "region_XYZ")
  - `get_data()` returns matrix
  - Add `to_dataframe()` helper using actual region labels

#### Task C2: Update StructuralNetworkMapping
- **File:** `src/lacuna/analysis/structural_network_mapping.py`
- **Changes:**
  - Remove disconnection computation from ConnectivityMatrixResult creation
  - Compute disconnection separately
  - Store disconnection as MiscResult or separate VoxelMapResult
  - Update to use atlas region labels from metadata

---

### Group D: TractogramResult Simplification (Priority: MEDIUM)
**Dependencies:** None  
**Estimated Impact:** StructuralNetworkMapping

#### Task D1: Simplify TractogramResult
- **File:** `src/lacuna/core/output.py`
- **Remove:**
  - `n_streamlines` attribute
  - `in_memory` attribute (always False)
  - `_streamlines` private storage
- **Keep:**
  - `tractogram_path: Path` (only stored data)
  - `metadata: dict`
- **Update:**
  - `get_data()` loads .tck file on demand using nibabel or appropriate library
  - Add lazy loading with optional caching

#### Task D2: Update StructuralNetworkMapping for TractogramResult
- **File:** `src/lacuna/analysis/structural_network_mapping.py`
- **Changes:**
  - Remove streamline counting
  - Store only path in TractogramResult
  - Update any code expecting n_streamlines

---

### Group E: Multi-Atlas Support (Priority: HIGH)
**Dependencies:** A1 (nested dict structure), B1-B3 (atlas labels)  
**Estimated Impact:** AtlasAggregation, RegionalDamage

#### Task E1: Update AtlasAggregation for Multi-Atlas
- **File:** `src/lacuna/analysis/atlas_aggregation.py`
- **Changes:**
  - When multiple atlases provided, create one ROIResult per atlas
  - Return `dict[str, ROIResult]` instead of single ROIResult
  - Key = atlas name from AtlasMetadata
  - Each ROIResult gets proper region labels from atlas

#### Task E2: Update RegionalDamage for Multi-Atlas
- **File:** `src/lacuna/analysis/regional_damage.py` (if separate from AtlasAggregation)
- **Changes:** Same as E1

#### Task E3: Update Analysis Base Class
- **File:** `src/lacuna/analysis/base.py`
- **Changes:**
  - Update `_add_results_to_lesion()` to handle dict[str, AnalysisResult]
  - Ensure compatibility with nested dict structure

---

### Group F: Spatial Transformations & Provenance (Priority: MEDIUM)
**Dependencies:** None  
**Estimated Impact:** All transform operations

#### Task F1: Audit Transform Provenance
- **File:** `src/lacuna/spatial/transform.py`
- **Actions:**
  - Review `transform_lesion_data()` provenance recording
  - Ensure TransformationRecord includes all metadata:
    - source_space, target_space
    - transform_path used
    - canonicalization info (if aAsym→cAsym occurred)
    - timestamp, software version

#### Task F2: Add Transformation Communication
- **File:** `src/lacuna/spatial/transform.py`
- **Changes:**
  - Add `logger.info()` when space canonicalization occurs
  - Example: "Using space equivalence: MNI152NLin2009aAsym → MNI152NLin2009cAsym"
  - Ensure users see transformation steps during runtime
  - Already partially implemented, verify coverage

#### Task F3: Implement Reverse Transformation for Voxel Outputs
- **File:** `src/lacuna/spatial/transform.py` or new module
- **Feature:** Transform VoxelMapResult back to original lesion space
- **Function:** `transform_voxel_result(voxel_result: VoxelMapResult, target_space: CoordinateSpace) -> VoxelMapResult`
- **Use Case:** Analysis happens in MNI, transform results back to native space

---

### Group G: Caching Homogenization (Priority: LOW)
**Dependencies:** None  
**Estimated Impact:** Performance, consistency

#### Task G1: Audit Caching Mechanisms
- **Files:**
  - `src/lacuna/spatial/cache.py`
  - `src/lacuna/batch/*` (batch caching)
  - `src/lacuna/assets/*` (asset caching via TemplateFlow)
- **Review:**
  - Cache directory structure consistency
  - Cache key generation patterns
  - Cache invalidation strategies
  - TTL/expiration handling

#### Task G2: Homogenize Cache Patterns
- **Actions:**
  - Unified cache directory: `~/.cache/lacuna/`
  - Consistent subdirectories: transforms/, results/, assets/
  - Document caching strategy in docs/
  - Add cache management utilities (clear, size, stats)

---

### Group H: Testing & Documentation (Priority: HIGH)
**Dependencies:** All above groups  
**Estimated Impact:** Test suite stability

#### Task H1: Update Tests for Nested Dict
- **Files:** All test files using `lesion.results`
- **Changes:**
  - Update from `results['Analysis'][0]` to `results['Analysis']['name']`
  - Test multi-result scenarios
  - Test helper methods

#### Task H2: Update Tests for ConnectivityMatrixResult
- **Files:** Tests for StructuralNetworkMapping, network analysis
- **Changes:**
  - Remove tests for `lesioned_matrix` and `compute_disconnection()`
  - Test region label integration
  - Test generic matrix storage

#### Task H3: Update Tests for TractogramResult
- **Files:** Tests for StructuralNetworkMapping
- **Changes:**
  - Remove `n_streamlines` assertions
  - Test lazy loading via `get_data()`
  - Test path-only storage

#### Task H4: Update Tests for Multi-Atlas
- **Files:** Tests for AtlasAggregation, RegionalDamage
- **Changes:**
  - Test single atlas → single result in nested dict
  - Test multiple atlases → multiple results in nested dict
  - Test atlas name as key

#### Task H5: Run Full Test Suite
- **Command:** `make test-fast && make ci-native`
- **Goal:** All tests passing with new architecture

---

## Implementation Order (Recommended)

### Day 1: Foundation
1. **B1-B3:** Atlas labels infrastructure (enables other features)
2. **A1:** Nested dict structure (breaking change, do early)
3. **H1:** Update tests for nested dict (validate A1)

### Day 2: Result Types
4. **C1:** Refactor ConnectivityMatrixResult (breaking change)
5. **D1:** Simplify TractogramResult (breaking change)
6. **H2-H3:** Update tests for new result types

### Day 3: Analysis Updates
7. **C2:** Update StructuralNetworkMapping for new ConnectivityMatrix
8. **D2:** Update StructuralNetworkMapping for new Tractogram
9. **E1-E2:** Multi-atlas support in AtlasAggregation/RegionalDamage
10. **H4:** Update tests for multi-atlas

### Day 4: Polish & Features
11. **A2:** Result retrieval helpers
12. **F1-F2:** Provenance audit and user communication
13. **F3:** Reverse transformation feature (if time)

### Day 5: Validation & Documentation
14. **H5:** Full test suite run
15. **G1-G2:** Caching audit (if time)
16. **Documentation:** Update user guide, API docs, examples

### Day 6: Let's discuss whether it makes sense to include actual files as fixtures (e.g. a lightweight .tck file) or not. 
---

## Breaking Changes Summary

1. **LesionData.results structure:** `dict[str, List]` → `dict[str, dict[str, AnalysisResult]]`
2. **ConnectivityMatrixResult:** Removed `lesioned_matrix`, `compute_disconnection()`
3. **TractogramResult:** Removed `n_streamlines`, `in_memory`
4. **Multi-atlas results:** Now returns dict instead of single combined result
5. **LesionData.add_results():** Now accepts `dict[str, AnalysisResult]` instead of `List`

---

## Questions for Tomorrow
- [ ] Atlas label format: 0-indexed or 1-indexed? -> 1-indexed preferred (0 is background)
- [ ] Should we add deprecation warnings before removing features, or clean break is OK? -> Clean break preferred
- [ ] What library to use for loading .tck files in TractogramResult.get_data()? nibabel
- [ ] Cache directory structure preferences? 

---

## Progress Tracking
- Phase 1: ✅ COMPLETE (LesionData core improvements)
- Phase 2: 🔄 PLANNED (0/15 tasks complete)
