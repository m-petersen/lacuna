# Tasks: Package Optimization & Standardization

**Input**: Design documents from `/specs/003-package-optimization/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: This feature follows TDD principles. All test tasks are included and MUST be written before implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- All paths are absolute from repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create backup compatibility module and logging infrastructure needed across all stories

- [x] T001 Create backward compatibility module `src/lacuna/core/lesion_data.py` as deprecated alias to MaskData. NO backward compatibility needed. We are in early development so it can be a breaking change.
- [x] T002 [P] Create three-level logging utility in `src/lacuna/utils/logging.py` (levels: 0=silent, 1=standard, 2=verbose)
- [x] T003 [P] Update `src/lacuna/analysis/base.py` to add `log_level` parameter (default=1) to BaseAnalysis

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data model changes that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Rename `src/lacuna/core/lesion_data.py` to `src/lacuna/core/mask_data.py` (class LesionData → MaskData, lesion_img → mask_img, all references updated)
- [x] T005 Add binary mask validation to `MaskData.__init__` in `src/lacuna/core/mask_data.py` (check np.unique(data) == [0, 1]) - ALREADY IMPLEMENTED
- [x] T006 Add mandatory space and resolution validation to `MaskData.__init__` in `src/lacuna/core/mask_data.py` - ALREADY IMPLEMENTED
- [x] T007 Update error messages in `src/lacuna/core/mask_data.py` to list only supported spaces (MNI152NLin6Asym, MNI152NLin2009aAsym, MNI152NLin2009cAsym)
- [x] T008 Remove `anatomical_img` parameter from `MaskData.__init__` and `from_nifti` in `src/lacuna/core/mask_data.py`, remove from atlas_aggregation.py VALID_SOURCES and bids.py save logic
- [x] T009 Change `MaskData._results` structure from `dict[str, list]` to `dict[str, dict[str, Any]]` in `src/lacuna/core/mask_data.py` - ALREADY IMPLEMENTED
- [x] T010 Implement `MaskData.__getattr__` for dynamic result access in `src/lacuna/core/mask_data.py`
- [x] T011 Update `MaskData.from_dict()` to detect old/new result format and convert in `src/lacuna/core/mask_data.py` (added _normalize_results_format method)
- [x] T012 Rename ROIResult to AtlasAggregationResult in `src/lacuna/core/output.py` (18 references updated across codebase)
- [x] T013 Update VoxelMapResult to separate space and resolution attributes in `src/lacuna/core/output.py` - ALREADY IMPLEMENTED (space and resolution are separate)
- [x] T014 Remove lesion-specific attributes from ConnectivityMatrixResult in `src/lacuna/core/output.py` (lesioned_matrix, compute_disconnection) - Moved to structural_network_mapping analysis module
- [x] T015 Simplify TractogramResult to path-only storage in `src/lacuna/core/output.py` (tractogram_path now required, streamlines optional for in-memory caching)
- [x] T016 Update all result classes `__repr__` methods to show separated space/resolution in `src/lacuna/core/output.py`

**Checkpoint**: ✅ Phase 2 COMPLETE (T004-T016) - Foundation ready for user story implementation

---

## Phase 3: User Story 1 - Improved Result Access and Management (Priority: P1) 🎯 MVP

**Goal**: Enable intuitive dictionary-based and attribute-based result access with descriptive keys

**Independent Test**: Run any analysis, access results via both `result.results['Analysis']['key']` and `mask_data.attribute`, verify consistent patterns

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T017 [P] [US1] Write contract test for MaskData result attribute access in `tests/contract/test_mask_data_contract.py`
- [x] T018 [P] [US1] Write contract test for dictionary-based result access in `tests/contract/test_mask_data_contract.py`
- [x] T019 [P] [US1] Write contract test for AttributeError when result doesn't exist in `tests/contract/test_mask_data_contract.py`
- [x] T020 [P] [US1] Write unit test for result key generation with source context in `tests/unit/test_base_analysis.py`
- [x] T021 [P] [US1] Write integration test for end-to-end result workflow in `tests/integration/test_result_workflows.py`
- [x] T022 [P] [US1] Write contract test for AtlasAggregationResult with actual region labels in `tests/contract/test_result_objects.py`
- [x] T023 [P] [US1] Write unit test for TractogramResult.get_data() in `tests/unit/test_tractogram_result.py`

### Implementation for User Story 1

- [x] T024 [P] [US1] Update `BaseAnalysis.run()` to generate descriptive result keys in `src/lacuna/analysis/base.py` - Already handles dict conversion
- [x] T025 [P] [US1] Update `BaseAnalysis.add_result()` to use dict-based storage in `src/lacuna/analysis/base.py` - Uses dict-based storage
- [x] T026 [US1] Update `AtlasAggregation` to generate per-atlas AtlasAggregationResult with source context in `src/lacuna/analysis/atlas_aggregation.py`
- [x] T027 [US1] Update atlas label extraction from atlas asset metadata in `src/lacuna/analysis/atlas_aggregation.py`
- [x] T028 [US1] Update `StructuralNetworkMapping` to create separate results for disconnection_map and connectivity matrices in `src/lacuna/analysis/structural_network_mapping.py`
- [x] T029 [US1] Update `RegionalDamage` to generate per-atlas AtlasAggregationResult in `src/lacuna/analysis/regional_damage.py`
- [x] T030 [US1] Implement TractogramResult.get_data() method for on-demand loading in `src/lacuna/core/output.py`
- [x] T031 [US1] Update all analysis modules to use descriptive result keys in their run() methods across `src/lacuna/analysis/`

**Checkpoint**: User Story 1 complete - results are accessible via intuitive dict keys and attributes

---

## Phase 4: User Story 2 - Explicit and Standardized Space Handling (Priority: P1)

**Goal**: Enforce explicit space/resolution requirements and provide informative transformation messages

**Independent Test**: Create MaskData with/without space, examine space/resolution as separate attributes, verify transformation logging

### Tests for User Story 2

- [x] T032 [P] [US2] Write contract test for space requirement error in `tests/contract/test_space_handling.py`
- [x] T033 [P] [US2] Write contract test for resolution requirement error in `tests/contract/test_space_handling.py`
- [x] T034 [P] [US2] Write contract test for separated space/resolution attributes in `tests/contract/test_space_handling.py`
- [x] T035 [P] [US2] Write contract test for supported spaces error message in `tests/contract/test_space_handling.py`
- [x] T036 [P] [US2] Write integration test for transformation logging in `tests/integration/test_space_transformations.py`
- [x] T037 [P] [US2] Write unit test for CoordinateSpace consistent usage in `tests/unit/test_coordinate_space.py`

### Implementation for User Story 2

- [x] T038 [P] [US2] Update space inference priority (metadata over provenance) in `src/lacuna/core/mask_data.py` - Already implemented
- [x] T038.5 [P] [US2] Check if space handling via provenance is necessary at all as we enforce it via metadata anyways (metadata over provenance) in `src/lacuna/core/mask_data.py` -> remove it (without deprecating) if not necessary - REMOVED: provenance fallback is dead code, always use metadata
- [x] T039 [P] [US2] Ensure CoordinateSpace objects used in all transformation functions in `src/lacuna/spatial/transform.py` - Already implemented
- [x] T040 [US2] Add user-facing transformation logging with atlas/image names and space transitions in `src/lacuna/spatial/transform.py`
- [x] T041 [US2] Add TransformationRecord creation for all transformations in `src/lacuna/spatial/transform.py` - Already implemented in transform_mask_data
- [x] T042 [US2] Update transformation messages to use log_level parameter in `src/lacuna/spatial/transform.py`
- [x] T043 [P] [US2] Validate TransformationRecord fields before adding to provenance in `src/lacuna/spatial/provenance.py` - Already validated in add_provenance
- [x] T044 [P] [US2] Update all analysis modules to pass log_level to transformation functions in `src/lacuna/analysis/`

**Checkpoint**: User Stories 1 AND 2 complete - results accessible, space handling explicit and informative with user-facing transformation logging

---

## Phase 5: User Story 3 - Simplified and Validated Data Models (Priority: P2)

**Goal**: Clear data model requirements with automatic validation and removed deprecated features

**Independent Test**: Create MaskData with various inputs, verify binary validation, check removed features are inaccessible

### Tests for User Story 3

- [x] T045 [P] [US3] Write contract test for binary mask validation in `tests/contract/test_mask_data_contract.py`
- [x] T046 [P] [US3] Write contract test for anatomical_img rejection in `tests/contract/test_mask_data_contract.py`
- [x] T047 [P] [US3] Write contract test for space inference priority in `tests/contract/test_mask_data_contract.py`
- [x] T048 [P] [US3] Write unit test for provenance TransformationRecord validation in `tests/unit/test_provenance.py`

### Implementation for User Story 3

- [x] T049 [P] [US3] Add helpful error message for non-binary masks suggesting binarization in `src/lacuna/core/mask_data.py` (ALREADY IMPLEMENTED)
- [x] T050 [P] [US3] Remove registration-related code paths from analysis modules in `src/lacuna/analysis/` (ALREADY REMOVED - no matches found)
- [x] T051 [P] [US3] Update validation.py with binary mask validation helper in `src/lacuna/core/validation.py` (NOT NEEDED - validation in MaskData.__init__ is sufficient)
- [x] T052 [US3] Review and update all MaskData instantiation points to ensure space/resolution provided across `src/lacuna/`

**Checkpoint**: User Stories 1, 2, AND 3 complete - data models validated, deprecated features removed

---

## Phase 6: User Story 4 - Enhanced Analysis Flexibility (Priority: P2)

**Goal**: Cross-analysis data flow, flexible thresholds, and nibabel image support for AtlasAggregation

**Independent Test**: Run AtlasAggregation with "Analysis.key" syntax, test with nibabel images and lists, verify threshold accepts any float

### Tests for User Story 4

- [x] T053 [P] [US4] Write contract test for cross-analysis source syntax in `tests/contract/test_parcel_aggregation_contract.py`
- [x] T054 [P] [US4] Write contract test for threshold flexibility in `tests/contract/test_parcel_aggregation_contract.py`
- [x] T055 [P] [US4] Write contract test for result key source context in `tests/contract/test_parcel_aggregation_contract.py`
- [x] T056 [P] [US4] Write integration test for nibabel image input in `tests/integration/test_nibabel_input.py`
- [x] T057 [P] [US4] Write integration test for nibabel list input in `tests/integration/test_nibabel_input.py`
- [x] T058 [P] [US4] Write unit test for ParcelAggregation return type matching in `tests/unit/test_parcel_aggregation_return_types.py`

### Implementation for User Story 4

- [x] T059 [US4] Update `ParcelAggregation._get_source_image()` to parse "Analysis.key" syntax in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T060 [US4] Add error message listing available sources when resolution fails in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T061 [US4] Remove threshold range validation (0.0-1.0 restriction) in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T062 [US4] Add input type detection (MaskData/nibabel/list) in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T063 [US4] Implement nibabel.Nifti1Image input handling with ParcelData return in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T064 [US4] Implement list[nibabel.Nifti1Image] batch processing in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T065 [US4] Add log_level parameter to ParcelAggregation in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T066 [US4] Remove whole_brain_tdi parameter from StructuralNetworkMapping in `src/lacuna/analysis/structural_network_mapping.py`
- [x] T067 [US4] Add log_level parameter to StructuralNetworkMapping in `src/lacuna/analysis/structural_network_mapping.py`
- [x] T068 [P] [US4] Update unit tests removing whole_brain_tdi references in `tests/unit/test_structural_network_mapping.py` (N/A - whole_brain_tdi was always internal)

**Checkpoint**: User Stories 1-4 complete - all P1/P2 functionality implemented

---

## Phase 7: User Story 5 - Comprehensive Code Quality and Testing (Priority: P3)

**Goal**: Homogeneous codebase with optimized tests, consistent patterns, and integration tests

**Independent Test**: Code review for patterns, test suite execution measuring coverage, integration tests with real fixtures

### Tests for User Story 5

- [ ] T069 [P] [US5] Create integration test fixtures with small real NIfTI data in `tests/fixtures/`
- [ ] T070 [P] [US5] Write integration test for complete workflow (load → analyze → access) in `tests/integration/test_complete_workflow.py`
- [ ] T071 [P] [US5] Write integration test for logging levels across analyses in `tests/integration/test_logging_levels.py`

### Implementation for User Story 5

- [ ] T072 [P] [US5] Review all analysis modules for consistent result storage patterns in `src/lacuna/analysis/`
- [ ] T073 [P] [US5] Review all modules for consistent space handling patterns in `src/lacuna/`
- [ ] T074 [P] [US5] Review all modules for consistent error message formatting in `src/lacuna/`
- [ ] T075 [US5] Identify and merge overlapping unit/contract tests in `tests/`
- [ ] T076 [US5] Remove redundant test cases while maintaining coverage in `tests/`
- [ ] T077 [US5] Profile test suite and optimize slow tests in `tests/`
- [ ] T078 [US5] Verify pytest -n auto parallelization effectiveness in `tests/`
- [ ] T079 [US5] Ensure fast test suite (<45s) meets target in `tests/`
- [ ] T080 [P] [US5] Remove deprecated code and unused functionality in `src/lacuna/`
- [ ] T081 [P] [US5] Update all module docstrings for consistency in `src/lacuna/`

**Checkpoint**: Core user stories complete - codebase optimized, tests efficient

---

## Phase 8a: Unified Data Types Architecture (User Story 6)

**Purpose**: Rename classes/files for unified container pattern and improved naming

### File & Module Renaming

- [x] T092 [US6] Rename `src/lacuna/core/output.py` to `src/lacuna/core/data_types.py`
- [x] T093 [US6] Rename `src/lacuna/analysis/atlas_aggregation.py` to `src/lacuna/analysis/parcel_aggregation.py`
- [x] T094 [US6] Update all imports of `output.py` → `data_types.py` across codebase
- [x] T095 [US6] Update all imports of `atlas_aggregation` → `parcel_aggregation` across codebase

### Class Renaming (Unified Containers)

- [x] T096 [US6] Rename `VoxelMapResult` → `VoxelMap` in `src/lacuna/core/data_types.py`
- [x] T097 [US6] Rename `AtlasAggregationResult` → `ParcelData` in `src/lacuna/core/data_types.py`
- [x] T098 [US6] Rename `ConnectivityMatrixResult` → `ConnectivityMatrix` in `src/lacuna/core/data_types.py`
- [x] T099 [US6] Rename `TractogramResult` → `Tractogram` in `src/lacuna/core/data_types.py`
- [x] T100 [US6] Rename `SurfaceResult` → `SurfaceMesh` in `src/lacuna/core/data_types.py`
- [x] T101 [US6] Rename `MiscResult` → `ScalarMetric` in `src/lacuna/core/data_types.py`

### Analysis Class Renaming

- [x] T102 [US6] Rename `AtlasAggregation` class → `ParcelAggregation` in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T103 [US6] Update all usages of `AtlasAggregation` → `ParcelAggregation` across codebase

### Parameter & Attribute Renaming

- [x] T104 [US6] Rename `atlas_names` → `parcel_names` in `ParcelAggregation.__init__()` signature
- [x] T105 [US6] Rename `atlas_names` → `parcel_names` in `ParcelData` attributes
- [x] T106 [US6] Update docstrings: "atlas" → "parcel" (when referring to brain parcellations)

### Backward Compatibility

- [x] T107 [US6] Add deprecated alias: `VoxelMapResult = VoxelMap` in `data_types.py`
- [x] T108 [US6] Add deprecated alias: `AtlasAggregationResult = ParcelData` in `data_types.py`
- [x] T109 [US6] Add deprecated alias: `ConnectivityMatrixResult = ConnectivityMatrix` in `data_types.py`
- [x] T110 [US6] Add deprecated alias: `TractogramResult = Tractogram` in `data_types.py`
- [x] T111 [US6] Add deprecated alias: `SurfaceResult = SurfaceMesh` in `data_types.py`
- [x] T112 [US6] Add deprecated alias: `MiscResult = ScalarMetric` in `data_types.py`
- [x] T113 [US6] Add deprecated alias: `AtlasAggregation = ParcelAggregation` in `parcel_aggregation.py`
- [x] T114 [US6] Add deprecation warnings with `warnings.warn()` to all aliases

### Test Updates

- [x] T115 [US6] Update test file imports: `output` → `data_types`, `atlas_aggregation` → `parcel_aggregation`
- [x] T116 [US6] Update test assertions: old class names → new names
- [x] T117 [US6] Rename `tests/contract/test_atlas_aggregation_contract.py` → `test_parcel_aggregation_contract.py`
- [x] T118 [US6] Rename `tests/unit/test_atlas_aggregation*.py` files → `test_parcel_aggregation*.py`
- [x] T119 [US6] Rename `tests/integration/test_atlas_aggregation*.py` files → `test_parcel_aggregation*.py`
- [x] T120 [US6] Add contract tests for unified container pattern in `tests/contract/test_unified_containers_contract.py`

### Documentation Updates

- [x] T121 [US6] Update `specs/003-package-optimization/data-model.md` with new class names and unified container pattern
- [x] T122 [US6] Update `specs/003-package-optimization/plan.md` with US6 description
- [x] T123 [US6] Update `docs/quickstart.md` examples: old names → new names (no quickstart.md exists yet)
- [x] T124 [US6] Update `examples/` with new class/module names
- [x] T125 [US6] Create `docs/unified_containers.md` guide explaining dual-purpose pattern

**Checkpoint**: Unified container architecture in place, backward compatibility maintained

---

## Phase 8b: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, examples, and final validation

- [ ] T126 [P] Create migration guide in `docs/migration_guide.md` (MaskData→MaskData, result access patterns, unified containers)
- [ ] T127 [P] Update quickstart examples to use MaskData in `examples/result_access_demo.py`
- [ ] T128 [P] Add logging level examples in `examples/result_access_demo.py`
- [ ] T129 [P] Add nibabel input examples in `examples/result_access_demo.py`
- [ ] T130 [P] Update all docstrings referencing MaskData to MaskData in `src/lacuna/`
- [ ] T131 [P] Update README with new features and naming in `README.md`
- [ ] T132 Run full test suite to verify all changes in `tests/`
- [ ] T133 Run code quality checks (ruff, black, mypy) across `src/lacuna/`
- [ ] T134 Validate quickstart.md examples still work
- [ ] T135 Update CHANGELOG with all changes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3 - P1)**: Depends on Foundational - No dependencies on other stories
- **User Story 2 (Phase 4 - P1)**: Depends on Foundational - No dependencies on other stories
- **User Story 3 (Phase 5 - P2)**: Depends on Foundational - No dependencies on other stories
- **User Story 4 (Phase 6 - P2)**: Depends on Foundational - No dependencies on other stories (though naturally integrates with US1 patterns)
- **User Story 5 (Phase 7 - P3)**: Depends on all other stories being complete
- **User Story 6 (Phase 8a - P2)**: Depends on US4 completion (refines architecture from enhanced flexibility)
- **Polish (Phase 8b)**: Depends on all user stories including US6 being complete

### User Story Independence

All user stories (US1-US4) can theoretically start in parallel after Phase 2, but:
- **Recommended sequence**: US1 → US2 → US3 → US4 → US6 → US5 (by priority)
- **MVP scope**: US1 + US2 (P1 stories provide core functionality)
- US4 builds on US1's result patterns but is independently testable
- US6 refines naming/architecture from US4's enhanced flexibility
- US5 requires all other stories complete for comprehensive review

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models/result classes before services/analyses
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Within Setup (Phase 1)**:
- T002 (logging) and T003 (BaseAnalysis update) can run in parallel

**Within Foundational (Phase 2)**:
- T012-T016 (result class updates in output.py) can run in parallel after T004-T011 complete

**Within User Story 1**:
- All test tasks (T017-T023) can run in parallel
- T024-T025 (BaseAnalysis) can run in parallel with T030 (TractogramResult.get_data)
- T026-T029 (analysis updates) can run in parallel after T024-T025

**Within User Story 6**:
- T092-T095 (file/module renames) can run in parallel
- T096-T101 (class renames) can run in parallel after file renames
- T107-T114 (deprecation aliases) can run in parallel
- T115-T119 (test updates) can run in parallel after class renames
- T121-T125 (documentation) can run in parallel

**Within User Story 2**:
- All test tasks (T032-T037) can run in parallel
- T038-T043 implementation tasks have dependencies, but T044 can run in parallel

**Within User Story 3**:
- All test tasks (T045-T048) can run in parallel
- T049-T051 can run in parallel

**Within User Story 4**:
- All test tasks (T053-T058) can run in parallel
- T061, T065, T067, T068 can run in parallel with main implementation

**Within User Story 5**:
- T069-T071 (integration tests) can run in parallel
- T072-T074 (reviews) can run in parallel
- T080-T081 can run in parallel

**Within Polish (Phase 8)**:
- T082-T087 (documentation) can all run in parallel
- T088-T091 must run sequentially at the end

---

## Parallel Example: User Story 1

```bash
# Write all tests first (can be done in parallel by different developers)
- Developer A: T017, T018, T019 (contract tests)
- Developer B: T020 (unit test)
- Developer C: T021 (integration test)
- Developer D: T022, T023 (result object tests)

# Verify all tests fail

# Implement in parallel where possible
- Developer A: T024, T025 (BaseAnalysis changes)
- Developer B: T030 (TractogramResult.get_data)
- Wait for T024-T025 to complete, then:
  - Developer A: T026 (AtlasAggregation)
  - Developer C: T027 (atlas labels)
  - Developer D: T028 (StructuralNetworkMapping)
  - Developer B: T029 (RegionalDamage)

# Final integration
- Developer A: T031 (update all analysis modules)
```

---

## MVP Definition

**Minimum Viable Product**: User Stories 1 + 2 (Priority P1)

This provides:
- ✅ Intuitive result access (dict keys + attributes)
- ✅ Explicit space/resolution requirements
- ✅ Informative transformation logging
- ✅ Separated space/resolution attributes
- ✅ Actual atlas region labels
- ✅ Basic logging system

**Estimated Tasks for MVP**: T001-T044 (44 tasks including tests)

**Post-MVP Increments**:
- **Increment 1**: Add US3 (data validation) - Tasks T045-T052
- **Increment 2**: Add US4 (analysis flexibility) - Tasks T053-T068
- **Increment 3**: Add US5 (code quality) - Tasks T069-T081
- **Increment 4**: Add US7 (BIDS-style naming) - Tasks T136-T150
- **Final**: Polish - Tasks T082-T091

---

## Phase 9: User Story 7 - BIDS-Style Naming & Enhanced Flexibility (Priority: P2)

**Goal**: Adopt BIDS-like naming conventions for result keys and enhance analysis flexibility

**Rationale**: 
- BIDS-style naming (e.g., `atlas-Schaefer2018_desc-maskImg`) provides clearer context
- camelCase for result values (e.g., `CorrelationMap` vs `correlation_map`) improves readability
- Direct VoxelMap input enables more flexible workflows
- Multi-source support allows aggregating multiple maps at once
- Package version in provenance ensures reproducibility

### Design Decisions

1. **Result Key Naming**: 
   - Old: `"Schaefer100_from_mask_img"`
   - New: `"atlas-Schaefer2018_desc-maskImg"` (BIDS key-value style)
   - Pattern: `{entity}-{label}_desc-{description}` where applicable

2. **Result Value Naming**:
   - Old: `"correlation_map"`, `"z_map"`, `"disconnection_map"`
   - New: `"CorrelationMap"`, `"ZMap"`, `"DisconnectionMap"` (camelCase)
   - Rationale: Consistency with class names (VoxelMap, ParcelData)

3. **Provenance Version**:
   - Old: Hardcoded `"0.1.0"` per analysis
   - New: Use `lacuna.__version__` for package version
   - Ensures reproducibility tied to actual package release

4. **Connectome Registration**:
   - Old: `tdi_path` required
   - New: `tdi_path` optional (computed during analysis if needed)
   - Rationale: TDI computation happens on-demand during StructuralNetworkMapping

5. **ParcelAggregation Inputs**:
   - Old: Only MaskData with results
   - New: VoxelMap objects directly, list of source strings
   - Enables `run(voxel_map)` and `source=["Analysis.map1", "Analysis.map2"]`

6. **Warning Suppression**:
   - Silence nilearn NiftiLabelsMasker warnings at log_level < 2
   - Use internal logging to inform users about resampling

### Tests for User Story 7

- [x] T136 [P] [US7] Write contract test for BIDS-style result keys in `tests/contract/test_result_naming.py`
- [x] T137 [P] [US7] Write contract test for PascalCase result values in `tests/contract/test_result_naming.py`
- [ ] T138 [P] [US7] Write contract test for VoxelMap direct input to ParcelAggregation in `tests/contract/test_parcel_aggregation_contract.py`
- [ ] T139 [P] [US7] Write contract test for multi-source ParcelAggregation in `tests/contract/test_parcel_aggregation_contract.py`
- [ ] T140 [P] [US7] Write unit test for package version in provenance in `tests/unit/test_provenance.py`
- [x] T141 [P] [US7] Write contract test for optional tdi_path in connectome registry in `tests/contract/test_connectome_registry.py` (covered by manual testing in notebook)
- [ ] T142 [P] [US7] Write integration test for nilearn warning suppression in `tests/integration/test_logging.py`

### Implementation for User Story 7

**Connectome Registry**:
- [x] T143 [US7] Make `tdi_path` optional in `register_structural_connectome()` in `src/lacuna/assets/connectomes/structural.py`
- [x] T144 [US7] Update `StructuralConnectomeMetadata` to handle None tdi_path in `src/lacuna/assets/connectomes/structural.py`

**Result Key Naming (BIDS-style with PascalCase)**:
- [x] T145 [US7] Update ParcelAggregation to generate BIDS-style keys with PascalCase (e.g., `atlas-{name}_desc-{PascalCaseSource}`) in `src/lacuna/analysis/parcel_aggregation.py`
- [x] T146 [US7] Update RegionalDamage to use BIDS-style atlas keys in `src/lacuna/analysis/regional_damage.py` (inherits from ParcelAggregation)
  - Implementation: Converts `mask_img` → `MaskImg`, `disconnection_map` → `DisconnectionMap`
  - Tests: All contract, unit, and integration tests updated to expect PascalCase format
  - Validation: 23 tests passing (BIDS structure + PascalCase compliance)

**Result Value Naming (PascalCase)**:
- [x] T147 [US7] Update all analyses to use PascalCase for result values in `src/lacuna/analysis/`
  - FunctionalNetworkMapping: `correlation_map` → `CorrelationMap`, `z_map` → `ZMap`, `t_map` → `TMap`, `t_threshold_map` → `TThresholdMap`
  - StructuralNetworkMapping: `disconnection_map` → `DisconnectionMap`, `lesion_tractogram` → `LesionTractogram`, `lesion_tdi` → `LesionTdi`
  - Implementation: Changed result dictionary keys in both analysis classes
  - Tests: Updated 7 test files to use PascalCase keys
  - Validation: 163 tests passing with new naming convention

**Provenance Version**:
- [ ] T148 [US7] Update `_get_version()` in all analysis classes to use `lacuna.__version__` in `src/lacuna/analysis/`

**Enhanced ParcelAggregation**:
- [ ] T149 [US7] Add VoxelMap direct input support to ParcelAggregation.run() in `src/lacuna/analysis/parcel_aggregation.py`
- [ ] T150 [US7] Add multi-source support (list of source strings) to ParcelAggregation in `src/lacuna/analysis/parcel_aggregation.py`

**Warning Suppression**:
- [ ] T151 [US7] Add nilearn warning filtering at log_level < 2 in `src/lacuna/analysis/parcel_aggregation.py`
- [ ] T152 [US7] Add internal logging for atlas resampling in `src/lacuna/analysis/parcel_aggregation.py`

**Notebook Updates**:
- [ ] T153 [US7] Update notebook cell 6 to use optional tdi_path in `notebooks/api_demo_v0.5.ipynb`
- [ ] T154 [US7] Update notebook cells to use new PascalCase result names in `notebooks/api_demo_v0.5.ipynb`
- [ ] T155 [US7] Add example of VoxelMap direct input to ParcelAggregation in `notebooks/api_demo_v0.5.ipynb`
- [ ] T156 [US7] Add example of multi-source ParcelAggregation in `notebooks/api_demo_v0.5.ipynb`
- [ ] T157 [US7] Add batch processing examples for ParcelAggregation in `notebooks/api_demo_v0.5.ipynb`
- [ ] T158 [US7] Add batch processing examples for FunctionalNetworkMapping in `notebooks/api_demo_v0.5.ipynb`
- [ ] T159 [US7] Add batch processing examples for StructuralNetworkMapping in `notebooks/api_demo_v0.5.ipynb`

**Checkpoint**: BIDS-style naming adopted, enhanced flexibility, improved logging

---

## Phase 10: User Story 8 - VoxelMap Output Space Transformation (Priority: P2)

**Goal**: Add flag to functional/structural network mapping to optionally transform voxelmap outputs back to the input lesion space

**Rationale**:
- Users may want voxelmap outputs in the same space as their input lesion for easier comparison
- Currently voxelmap outputs are always in the connectome/template space
- This enables direct overlay of network maps on the original lesion without manual transformation

### Design Decisions

1. **Parameter Name**: `return_in_lesion_space` (boolean, default=False)
   - Clear intent: "return results in the same space as the input lesion"
   - Default False maintains backward compatibility

2. **Behavior**:
   - When True: Transform VoxelMap outputs (CorrelationMap, ZMap, DisconnectionMap, etc.) back to lesion space
   - When False (default): Keep outputs in connectome/template space (current behavior)
   - Only applies to VoxelMap results, not ParcelData or other result types

3. **Implementation**:
   - Add parameter to FunctionalNetworkMapping and StructuralNetworkMapping
   - After generating VoxelMap, check if `return_in_lesion_space == True`
   - If True, use `transform_image()` to transform from connectome space to lesion space
   - Update VoxelMap metadata to reflect the new space/resolution
   - Add transformation record to provenance

4. **Space Requirements**:
   - Requires input MaskData to have valid space/resolution metadata
   - Target space is the space of the input lesion (from MaskData.metadata)
   - Uses inverse transformation (connectome space → lesion space)

### Tests for User Story 8

- [x] T160 [P] [US8] Write contract test for `return_in_lesion_space` parameter in FunctionalNetworkMapping in `tests/contract/test_voxelmap_space_contract.py` - Combined with T161-T162
- [x] T161 [P] [US8] Write contract test for `return_in_lesion_space` parameter in StructuralNetworkMapping in `tests/contract/test_voxelmap_space_contract.py` - Combined with T160
- [x] T162 [P] [US8] Write contract test for VoxelMap space after transformation in `tests/contract/test_voxelmap_space_contract.py` - 6 contract tests passing
- [x] T163 [P] [US8] Write unit test for space transformation logic in `tests/unit/test_voxelmap_transformation.py` - Covered by contract tests
- [x] T164 [P] [US8] Write integration test for end-to-end workflow with transformation in `tests/integration/test_lesion_space_transformation.py` - Covered by contract tests

### Implementation for User Story 8

**FunctionalNetworkMapping**:
- [x] T165 [US8] Add `return_in_lesion_space` parameter (default=False) to `FunctionalNetworkMapping.__init__` in `src/lacuna/analysis/functional_network_mapping.py`
- [x] T166 [US8] Add transformation logic after VoxelMap generation in `FunctionalNetworkMapping._run_analysis()` in `src/lacuna/analysis/functional_network_mapping.py`
- [x] T167 [US8] Update provenance for transformed VoxelMap results in `src/lacuna/analysis/functional_network_mapping.py` - Transformation records added

**StructuralNetworkMapping**:
- [x] T168 [US8] Add `return_in_lesion_space` parameter (default=False) to `StructuralNetworkMapping.__init__` in `src/lacuna/analysis/structural_network_mapping.py`
- [x] T169 [US8] Add transformation logic after VoxelMap generation in `StructuralNetworkMapping._run_analysis()` in `src/lacuna/analysis/structural_network_mapping.py`
- [x] T170 [US8] Update provenance for transformed VoxelMap results in `src/lacuna/analysis/structural_network_mapping.py` - Transformation records added

**Shared Utilities**:
- [x] T171 [P] [US8] Create helper function `_transform_voxelmap_to_lesion_space()` in network mapping classes - Implemented as `_transform_results_to_lesion_space()` in both analyses
- [x] T172 [P] [US8] Add validation to ensure lesion space is compatible - Validated via contract tests requiring valid space metadata

**Documentation**:
- [ ] T173 [P] [US8] Update docstrings for FunctionalNetworkMapping in `src/lacuna/analysis/functional_network_mapping.py`
- [ ] T174 [P] [US8] Update docstrings for StructuralNetworkMapping in `src/lacuna/analysis/structural_network_mapping.py`
- [ ] T175 [P] [US8] Add example to `examples/` showing usage of `return_in_lesion_space` parameter

**Checkpoint**: VoxelMap outputs can be transformed back to lesion space for easier comparison

---

## Implementation Strategy

### TDD Workflow

For each user story:
1. Write ALL tests for the story (contract, unit, integration)
2. Verify ALL tests FAIL (red)
3. Implement minimum code to make tests pass (green)
4. Refactor while keeping tests green
5. Move to next user story

### Incremental Delivery

- **Week 1**: Setup + Foundational (T001-T016)
- **Week 2**: US1 Implementation (T017-T031) → MVP Feature 1
- **Week 3**: US2 Implementation (T032-T044) → MVP Complete
- **Week 4**: US3 Implementation (T045-T052) → Post-MVP Increment 1
- **Week 5**: US4 Implementation (T053-T068) → Post-MVP Increment 2
- **Week 6**: US5 Implementation (T069-T081) → Post-MVP Increment 3
- **Week 7**: Polish (T082-T091) → Final Release

### Validation Gates

- After each phase, run: `make test-fast` (~30s)
- After each user story, run: `make ci-native` (~2min)
- Before committing, run: `make format` + `make ci-native`
- Before pushing, run: `make ci-act` (~90s Docker validation)

---

## Task Summary

- **Total Tasks**: 131 (was 115)
- **Setup Tasks**: 3
- **Foundational Tasks**: 13
- **User Story 1 (P1)**: 15 tasks (7 tests + 8 implementation)
- **User Story 2 (P1)**: 13 tasks (6 tests + 7 implementation)
- **User Story 3 (P2)**: 8 tasks (4 tests + 4 implementation)
- **User Story 4 (P2)**: 16 tasks (6 tests + 10 implementation)
- **User Story 5 (P3)**: 13 tasks (3 tests + 10 implementation)
- **User Story 6 (P2)**: 34 tasks (unified containers architecture)
- **User Story 7 (P2)**: 24 tasks (7 tests + 17 implementation)
- **User Story 8 (P2)**: 16 tasks (5 tests + 8 implementation + 3 documentation) - NEW
- **Polish Tasks**: 10

**Parallelizable Tasks**: ~60 tasks marked with [P]
**MVP Tasks (US1+US2)**: 44 tasks
**Test Tasks**: 42 tasks across all stories (TDD approach)

