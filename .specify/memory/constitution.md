<!--
SYNC IMPACT REPORT
- Version change: 1.0.0 -> 1.1.0
- Modified principles:
	- Enhanced Principle 1: Added type hints requirement for public API surfaces
	- Enhanced Principle 3: Added BIDS compliance and immutability requirements
	- Added Principle 5: Data Immutability & Pipeline Traceability (NEW)
	- Added Principle 6: Scientific Standards Compliance (NEW)
	- Added Principle 7: Spatial Correctness (NEW)
- Added sections:
	- Architectural Commitments: Enhanced with module boundary rules and extensibility
- Removed sections:
	- None
- Templates updated:
	- ⚠ pending: .specify/templates/plan-template.md (add immutability/provenance checks)
	- ⚠ pending: .specify/templates/spec-template.md (add BIDS/spatial requirements)
	- ⚠ pending: .specify/templates/tasks-template.md (add provenance tracking tasks)
- Follow-up TODOs:
	- TODO(RATIFICATION_DATE): Set the official ratification date upon team review.
-->

# Lesion Decoding Toolkit (ldk) Constitution

## Core Principles

These principles are the non-negotiable pillars of the project. All code, documentation,
and community interactions MUST adhere to them.

### 1. Pragmatic & Pythonic Code Quality

All code MUST be clear, readable, and maintainable. The project adopts a pragmatic
hybrid of Object-Oriented and Functional paradigms: stateful "nouns" (e.g., `LesionData`)
MUST be modeled as classes; stateless, transformational operations MUST be implemented
as functions. Code MUST follow PEP 8 conventions (automated via linters like `ruff`),
and contributors MUST provide type hints where they materially improve readability
or correctness (e.g., public API surfaces).

Rationale: A consistent, multi-paradigm style improves developer productivity,
reduces bugs, and makes the codebase accessible to new contributors while retaining
scientific rigor.

### 2. Test-Driven Development (TDD) & Reproducibility

All new functionality, including bug fixes and refactors that change behavior, MUST
be introduced via Test-Driven Development. Contributors MUST write failing tests
that express the desired behaviour before implementing code. Every PR MUST include
tests that validate the change. The test suite MUST be run by CI and pass before
merging. The project uses `pytest` as the canonical test runner and encourages
fast unit tests, supplemented by integration and regression tests.

Rationale: Scientific software carries high correctness requirements. TDD enforces
clarity of intent, guards against regressions, and documents expected behaviour
programmatically.

### 3. Consistent and Predictable User Experience (API Contract)

The `ldk.core.LesionData` object is the canonical API contract for pipeline stages.
All publicly exported functions and classes MUST document their inputs, outputs,
side-effects, and required metadata in docstrings (preferably using NumPy or Google
docstring style). The project MUST maintain a well-structured `pyproject.toml` with
optional extras (`viz`, `dev`, `doc`) to avoid dependency bloat for the core
installation. The I/O subsystem MUST support BIDS-compliant dataset organization and
provide BIDS-aware loading functions. Backwards-compatible API additions SHOULD be
preferred over breaking changes; breaking changes MUST follow the governance process
below.

Rationale: Predictable APIs reduce user errors, accelerate onboarding, and enable
interchangeable analysis modules that rely on a stable data contract. BIDS compliance
ensures interoperability with the broader neuroimaging ecosystem.

### 4. Performance by Design

Performance considerations MUST be integrated into design decisions. Implementations
SHOULD favor clear algorithms that use optimized numerical backends (NumPy, SciPy,
Numba, or libraries like nilearn) where appropriate. Performance-critical code that
adds complexity MUST be justified with benchmarks and included tests that guard
against regressions in speed or memory usage.

Rationale: The toolkit targets real-world research workflows and datasets; early
attention to performance prevents technical debt that would hinder adoption.

### 5. Data Immutability & Pipeline Traceability

Pipeline functions MUST favor immutability: processing functions SHOULD return new
`LesionData` objects rather than modifying existing ones in-place (exceptions
allowed for performance-critical operations with explicit documentation). Every
`LesionData` object MUST track its provenance: the sequence of transformations
applied (function names, parameters, timestamps). This processing history MUST be
accessible via a standardized interface and SHOULD be preserved when objects are
serialized.

Rationale: Immutability makes data flow explicit, prevents unintended side effects,
and simplifies debugging. Provenance tracking is essential for scientific
reproducibility and allows researchers to audit and reconstruct analysis workflows.

### 6. Scientific Standards Compliance (BIDS & Neuroimaging)

The I/O subsystem MUST support BIDS-compliant dataset structures as a first-class
input format. Loading functions MUST parse BIDS metadata automatically and populate
`LesionData` objects with subject IDs, session information, and relevant sidecar
metadata. Output data SHOULD be saved in BIDS-derivative format when applicable.
The package MUST provide clear documentation on BIDS expectations and examples of
compliant datasets.

Rationale: BIDS is the community standard for neuroimaging data organization.
First-class BIDS support ensures interoperability with other tools, reduces user
friction, and promotes adoption of best practices.

### 7. Spatial Correctness & Coordinate System Management

All spatial operations (registration, resampling, coordinate transformations) MUST
be implemented using well-validated libraries (primarily `nilearn`, with `nibabel`
for low-level operations). The `LesionData` object MUST always store images in a
clearly documented coordinate space (native or standard template) with the correct
affine matrix. Functions that perform spatial transformations MUST return new
objects with updated affines and MUST document their coordinate space assumptions.
The toolkit MUST provide utilities to validate spatial consistency (e.g., checking
if two images share the same space).

Rationale: Coordinate system errors are subtle, hard to detect, and can invalidate
scientific results. Mandating validated libraries and explicit coordinate tracking
prevents a critical class of bugs in neuroimaging analysis.

## Additional Constraints

- Source layout: the package MUST use the `src/ldk` layout to guarantee test
	integrity and avoid accidental local imports during testing.
- Packaging: `pyproject.toml` is the authoritative project configuration and MUST
	include extras for `viz`, `dev`, and `doc`.
- Documentation: All public APIs and high-level workflows MUST have example usage
	in `docs/` or `examples/` and a quickstart demonstrating a minimal end-to-end
	pipeline.

## Architectural Commitments

The project adheres to specific architectural patterns to ensure modularity,
extensibility, and maintainability.

### Module Boundaries & Separation of Concerns

- **`ldk.core`**: Defines the `LesionData` class and core abstractions. MUST NOT
	depend on analysis-specific logic from other subpackages.
- **`ldk.io`**: Handles all file I/O (NIfTI, BIDS). MUST return fully-formed
	`LesionData` objects; analysis modules MUST NOT perform direct file operations.
- **`ldk.preprocess`**: Preprocessing and spatial operations. Functions MUST accept
	and return `LesionData` objects.
- **`ldk.analysis`**: Domain-specific analyses (lesion network mapping,
	transcriptomics, etc.). Each module MUST be independently usable and MUST
	append results to `LesionData.results` without side effects on other modules.
- **`ldk.modeling`** and **`ldk.reporting`**: Post-analysis tasks (modeling, reporting). MUST depend
	only on the structure of `LesionData.results`, not on internal analysis
	implementation details.
- **`ldk.utils`** and **`ldk.viz`**: Auxiliary functionality. MUST be stateless
	and have no circular dependencies with core modules.

### Extensibility Requirements (Dependency Inversion)

New analysis modules MUST depend only on the `LesionData` abstraction, not on
concrete I/O implementations or other analysis modules. This enables "plug-and-play"
extensibility: contributors can add new analyses by writing functions that conform
to the `LesionData` contract without modifying existing pipeline code. Breaking
this principle (e.g., tight coupling between analysis modules) requires explicit
justification and maintainer approval.

## Development Workflow & Quality Gates

- Branch policy: feature work occurs on feature branches; PRs target `main` (or
	the repository's default branch) and MUST reference a related issue or spec.
- Code review: At least one approving review from a maintainer is required before
	merging. Reviews MUST verify style, tests, and that the change adheres to this
	constitution.
- CI gates: Every PR MUST pass linting, unit tests, and a minimal integration
	smoke test. Release candidates MUST pass the full test matrix.

## Governance

- Amendment process: Constitutional changes require a PR that clearly documents the
	rationale, a Sync Impact Report, and approval by at least one core maintainer.
- Versioning: This constitution follows Semantic Versioning for governance text:
	- MAJOR: Backward-incompatible changes to principles or governance.
	- MINOR: Additions of new principles or material expansions.
	- PATCH: Clarifications, typo-fixes, and non-semantic refinements.
- Compliance: Automated checks (linters, CI, tests) and peer review enforce
	adherence. Significant deviations MUST be documented with a justification in the
	related PR.

**Version**: 1.1.0 | **Ratified**: TODO(RATIFICATION_DATE) | **Last Amended**: 2025-10-27

