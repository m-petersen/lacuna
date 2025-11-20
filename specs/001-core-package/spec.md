# Feature Specification: Lacuna Core Package

**Feature Branch**: `001-core-package`  
**Created**: 2025-10-27  
**Status**: Draft  
**Input**: User description: "build the lesion decoding toolkit as described in the notes"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load and Validate Lesion Data (Priority: P1)

A neuroimaging researcher has lesion mask data (NIfTI files) in MNI152 for multiple subjects and needs to load this data into a standardized format that can be used for subsequent analysis. The researcher may have data organized in BIDS format or as individual NIfTI files.

**Why this priority**: This is the foundational capability—all analysis depends on the ability to load and represent lesion data correctly. Without this, no other functionality is possible.

**Independent Test**: Can be fully tested by loading sample NIfTI files and BIDS datasets, then verifying that data objects are created with correct image data, affine matrices, and metadata.

**Acceptance Scenarios**:

1. **Given** a valid NIfTI lesion mask file, **When** the researcher loads it using the I/O module, **Then** a LesionData object is created with the image data, affine matrix, and any provided metadata
2. **Given** a BIDS-compliant dataset directory, **When** the researcher loads the dataset, **Then** LesionData objects are created for all subjects with lesion masks, with metadata automatically parsed from BIDS structure
3. **Given** a LesionData object, **When** the researcher inspects it, **Then** they can access the lesion image, anatomical image (if provided), spatial metadata, and subject identifiers
4. **Given** invalid or corrupted NIfTI files, **When** the researcher attempts to load them, **Then** clear error messages are provided explaining what is wrong

---

### User Story 2 - Save and Export Analysis Results (Priority: P2)

After completing analyses, a researcher needs to save the results in standard formats (NIfTI for images, CSV/TSV for tables) that can be used in publications or further statistical analyses.

**Why this priority**: Essential for completing the analysis workflow, but less critical than the analysis itself since researchers could manually export data if needed.

**Independent Test**: Can be tested by running analyses, saving results, and verifying that output files are correctly formatted and contain the expected data.

**Acceptance Scenarios**:

1. **Given** a LesionData object with analysis results, **When** the researcher saves it, **Then** image data is written as NIfTI files and tabular results as CSV files in a structured directory
2. **Given** BIDS-derivative output format is requested, **When** results are saved, **Then** they follow BIDS-derivative naming conventions and include appropriate metadata files
3. **Given** a saved analysis, **When** the researcher reloads it later, **Then** all results and provenance information are preserved

--- 

### User Story 3 - Lesion Network Mapping Analysis (Priority: P3)

A researcher wants to understand the functional network impacts of a lesion by mapping it to a normative brain connectome. This involves overlaying the lesion mask with connectivity data to identify disconnected or affected brain networks.

**Why this priority**: This is a core analysis capability that provides high scientific value. The initial implementation will focus on lesions already in standard space (MNI152), making this a high-priority feature that can be implemented early in the development cycle.

**Independent Test**: Can be tested by taking lesion data in standard space, applying it to a sample connectome, and verifying that network disruption metrics are calculated and stored in the results attribute.

**Acceptance Scenarios**:

1. **Given** a LesionData object in MNI152 space and a connectome atlas, **When** the researcher performs lesion network mapping, **Then** network disruption metrics are calculated and added to the LesionData results
2. **Given** analysis results in a LesionData object, **When** the researcher accesses them, **Then** they can retrieve network-level statistics and affected regions
3. **Given** multiple subjects with lesion data in MNI152 space, **When** batch analysis is performed, **Then** results are computed for all subjects and can be exported in a structured format

---

### User Story 4 - Spatial Preprocess and Normalization (Priority: P4)

A researcher needs to transform lesion masks from native subject space to a standard template space (e.g., MNI152) to enable comparisons across subjects or mapping to normative atlases. This involves spatial normalization, resampling, and coordinate system transformations.

**Why this priority**: While spatial normalization is useful for some workflows, the initial development will focus on lesions already in standard space. This allows core functionality (I/O, data model, and lesion network mapping) to be implemented and validated first. Native-to-MNI transformation will be added in a later phase once the core pipeline is stable and validated.

**Independent Test**: Can be tested by taking lesion data in native space, applying spatial transformations, and verifying that the output is correctly aligned with a standard template (verified through visual inspection and affine matrix validation).

**Acceptance Scenarios**:

1. **Given** a LesionData object in native space, **When** the researcher applies spatial normalization, **Then** a new LesionData object is returned with the lesion transformed to standard space with updated affine matrix
2. **Given** a LesionData object, **When** the researcher resamples it to a different resolution, **Then** a new LesionData object is returned with the correctly resampled image and updated header
3. **Given** two LesionData objects in different spaces, **When** the researcher checks spatial compatibility, **Then** the system correctly identifies whether they share the same coordinate space
4. **Given** a spatial transformation operation, **When** it is applied, **Then** the provenance is recorded in the LesionData object's history

---

### User Story 5 - Visualize Lesions and Results (Priority: P5)

A researcher wants to create publication-quality visualizations of lesion masks overlaid on anatomical images, and to visualize analysis results like network disruption maps.

**Why this priority**: Important for interpretation and communication but optional if researchers can use other visualization tools with exported data.

**Independent Test**: Can be tested by creating various plots and verifying they correctly display the data with appropriate labels and color schemes.

**Acceptance Scenarios**:

1. **Given** a LesionData object, **When** the researcher creates a visualization, **Then** the lesion is overlaid on an anatomical template with appropriate color mapping
2. **Given** analysis results, **When** visualizations are generated, **Then** plots show network disruption patterns or other analysis outcomes
3. **Given** visualization functions, **When** optional visualization dependencies are not installed, **Then** clear error messages guide users to install the `viz` extra

---

### Edge Cases

- **What happens when a lesion mask file has an unusual voxel orientation or non-standard affine matrix?**
  - System uses `ensure_ras_plus()` to reorient to RAS+ canonical orientation per Principle 7
  - Non-standard affines are validated and corrected using nilearn's canonical orientation utilities
  - Provenance records the reorientation operation

- **How does the system handle lesion masks with floating-point values (non-binary) vs. binary masks?**
  - Both are supported: floating-point values treated as probability maps
  - `binarize()` function (Phase 6, T084) provides explicit thresholding (default: 0.5)
  - Binary operations (volume calculation) automatically threshold at >0

- **What happens when BIDS metadata is incomplete or malformed?**
  - System validates required BIDS fields (subject_id) and raises `BIDSValidationError` with specific missing fields
  - Optional fields are logged as warnings but don't block loading
  - Graceful degradation: partial metadata still creates valid LesionData objects (T030)

- **How does the system handle very large lesions that exceed anatomical boundaries?**
  - No automatic clipping; analysis proceeds with full lesion extent
  - Warnings logged if lesion extends beyond template mask boundaries
  - Researchers responsible for preprocess if clipping needed

- **What happens when attempting spatial operations on data that is already in the target space?**
  - Coordinate space check (T028.1) detects matching spaces
  - No-op: returns shallow copy of input with provenance note "already in target space"
  - Prevents unnecessary computation and transformation artifacts

- **How does the system handle multi-session data where the same subject has lesion masks at different timepoints?**
  - BIDS loader (T027-T028) parses session information from filenames
  - Each session creates separate LesionData object with session metadata
  - Batch processing utilities handle session grouping for longitudinal analysis (v2.0 feature)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a standardized LesionData class that encapsulates lesion image data, affine matrix, metadata, and analysis results
- **FR-002**: System MUST load NIfTI format lesion masks and create LesionData objects with all header information preserved
- **FR-003**: System MUST support BIDS-compliant dataset organization and automatically parse subject/session metadata
- **FR-004**: System MUST validate spatial consistency of image data and affine matrices
- **FR-005**: System MUST provide spatial transformation functions that return new LesionData objects (preserving immutability)
- **FR-006**: System MUST track provenance by recording the sequence of transformations applied to each LesionData object
- **FR-007**: System MUST support spatial normalization to standard template spaces (MNI152)
- **FR-008**: System MUST provide coordinate space validation utilities to check if two images are in the same space
- **FR-009**: System MUST implement lesion network mapping functionality that computes network disruption metrics
- **FR-010**: System MUST store all analysis results in the LesionData object's results attribute without modifying other attributes
- **FR-011**: System MUST support batch processing of multiple subjects
- **FR-012**: System MUST save results in BIDS-derivative format when requested
- **FR-013**: System MUST provide visualization functions as an optional feature (requiring extra dependencies)
- **FR-014**: System MUST follow the src/ layout for package structure
- **FR-015**: System MUST define optional dependency groups (viz, dev, doc) in pyproject.toml
- **FR-016**: System MUST use established libraries (nibabel, nilearn, pybids) for critical operations
- **FR-017**: System MUST provide clear error messages for invalid input data or incompatible operations
- **FR-018**: System MUST enforce module boundaries (core, io, preprocess, analysis, modeling, reporting) with no circular dependencies
- **FR-019**: System MUST ensure that new analysis modules can be added by depending only on the LesionData abstraction
- **FR-020**: System MUST provide comprehensive test coverage using pytest

### Key Entities

- **LesionData**: The central data container representing a single subject's lesion analysis. Contains lesion image (NIfTI), optional anatomical image, affine matrix, metadata dictionary (subject ID, session, BIDS info), provenance history, and results dictionary.

- **Subject Metadata**: Information identifying and describing a research participant, including subject ID, session ID, clinical/demographic variables, and BIDS-compliant attributes.

- **Spatial Transformation**: A record of a spatial operation applied to image data, including the transformation type (normalization, resampling, coregistration), parameters, target space, and timestamp.

- **Analysis Results**: Output from an analysis module, structured as key-value pairs or nested dictionaries containing metrics, maps, or statistical outcomes.

- **Connectome Reference**: A normative brain connectivity dataset used for lesion network mapping, defining relationships between brain regions.

### Batch Processing Architecture

The system implements a flexible batch processing architecture that optimizes execution strategy based on analysis characteristics and computational resources.

**Batch Strategies:**
- **Parallel**: For independent per-subject analyses (RegionalDamage, AtlasAggregation). Uses multiprocessing to parallelize across subjects, providing 4-8x speedup on multi-core systems proportional to available CPU cores.
- **Vectorized**: For matrix-based analyses (FunctionalNetworkMapping, StructuralNetworkMapping). Stacks lesion data into matrices and performs batch operations via optimized BLAS libraries, providing 10-50x speedup for network analyses.
- **Streaming**: For memory-intensive operations with large connectomes. Processes subjects sequentially with immediate saving to disk, managing memory constraints while maintaining throughput.

**Strategy Selection**: The `batch_process()` function automatically selects the optimal strategy based on:
1. Analysis type (declared via `batch_strategy` class attribute)
2. Number of subjects and estimated memory requirements
3. Available system resources (CPU cores, RAM)

**Analysis Declaration**: Each analysis class declares its preferred batch strategy. Custom batch implementations can override `run_batch()` for specialized optimizations beyond the default strategy.

**Design Rationale**: Batch processing is implemented AFTER save/export functionality (Phase 4) to ensure proper validation of single-subject pipelines and establish a foundation for handling batch results. See `batch-architecture.md` for detailed implementation specifications.

### Data Management Architecture

The system implements a tiered data management strategy to handle large reference datasets (atlases, connectomes, templates) while providing flexibility for different computational environments.

#### Storage Strategy

**Reference Data Types:**
- **Atlases**: Pre-registered brain parcellation schemes (Harvard-Oxford, AAL, Schaefer, etc.). Small files (~10-50MB each) suitable for automatic downloading.
- **Templates**: Standard space reference images (MNI152). Small files (~10MB) suitable for automatic downloading.
- **Connectomes**: Large normative connectivity datasets (50-100GB). User-managed due to licensing and size constraints.

**Storage Locations:**
- **Default cache**: `~/.cache/lacuna/` (XDG Base Directory compliant)
  - Contains auto-downloaded atlases and templates
  - Can be safely deleted and regenerated
  - Respects `XDG_CACHE_HOME` environment variable
- **User-specified data**: Configurable via `LACUNA_DATA_DIR` environment variable
  - Allows placement on fast storage (NVMe) or network drives
  - Persists across cache clearing operations
- **Connectome storage**: User-managed, specified via `LACUNA_CONNECTOME_DIR` environment variable
  - Users download and convert connectomes themselves
  - No automatic downloading due to licensing restrictions

#### Atlas Management

**Atlas Registry System:**
The system maintains a centralized registry (`ATLAS_REGISTRY`) that provides access to bundled atlases and supports user registration of custom atlases. All atlases are accessed through this registry, providing consistent metadata and space information.

**Bundled Atlases:**
- Schaefer2018 (100, 200, 400, 1000 parcels) - MNI152NLin6Asym, 1mm
- TianSubcortex (S1, S2, S3 scales) - MNI152NLin6Asym, 1mm  
- HCP1065 white matter tracts - MNI152NLin2009aAsym, 1mm

**Atlas Access Workflow:**
1. **Use bundled atlas**: `AtlasAggregation(atlas_names=["Schaefer2018_100Parcels7Networks"])`
2. **Register custom atlas**: `register_atlas_from_files(name, image_path, labels_path, space, resolution)`
3. **Bulk register directory**: `register_atlases_from_directory("/path/to/atlases")`
4. **List available**: `list_atlases(space="MNI152NLin6Asym", resolution=1)`

**Atlas Registration Requirements:**
Users must explicitly register custom atlases before use. The registry enforces:
- Unique atlas names (prevents accidental overwrite)
- Space and resolution metadata (enables automatic space handling)
- Label file pairing (ensures ROI names are available)
- BIDS filename parsing or header-based space detection

**Atlas file format:**
- Image file: `.nii.gz` with integer labels (3D) or probabilities (4D)
- Label file: `_labels.txt` with format `"1 RegionName"` or `"RegionName"` per line
- Paired files must share base name (e.g., `atlas.nii.gz` + `atlas_labels.txt`)
- BIDS naming preferred: `tpl-{SPACE}_res-{RES}_atlas-{NAME}_dseg.nii.gz`

#### Connectome Management

**User-managed workflow:**
Due to licensing restrictions, connectomes are not distributed with the package. Users must:
1. Obtain connectome data from original sources (GSP1000, HCP, etc.)
2. Convert to LDK-compatible format using conversion utilities
3. Specify location via environment variable or direct path

**Conversion utilities** (in `lacuna.io.convert`):
- `gsp1000_to_ldk()`: Converts GSP1000 batched fMRI data to HDF5 format
- `tractogram_to_ldk()`: Converts MRtrix3 .tck files to LDK format
- CLI tool: `lacuna convert gsp1000 /path/to/raw /output/path`

**Expected HDF5 structure** (functional connectomes):
```
connectome.h5
├── timeseries (subjects, timepoints, voxels)
├── mask_indices (voxel_count, 3)  # MNI152 coordinates
└── attributes:
    ├── mask_shape: [91, 109, 91]
    ├── affine: 4x4 matrix
    └── description: "GSP1000 functional connectome"
```

**Connectome resolution order:**
1. If provided as Path → use directly
2. If `LACUNA_CONNECTOME_DIR` set → look for named file
3. Else → raise `ConnectomeNotFoundError` with setup instructions

#### Implementation Details

**Technology stack:**
- **Pooch**: Download management, caching, checksums (for atlases/templates)
- **XDG Base Directory**: Standard cache location (`~/.cache/lacuna/`)
- **Environment variables**: User configuration without code changes
- **HDF5**: Efficient storage for large connectome matrices

**Environment variables:**
- `LACUNA_DATA_DIR`: Override default cache directory for all reference data
- `LACUNA_ATLAS_DIR`: Additional atlas search directory (scanned for custom atlases)
- `LACUNA_CONNECTOME_DIR`: Directory containing user-prepared connectomes
- `XDG_CACHE_HOME`: System-wide XDG cache location (respected by default)

**Functional requirements:**
- **FR-021**: System MUST maintain a centralized ATLAS_REGISTRY with bundled atlases including space metadata
- **FR-022**: System MUST provide functions to list, register, and unregister atlases from the registry
- **FR-023**: System MUST support bulk registration of atlases from a directory with automatic space detection
- **FR-024**: System MUST enforce unique atlas names and prevent accidental overwrites
- **FR-025**: System MUST respect XDG Base Directory specification for cache location
- **FR-026**: System MUST support user-specified data directories via environment variables
- **FR-027**: System MUST provide conversion utilities for preparing connectome data
- **FR-028**: System MUST raise clear errors when required data is missing, with setup instructions
- **FR-029**: System MUST load atlas data with both image and label information from registry
- **FR-030**: System MUST detect atlas space from BIDS filenames or header inspection

**Success criteria:**
- **SC-011**: Pre-registered atlas downloads complete in under 30 seconds on standard broadband
- **SC-012**: Cached atlas access completes in under 100ms
- **SC-013**: Users can specify custom data locations without modifying code
- **SC-014**: Connectome conversion utilities process 50GB datasets in under 10 minutes

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can load a single-subject lesion mask from a NIfTI file and create a valid LesionData object in under 5 seconds
- **SC-002**: Researchers can load a BIDS dataset containing 50 subjects with lesion masks in under 2 minutes
- **SC-003**: Spatial normalization operations complete in under 30 seconds per subject on standard hardware (4-core CPU, 16GB RAM, SSD storage)
- **SC-004**: The package correctly handles 95% of BIDS-compliant NIfTI files without errors
- **SC-005**: All spatial transformations preserve data integrity (no voxel data corruption) as verified by checksums
- **SC-006**: Researchers can add a new analysis module without modifying existing code in core, io, or preprocess modules
- **SC-007**: The test suite achieves at least 85% code coverage
- **SC-008**: Researchers can complete a full pipeline (load, preprocess, analyze, save) for a single subject with fewer than 10 lines of code
- **SC-009**: Error messages for common mistakes (wrong file path, incompatible spaces) clearly explain the problem and suggest solutions
- **SC-010**: Package installation (core only, no extras) completes in under 2 minutes and requires fewer than 10 dependencies

### Assumptions

- Lesion masks are provided as binary or continuous-valued 3D NIfTI images
- Standard template space defaults to MNI152 unless specified otherwise
- Researchers have basic familiarity with Python and neuroimaging concepts
- BIDS datasets follow version 1.6+ of the BIDS specification
- Computational environment has at least 8GB RAM for typical datasets
- Default coordinate system validation uses affine matrix comparison with tolerance of 1e-3
