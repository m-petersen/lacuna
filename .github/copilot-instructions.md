# Lacuna Development Guidelines

## Active Technologies
- Python 3.10+ (project constraint from existing implementation) + nibabel, nilearn, numpy, scipy, pandas, templateflow, pytest, pytest-xdist, ruff, black, mypy (003-package-optimization)
- File-based (NIfTI images, HDF5 connectomes, .tck tractograms), no database (003-package-optimization)
- Python 3.10+ (existing project constraint) + nibabel, nilearn, numpy, scipy, pandas, templateflow, tqdm, joblib, h5py, nitransforms (005-package-consolidation)
- File-based (NIfTI images, HDF5 connectomes, .tck tractograms) - no database (005-package-consolidation)
- Python 3.10+ (project requirement from pyproject.toml) + pooch (download management, already a dependency), requests (HTTP, transitive via pooch), cloudscraper (Figshare Cloudflare bypass, NEW), tqdm (progress, already a dependency), h5py (HDF5, already a dependency), nibabel (tractogram conversion, already a dependency) (006-connectome-fetching)
- File-based (NIfTI → HDF5 batches, TRK → TCK), cached in `~/.cache/lacuna/` (006-connectome-fetching)
- Python 3.10+ (documentation tooling) + mkdocs, mkdocs-material, mkdocstrings[python], mkdocs-gen-files, mkdocs-literate-nav (008-mkdocs-documentation)
- Static files (Markdown in `docs/`, generated site in `site/`) (008-mkdocs-documentation)

- Python 3.10+ (existing project constraint)

## Project Structure

```text
src/lacuna/    # Main package
tests/         # Test suite
  contract/    # API contract tests
  integration/ # Integration tests
  unit/        # Unit tests
```

## Development Workflow

### Test-Driven Development (TDD)
Follow TDD: write tests before implementation. If bugs are found, create tests that cover the bug before fixing it.

### Three-Loop Testing Strategy

Use the right tool for the right speed:

#### 🔥 Inner Loop (Every 5 minutes) - Native
```bash
make test-fast   # Unit + contract tests (~30 sec)
make test-unit   # Unit tests only (~15 sec)
```
**Use constantly during development.**

#### ⚡ Outer Loop (Before commit) - Native CI
```bash
make ci-native   # Full CI suite (~2 min)
```
**Run before every commit.**

#### 🐳 Verification (Before push) - Docker
```bash
make ci-act      # Verify in Docker (~90 sec)
```
**Run only before pushing to verify Docker/OS compatibility.**

### Test Organization

Tests are categorized by speed and scope:
- **Unit tests** (`tests/unit/`): Fast (<1s), isolated components
- **Contract tests** (`tests/contract/`): Fast (1-5s), API validation
- **Integration tests** (`tests/integration/`): Slower (5-30s), multiple components

Use pytest markers for fine-grained control:
```python
@pytest.mark.fast                  # Quick tests (<1s)
@pytest.mark.slow                  # Slow tests (>5s)
@pytest.mark.integration           # Integration tests
@pytest.mark.contract              # Contract/API tests
@pytest.mark.requires_mrtrix       # Requires MRtrix3
@pytest.mark.requires_templateflow # Requires internet
```

**All test commands use pytest-xdist (`-n auto`) for parallel execution** (3-8x speedup).

See `docs/testing_strategy.md` for comprehensive testing guide.

### When to Use What

| Scenario | Command | Speed |
|----------|---------|-------|
| Changed code | `make test-fast` | ~30s |
| Before commit | `make ci-native` | ~2min |
| Before push | `make ci-act` | ~90s |
| CI failed | `sudo act -j test -v` | ~2min |
| Only unit tests | `make test-unit` | ~15s |
| Only contract tests | `make test-contract` | ~15s |
| Only integration tests | `make test-integration` | ~1min |

### Code Quality

Before committing:
1. `make test` - Tests pass
2. `make ci-native` - Full CI check
3. `make format` - Code formatted

### Commit Message Requirements

All commits must follow these requirements:

1. **Type must be clear**: Use conventional commit types (feat, fix, refactor, test, docs, chore, perf, style, ci, build)
2. **Subject must be imperative**: Use "add" not "added", "fix" not "fixed" (imperative mood)
3. **Subject must be under 50 characters**: Keep it concise
4. **Blank line between subject and body**: Required for multi-line commits
5. **Body must explain why**: Describe the motivation and context, not what changed (code shows what)

Examples:
```
feat: add binary mask validation to MaskData

Enforce binary masks (0/1 values only) to prevent invalid analysis
results. Users previously could pass continuous values, leading to
incorrect atlas aggregation and connectivity mapping.
```

```
refactor: rename LesionData to MaskData

Breaking change to reflect broader use cases beyond lesions.
Updated 200+ references across codebase. No backward compatibility
maintained per user approval.
```

### Version Management

Version automatically derived from git tags via setuptools-scm:
- Tagged: `v0.1.0` → `0.1.0`
- Dev: `0.1.0.devN+gHASH`

**Never edit version numbers manually.**

### Dependencies

- **Core**: nibabel, numpy, scipy, nilearn, pandas, templateflow
- **Optional**: `viz` (matplotlib)
- **Dev**: pytest, pytest-xdist, ruff, black, mypy



### Configuration Files

- `Makefile` - All commands (single source of truth)
- `.actrc` - act configuration
- `pyproject.toml` - Package metadata and dependencies
- `.github/workflows/` - CI workflows (call make commands)

See `DEVELOPMENT.md` for detailed workflow documentation.

## Recent Changes
- 008-mkdocs-documentation: Added Python 3.10+ (documentation tooling) + mkdocs, mkdocs-material, mkdocstrings[python], mkdocs-gen-files, mkdocs-literate-nav
- 006-connectome-fetching: Added Python 3.10+ (project requirement from pyproject.toml) + pooch (download management, already a dependency), requests (HTTP, transitive via pooch), cloudscraper (Figshare Cloudflare bypass, NEW), tqdm (progress, already a dependency), h5py (HDF5, already a dependency), nibabel (tractogram conversion, already a dependency)
- 005-package-consolidation: Added Python 3.10+ (existing project constraint) + nibabel, nilearn, numpy, scipy, pandas, templateflow, tqdm, joblib, h5py, nitransforms
