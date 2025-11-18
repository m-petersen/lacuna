# Lacuna Development Guidelines

## Active Technologies

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
make test        # Run tests (~5 sec)
make test-fast   # Stop at first failure
make lint        # Check code quality
make format      # Auto-format code
```
**Use constantly during development.**

#### ⚡ Outer Loop (Before commit) - Native CI
```bash
make ci-native   # Full CI suite (~30 sec)
```
**Run before every commit.**

#### 🐳 Verification (Before push) - Docker
```bash
make ci-act      # Verify in Docker (~90 sec)
```
**Run only before pushing to verify Docker/OS compatibility.**

### When to Use What

| Scenario | Command | Speed |
|----------|---------|-------|
| Changed code | `make test` | ~5s |
| Before commit | `make ci-native` | ~30s |
| Before push | `make ci-act` | ~90s |
| CI failed | `sudo act -j test -v` | ~2m |

### Code Quality

Before committing:
1. `make test` - Tests pass
2. `make ci-native` - Full CI check
3. `make format` - Code formatted

### Version Management

Version automatically derived from git tags via setuptools-scm:
- Tagged: `v0.1.0` → `0.1.0`
- Dev: `0.1.0.devN+gHASH`

**Never edit version numbers manually.**

### Dependencies

- **Core**: nibabel, numpy, scipy, nilearn, pandas, templateflow
- **Optional**: `viz` (matplotlib), `bids` (pybids), `preprocess` (ANTsPy)
- **Dev**: pytest, pytest-xdist, ruff, black, mypy

ANTsPy is optional (only for native→MNI registration).

### Configuration Files

- `Makefile` - All commands (single source of truth)
- `.actrc` - act configuration
- `pyproject.toml` - Package metadata and dependencies
- `.github/workflows/` - CI workflows (call make commands)

See `DEVELOPMENT.md` for detailed workflow documentation.
