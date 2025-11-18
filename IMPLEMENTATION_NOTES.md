# Implementation Summary: Git-Tag Versioning + act-based CI/CD

## What Was Implemented

### 1. Git-Tag Driven Versioning
- ✅ Added `setuptools-scm` to automatically derive version from git tags
- ✅ Updated `pyproject.toml` to use `dynamic = ["version"]`
- ✅ Modified `src/lacuna/__init__.py` to import version from auto-generated `_version.py`
- ✅ Added `_version.py` to `.gitignore` (auto-generated file)

**Result**: Version is now automatically derived from git tags:
- Tagged release: `v0.1.0` → package version `0.1.0`
- Development: `0.1.0.dev87+g2503ac7` (auto-generated)

### 2. ANTs Dependency Removed
- ✅ Moved `antspyx` to optional `[preprocess]` dependency group
- ✅ Added conditional import with `HAS_ANTS` flag in `spaces.py`
- ✅ Added helpful error message if ANTsPy functions are called without installation
- ✅ Updated `pytest-xdist` added for parallel test execution

**Result**: Core package no longer requires ANTsPy. Only needed for native→MNI registration.

### 3. act as Primary Task Runner (Replacing Make)
- ✅ Created `.actrc` configuration for local GitHub Actions execution
- ✅ Replaced Makefile/docker-compose with act workflows
- ✅ All tasks defined once in `.github/workflows/` - work locally AND in CI

**Philosophy**: "Think Globally, Act Locally" - write workflows once, run anywhere.

### 4. GitHub Actions Workflows

**CI Workflows** (auto-run on GitHub):
- `test.yml` - Full test matrix (Python 3.10, 3.11, 3.12) with coverage
- `pre-commit.yml` - Linting and formatting checks
- `release.yml` - Build and publish on version tags

**Local Workflows** (designed for act):
- `local-test-fast.yml` - Quick tests without coverage
- `local-format.yml` - Auto-format code with ruff + black
- `local-clean.yml` - Clean build artifacts and caches

### 5. Documentation
- ✅ Created `DEVELOPMENT.md` - Comprehensive dev workflow guide
- ✅ Updated `README.md` - Added contributor quick start
- ✅ Kept `Dockerfile` for pure Docker users (optional)

## How to Use

### Install act
```bash
# macOS
brew install act

# Linux
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### Common Commands
```bash
# List all workflows
act -l

# Run tests (matches CI exactly)
act -j test

# Run fast tests
act -j test-fast -W .github/workflows/local-test-fast.yml

# Run linting
act -j lint -W .github/workflows/pre-commit.yml

# Format code
act -j format -W .github/workflows/local-format.yml

# Clean artifacts
act -j clean -W .github/workflows/local-clean.yml

# Test specific Python version
act -j test --matrix python-version:3.11
```

### Create a Release
```bash
# 1. Test everything locally
act -j test
act -j lint -W .github/workflows/pre-commit.yml

# 2. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 3. GitHub Actions automatically:
#    - Builds package
#    - Creates GitHub release
#    - Publishes to PyPI (when configured)
```

## Why This Approach?

### Instead of Make
- ✅ Same workflow runs locally and in CI (no duplication)
- ✅ Docker-based ensures consistent environment
- ✅ Matrix builds work identically
- ✅ Better secret/variable management
- ✅ YAML > Makefile (IDE support, validation)

### Instead of Manual Versioning
- ✅ No more forgetting to bump version numbers
- ✅ Version always matches git tag
- ✅ Dev builds get automatic `.devN+gHASH` suffixes
- ✅ Single source of truth (git tags)

### Instead of Requiring ANTs
- ✅ Smaller dependency footprint
- ✅ Faster installation
- ✅ Only install what you need
- ✅ Clear separation: core vs optional features

## Configuration Files

```
.actrc                              # act configuration
.github/workflows/
  ├── test.yml                      # CI: Full test matrix
  ├── pre-commit.yml                # CI: Linting
  ├── release.yml                   # CI: Build & publish
  ├── local-test-fast.yml          # Local: Quick tests
  ├── local-format.yml             # Local: Auto-format
  └── local-clean.yml              # Local: Clean artifacts
pyproject.toml                      # Package config + dependencies
Dockerfile                          # Optional: pure Docker workflow
DEVELOPMENT.md                      # Developer guide
```

## Benefits

1. **Fast Local Feedback**: Test CI changes without pushing
2. **No Duplication**: Write once, run everywhere
3. **Consistent Environment**: Docker ensures parity
4. **Automatic Versioning**: Git tags drive releases
5. **Lighter Dependencies**: ANTsPy optional
6. **Better DX**: Modern tooling (act, ruff, black, pytest-xdist)

## Migration Notes

### Removed Files
- `Makefile` - Replaced by act workflows
- `docker-compose.yml` - Replaced by act workflows
- `CI_CD_README.md` - Merged into DEVELOPMENT.md

### New Files
- `.actrc` - act configuration
- `DEVELOPMENT.md` - Developer guide
- `.github/workflows/*.yml` - All workflows
- `Dockerfile` - Optional pure Docker workflow

### Modified Files
- `pyproject.toml` - setuptools-scm, optional ANTsPy
- `src/lacuna/__init__.py` - Dynamic version import
- `src/lacuna/preprocess/spaces.py` - Conditional ANTsPy
- `.gitignore` - Ignore auto-generated _version.py
- `README.md` - Updated contributor section

## Next Steps

1. Install act: `brew install act` (macOS) or use curl (Linux)
2. Test locally: `act -j test`
3. Read `DEVELOPMENT.md` for detailed workflow
4. When ready to release: `git tag v0.2.0 && git push origin v0.2.0`
