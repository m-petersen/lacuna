# Development Workflow with act

This project uses **nektos/act** to run GitHub Actions workflows locally, eliminating the need for Make or separate task runners. Git tags drive versioning via setuptools-scm.

## Philosophy: "Think Globally, Act Locally"

Define tasks once as GitHub Actions workflows → run them anywhere:
- ✅ Locally with `act` (instant feedback, no git push needed)
- ✅ In CI on GitHub (automatic on push/PR)
- ✅ Same Docker environment everywhere

## Quick Start

### 1. Install act

```bash
# macOS
brew install act

# Linux
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### 2. Install project

```bash
pip install -e ".[dev]"
```

### 3. Run a workflow locally

```bash
# List all available jobs
act -l

# Run tests
act -j test

# Run linting
act -j lint -W .github/workflows/pre-commit.yml
```

## Common Tasks

### Testing

```bash
# Full test suite with coverage (matches CI)
act -j test

# Fast tests without coverage
act -j test-fast -W .github/workflows/local-test-fast.yml

# Test specific Python version
act -j test --matrix python-version:3.11

# Test all Python versions (3.10, 3.11, 3.12)
act -j test  # Runs full matrix
```

### Code Quality

```bash
# Run all linting checks
act -j lint -W .github/workflows/pre-commit.yml

# Auto-format code (ruff + black)
act -j format -W .github/workflows/local-format.yml
```

### Cleanup

```bash
# Clean build artifacts and caches
act -j clean -W .github/workflows/local-clean.yml
```

### Building & Releasing

```bash
# Build package distribution
act -j build -W .github/workflows/release.yml

# Test full release workflow (requires git tag)
git tag v0.1.0
act -j build -W .github/workflows/release.yml -e <(echo '{"ref": "refs/tags/v0.1.0"}')
```

## Workflow Organization

### CI Workflows (`.github/workflows/`)

- **test.yml** - Full test suite, runs on push/PR
  - Tests Python 3.10, 3.11, 3.12
  - Coverage reporting to Codecov
  - Use: `act -j test`

- **pre-commit.yml** - Code quality checks
  - Ruff linting
  - Black formatting
  - Mypy type checking
  - Use: `act -j lint -W .github/workflows/pre-commit.yml`

- **release.yml** - Package building and publishing
  - Triggered by version tags (`v*.*.*`)
  - Builds wheel and sdist
  - Creates GitHub release
  - Publishes to PyPI (when configured)
  - Use: `act -j build -W .github/workflows/release.yml`

### Local Workflows (`.github/workflows/local-*.yml`)

- **local-test-fast.yml** - Quick tests without coverage
  - Use: `act -j test-fast -W .github/workflows/local-test-fast.yml`

- **local-format.yml** - Auto-format code
  - Use: `act -j format -W .github/workflows/local-format.yml`

- **local-clean.yml** - Clean build artifacts
  - Use: `act -j clean -W .github/workflows/local-clean.yml`

## Configuration

### .actrc

Project-wide act settings (already configured):
- Uses `catthehacker/ubuntu:act-latest` Docker image
- Enables container reuse for speed
- Offline mode support (caches everything)

### pyproject.toml

- **Version**: Dynamic, from git tags via `setuptools-scm`
- **Dependencies**: All in `[project.dependencies]`
- **Optional deps**: `viz`, `bids`, `preprocess` (ANTsPy), `dev`, `doc`

## Versioning Workflow

```bash
# 1. Make changes and test locally
act -j test

# 2. Commit changes
git add .
git commit -m "Add new feature"

# 3. Create version tag
git tag -a v0.2.0 -m "Release v0.2.0"

# 4. Push (triggers release workflow in CI)
git push origin main
git push origin v0.2.0
```

Version is automatically derived:
- `v0.2.0` tag → package version `0.2.0`
- Untagged commits → `0.2.0.dev5+g3a2b1c4`

## Advanced act Usage

### Secrets

```bash
# Interactive secret input
act -s MY_SECRET

# From environment
act -s MY_SECRET=value

# From file
echo "MY_SECRET=value" > .secrets
act --secret-file .secrets
```

### Events

```bash
# Trigger pull_request event
act pull_request

# Trigger workflow_dispatch with inputs
echo '{"inputs": {"name": "test"}}' > event.json
act workflow_dispatch -e event.json

# Trigger schedule event
act schedule
```

### Matrix Selection

```bash
# Run specific matrix combination
act -j test --matrix python-version:3.10

# Multiple matrix values
act -j test --matrix python-version:3.11 --matrix os:ubuntu-latest
```

### Debugging

```bash
# Verbose output
act -v -j test

# List workflows and jobs
act -l

# Dry run (show what would run)
act -n

# Use specific workflow file
act -W .github/workflows/test.yml
```

## Docker Alternative

If you prefer pure Docker without act:

```bash
# Build image
docker build -t lacuna-test .

# Run tests
docker run --rm -v $PWD:/workspace lacuna-test

# Or use docker-compose
docker-compose run test
docker-compose run lint
```

## Why Not Make?

From act's documentation: "With act, you can use the GitHub Actions defined in your .github/workflows/ to replace your Makefile!"

Benefits:
- ✅ No duplication - same workflow locally and in CI
- ✅ No shell script portability issues
- ✅ Docker-based, consistent environment
- ✅ Matrix builds work identically
- ✅ Native secret/variable support
- ✅ YAML > Makefile (better IDE support, validation)

## Troubleshooting

### act pulls Docker images every time

Already configured in `.actrc` with `--reuse` and offline mode. First run downloads images, subsequent runs use cache.

### Tests pass locally but fail in CI

```bash
# Run exact CI environment
act -j test --pull  # Force fresh Docker image
```

### Need to test without Docker

```bash
# Skip Docker, run on host (use carefully!)
act -P ubuntu-latest=-self-hosted -j test
```

### act command not found

```bash
# Verify installation
which act

# Or use direct path
~/.local/bin/act -l
```

## Dependencies

### Core (always required)
- nibabel, numpy, scipy, nilearn, pandas
- templateflow, nitransforms
- pooch, h5py, tqdm, joblib

### Optional Groups
```bash
pip install lacuna[viz]        # matplotlib
pip install lacuna[bids]       # pybids
pip install lacuna[preprocess] # antspyx (only for native→MNI registration)
pip install lacuna[dev]        # pytest, ruff, black, mypy
pip install lacuna[all]        # everything
```

**Note**: ANTsPy is optional - core functionality doesn't require it.
