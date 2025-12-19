# Manual Testing Guide

This guide covers CLI and Docker testing procedures that cannot be fully automated.

## Prerequisites

- Docker installed and running
- Python 3.10+ with lacuna installed in development mode
- Sample NIfTI lesion mask for testing

## Docker Testing

### 1. Build the Docker Image

```bash
# Build the production image
docker build -t lacuna:latest .

# Verify the build
docker images | grep lacuna
```

Expected output:
```
lacuna    latest    <hash>    <date>    <size>
```

### 2. Test Container Startup

```bash
# Run container and check Python/lacuna are available
docker run --rm lacuna:latest python -c "import lacuna; print(lacuna.__version__)"
```

Expected: Version string like `0.1.0.dev42+g1234567`

### 3. Test MRtrix3 Installation

```bash
# Verify MRtrix3 binaries are available
docker run --rm lacuna:latest which tckgen
docker run --rm lacuna:latest tckinfo --version
```

Expected: Path to `tckgen` and MRtrix3 version info

### 4. Run Tests Inside Container

```bash
# Run the full test suite in Docker
docker run --rm lacuna:latest pytest tests/ -v

# Run only unit tests (faster)
docker run --rm lacuna:latest pytest tests/unit/ -v -n auto
```

### 5. Test with Real Data

Mount your data directory and run an analysis:

```bash
# Mount local data and run analysis
docker run --rm \
  -v /path/to/data:/data:ro \
  -v /path/to/output:/output \
  lacuna:latest python -c "
from lacuna import SubjectData, analyze
import nibabel as nib

# Load lesion
img = nib.load('/data/lesion.nii.gz')
subject = SubjectData(
    mask_img=img,
    space='MNI152NLin6Asym',
    resolution=2.0
)

# Run basic analysis
result = analyze(subject)
print('Analysis completed:', list(result.results.keys()))
"
```

## CLI Testing

### 1. Test Package Installation

```bash
# Verify lacuna is installed
python -c "import lacuna; print(lacuna.__version__)"

# Test imports
python -c "
from lacuna import SubjectData, Pipeline, analyze, batch_process
from lacuna.analysis import list_analyses, get_analysis
print('All imports successful')
"
```

### 2. Test Analysis Discovery

```bash
python -c "
from lacuna.analysis import list_analyses, get_analysis

# List available analyses
print('Available analyses:')
for name, cls in list_analyses():
    print(f'  - {name}')

# Get specific analysis
RegionalDamage = get_analysis('RegionalDamage')
print(f'\\nRegionalDamage class: {RegionalDamage}')
"
```

### 3. Test Pipeline API

```bash
python -c "
from lacuna import Pipeline
from lacuna.analysis import RegionalDamage

# Create pipeline
pipeline = Pipeline(name='Test Pipeline')
pipeline.add(RegionalDamage(parcel_names=['Schaefer2018_100Parcels7Networks']))

print(pipeline.describe())
"
```

### 4. Test analyze() Function

```bash
python -c "
from lacuna import analyze
import nibabel as nib
import numpy as np

# Create synthetic lesion for testing
data = np.zeros((91, 109, 91), dtype=np.uint8)
data[40:50, 50:60, 40:50] = 1
affine = np.eye(4) * 2
affine[3, 3] = 1
img = nib.Nifti1Image(data, affine)

from lacuna import SubjectData
subject = SubjectData(
    mask_img=img,
    space='MNI152NLin6Asym',
    resolution=2.0
)

# Test basic analyze
result = analyze(subject)
print('Basic analyze() works:', 'RegionalDamage' in result.results)

# Test with parcel_atlases
result2 = analyze(
    subject,
    parcel_atlases=['Schaefer2018_100Parcels7Networks'],
)
print('analyze() with parcel_atlases works:', bool(result2.results.get('RegionalDamage')))
"
```

## CI Workflow Testing

### 1. Local CI (Native)

Run the full CI suite locally without Docker:

```bash
make ci-native
```

This runs:
- Linting (ruff)
- Formatting check (black)
- Type checking (mypy) - may have pre-existing errors
- Tests with coverage

### 2. Docker CI (act)

Verify the GitHub Actions workflow works in Docker:

```bash
# Install act if not present
# See: https://github.com/nektos/act

# Run CI in Docker
make ci-act

# Or with verbose output
sudo act -j test -v
```

### 3. Manual GitHub Actions Verification

Push changes to a branch and verify:

1. Go to GitHub repository → Actions tab
2. Find the workflow run for your push
3. Verify all jobs pass:
   - `lint` - ruff and black
   - `test` - pytest with coverage

## Troubleshooting

### Docker Issues

**Container won't start:**
```bash
# Check Docker daemon is running
docker info

# Check for port conflicts
docker ps -a
```

**MRtrix3 not found:**
```bash
# Rebuild image to ensure MRtrix3 is installed
docker build --no-cache -t lacuna:latest .
```

### Import Errors

**Module not found:**
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Or in Docker
docker run --rm lacuna:latest pip install -e ".[dev]"
```

### Test Failures

**Tests pass locally but fail in Docker:**
- Check Python version matches (3.10+)
- Verify all dependencies are in pyproject.toml
- Check for OS-specific path issues (Windows vs Linux)

## Test Checklist

Before pushing changes, verify:

- [ ] `make test-fast` passes (~30s)
- [ ] `make ci-native` passes (~2min)  
- [ ] `make ci-act` passes in Docker (~90s)
- [ ] Docker build succeeds: `docker build -t lacuna:latest .`
- [ ] Container starts: `docker run --rm lacuna:latest python -c "import lacuna"`

## See Also

- [Testing Strategy](../testing_strategy.md) - Comprehensive testing documentation
- [DEVELOPMENT.md](../../DEVELOPMENT.md) - Development workflow
- [Dockerfile](../../Dockerfile) - Docker build configuration
