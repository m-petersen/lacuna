# Development Workflow with act

This project uses the **Task Runner Pattern**: Makefile is the single source of truth for all commands. Git tags drive versioning via setuptools-scm.

## Philosophy: Three-Loop Testing Strategy

**Don't use act for every change!** Use the right tool for the right speed:

1. **🔥 Inner Loop** (Every 5 mins): Native commands → Instant feedback
2. **⚡ Outer Loop** (Before commit): Full CI natively → Catch all issues
3. **🐳 Verification Loop** (Before push): act/Docker → Verify environment

## Quick Start

### 1. Install dependencies

```bash
make setup
```

### 2. Run tests (native, instant)

```bash
make test
```

### 3. Before commit (native CI)

```bash
make ci-native
```

### 4. Before push (verify with Docker)

```bash
make ci-act
```

## The Three Loops Explained

### 🔥 INNER LOOP: Native Testing (Use Constantly)

Run these **instantly** on your machine (no Docker overhead):

```bash
# Run tests (use this ALL THE TIME)
make test                # ~5 seconds

# Even faster (stops at first failure)
make test-fast           # ~3 seconds

# With coverage report
make test-coverage       # ~10 seconds

# Check code quality
make lint                # ~2 seconds

# Auto-format code
make format              # ~1 second

# Type check
make typecheck           # ~5 seconds
```

**When to use**: After every code change. These are fast enough to run continuously.

### ⚡ OUTER LOOP: Native CI (Before Commit)

Run everything CI will run, but natively (no Docker):

```bash
make ci-native           # ~30 seconds
```

This runs: linting → type checking → tests with coverage.

**When to use**: Before committing. Catches all issues without Docker overhead.

### 🐳 VERIFICATION LOOP: Docker/act (Before Push)

Verify your code works in the exact CI environment:

```bash
# With container reuse (faster after first run)
make ci-act              # ~90 seconds

# Clean slate (no cached containers)
make ci-act-clean        # ~2 minutes
```

**When to use**: 
- ✅ Before pushing to remote
- ✅ Debugging CI failures (reproduces exact environment)
- ✅ Testing workflow YAML changes
- ❌ NOT after every code change (too slow!)

## When to Use What

| Scenario | Command | Speed | Purpose |
|----------|---------|-------|---------|
| Changed 1 function | `make test` | ~5s | Quick logic check |
| Changed 3 files | `make test` | ~10s | Still instant |
| Fixed linting | `make lint` | ~2s | Verify fixes |
| About to commit | `make ci-native` | ~30s | Catch all issues |
| About to push | `make ci-act` | ~90s | Verify Docker/OS |
| CI failed on GitHub | `make ci-act-clean` | ~2m | Debug exact environment |

## Common Tasks

### Development

```bash
make help              # Show all commands
make setup             # Install dependencies
make test              # Run tests (your main command!)
make test-fast         # Fastest tests
make lint              # Check code quality
make format            # Auto-format code
make clean             # Clean build artifacts
```

### Before Commit

```bash
make ci-native         # Run full CI suite natively
```

### Before Push

```bash
make ci-act            # Verify in Docker
```

### Building & Releasing

```bash
make build             # Build package
make check-dist        # Verify package
make tag VERSION=0.2.0 # Tag and push release
```

## Installation

### Install act (optional, only for Docker verification)

```bash
# macOS
brew install act

# Linux
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

**Note**: You don't need act for daily development! Only install if you want to verify Docker/CI compatibility before pushing.

## Workflow Organization

### Makefile (Single Source of Truth)

All commands are defined in `Makefile`. GitHub Actions workflows call these same commands.

### GitHub Actions Workflows

- **test.yml** - Full test suite (runs `make setup`, `make lint`, `make test-coverage`)
  - Python 3.10, 3.11, 3.12 matrix
  - Coverage to Codecov
  
- **pre-commit.yml** - Code quality (runs `make lint`, `make typecheck`)
  
- **release.yml** - Package building
  - Triggered by version tags (`v*.*.*`)
  - Builds and publishes to PyPI

- **local-*.yml** - Helper workflows for act
  - `local-test-fast.yml` - Quick tests
  - `local-format.yml` - Auto-format
  - `local-clean.yml` - Clean artifacts

### Configuration Files

- `Makefile` - All commands (THE source of truth)
- `.actrc` - act configuration (container reuse, offline mode)
- `pyproject.toml` - Package metadata, dependencies
- `.github/workflows/` - CI workflows (call make commands)

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
