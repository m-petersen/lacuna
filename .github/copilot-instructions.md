# Lacuna Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-11-11

## Active Technologies

- Python 3.10+ (existing project constraint) (001-neuroimaging-space-handling)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.10+ (existing project constraint): Follow standard conventions

## Recent Changes

- 001-neuroimaging-space-handling: Added Python 3.10+ (existing project constraint)

<!-- MANUAL ADDITIONS START -->

## Development Workflow

### Test-Driven Development (TDD)
Follow a test driven development (TDD) approach for new features. Write tests before implementation to ensure requirements are met.
If bugs are found apart from testing, create tests that cover the bug before fixing it.

### The Three-Loop Testing Strategy

**IMPORTANT**: Use the right tool for the right feedback speed. Don't use act for every change - it's too slow!

#### 🔥 INNER LOOP: Native Testing (Every 5 minutes)
Use these commands constantly while developing. They run natively (no Docker) and are **instant**:

```bash
# Run tests (use this constantly!)
make test

# Run tests even faster (stops at first failure)
make test-fast

# Run with coverage report
make test-coverage

# Check code quality
make lint

# Auto-format code
make format

# Type check
make typecheck
```

**Rule**: Use `make test` after every significant code change. It's fast enough to run continuously.

#### ⚡ OUTER LOOP: CI Verification (Before commit)
Run the full CI suite natively to catch issues before using Docker:

```bash
# Run everything CI will run (natively, ~1-2 minutes)
make ci-native
```

This runs: linting → type checking → full test suite with coverage.

**Rule**: Run `make ci-native` before committing. Still fast, but comprehensive.

#### 🐳 VERIFICATION LOOP: act Testing (Before push)
Use act to verify your code works in the exact CI environment (Docker):

```bash
# Verify CI works in Docker (reuses containers, faster)
make ci-act

# Or directly:
sudo act -j test --reuse

# Clean slate test (no container reuse)
make ci-act-clean
```

**Rule**: Run `make ci-act` ONLY before pushing, not after every change. Docker overhead makes this slow (~1-2 minutes).

#### When to Use What

| Scenario | Command | Speed | Why |
|----------|---------|-------|-----|
| Changed 1 function | `make test` | ~5 sec | Instant feedback on logic |
| Changed 3 files | `make test` | ~10 sec | Still instant, run constantly |
| About to commit | `make ci-native` | ~30 sec | Catch all issues natively |
| About to push | `make ci-act` | ~90 sec | Verify Docker/CI compatibility |
| Debugging CI failure | `sudo act -j test -v` | ~2 min | Reproduce exact CI environment |

#### Quick Test Commands (Legacy - use Makefile instead!)

```bash
# List all available workflows
act --list

# Run full test suite (matches CI exactly)
# NOTE: Requires docker permissions - use sudo if needed
sudo act -j test --reuse

# Run fast tests without coverage (quicker feedback)
sudo act -j test-fast -W .github/workflows/local-test-fast.yml

# Run linting checks
sudo act -j lint -W .github/workflows/pre-commit.yml

# Dry run to check workflow without executing
act -n -j test -W .github/workflows/test.yml
```

#### When to Run Tests with act (Docker)

- ✅ **Before pushing to remote** - Verify CI will pass
- ✅ **Debugging CI failures** - Reproduce exact environment
- ✅ **Testing workflow changes** - Verify .github/workflows/*.yml modifications
- ❌ **After every code change** - Too slow! Use `make test` instead
- ❌ **During active development** - Use native commands for speed

#### Configuration

- `.actrc` - act configuration (reuse containers, offline mode, Docker image)
- `Makefile` - Task runner (single source of truth for all commands)
- `.github/workflows/` - All workflows (work locally with act AND in CI)

#### Troubleshooting act

If act fails with Docker permission errors:
```bash
# Option 1: Use sudo
sudo act -j test

# Option 2: Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
```

If you need to pull fresh Docker images:
```bash
sudo act -j test --pull
```

### Version Management

Version is automatically derived from git tags via setuptools-scm:
- Tagged releases: `v0.1.0` → package version `0.1.0`
- Development: `0.1.0.devN+gHASH` (auto-generated)

**Do NOT manually edit version numbers** - they are managed by git tags.

### Dependencies

- **Core dependencies**: Defined in `pyproject.toml` `[project.dependencies]`
- **Optional dependencies**: `viz`, `bids`, `preprocess` (ANTsPy - optional!)
- **Dev dependencies**: `pytest`, `pytest-xdist`, `ruff`, `black`, `mypy`

ANTsPy is OPTIONAL (only for native→MNI registration). Core functionality doesn't require it.

### Code Quality

Before committing, ensure:
1. ✅ Tests pass: `make test`
2. ✅ Full CI check: `make ci-native`
3. ✅ Code formatted: `make format`

### Workflow Philosophy

**"Task Runner Pattern"** - Makefile is the single source of truth:
- Commands run natively (instant feedback)
- GitHub Actions call make commands (no duplication)
- act verifies Docker compatibility (before push)
- Run automatically in CI on push/PR
- Run automatically in CI on push/PR
- Same Docker environment everywhere
- No Makefile needed - act replaces it

See `DEVELOPMENT.md` for comprehensive development workflow documentation.

<!-- MANUAL ADDITIONS END -->
