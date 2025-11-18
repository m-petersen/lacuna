.PHONY: help setup test test-fast test-coverage lint format typecheck clean ci-native ci-act

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === INNER LOOP: Fast Native Commands (use these constantly) ===

setup:  ## Install dependencies
	pip install -e ".[dev]"

test:  ## Run tests (fast, native, use this constantly)
	pytest -v -n auto

test-fast:  ## Run tests without coverage (fastest)
	pytest -v -n auto -x

test-coverage:  ## Run tests with coverage report
	pytest -v --cov=lacuna --cov-report=term-missing --cov-report=html -n auto

lint:  ## Run linting checks (fast, native)
	@echo "Running ruff..."
	@ruff check src tests
	@echo "Running black..."
	@black --check src tests

format:  ## Format code (fast, native)
	@echo "Formatting with ruff..."
	@ruff check --fix src tests
	@echo "Formatting with black..."
	@black src tests

typecheck:  ## Run type checking (fast, native)
	mypy src --ignore-missing-imports

clean:  ## Clean build artifacts and cache
	@rm -rf build/ dist/ *.egg-info
	@rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

# === OUTER LOOP: Verify CI compatibility (use before commit) ===

ci-native:  ## Run full CI checks natively (use before commit)
	@echo "=== Running Full CI Suite Natively ==="
	@echo "1. Linting..."
	@$(MAKE) lint
	@echo "\n2. Type checking..."
	@$(MAKE) typecheck
	@echo "\n3. Tests with coverage..."
	@$(MAKE) test-coverage
	@echo "\n✓ All CI checks passed!"

ci-act:  ## Verify CI works in Docker with act (use before push)
	@echo "=== Verifying CI in Docker ==="
	act -j test --reuse

ci-act-clean:  ## Run act without reusing containers (clean slate)
	act -j test

# === RELEASE ===

build:  ## Build package distribution
	python -m build

check-dist:  ## Check package distribution
	twine check dist/*

tag:  ## Create and push a version tag (usage: make tag VERSION=0.1.0)
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make tag VERSION=0.1.0"; \
		exit 1; \
	fi
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push origin "v$(VERSION)"
