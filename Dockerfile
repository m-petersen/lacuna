# Dockerfile for local testing and CI/CD
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests

# Install package with dev dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Default command: run tests
CMD ["pytest", "-v", "--cov=lacuna", "-n", "auto"]
