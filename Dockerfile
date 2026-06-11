# syntax=docker/dockerfile:1
# Lacuna Production Docker Image

# =============================================================================
# Stage 1: Install MRtrix3 via Conda
# =============================================================================
FROM continuumio/miniconda3:latest AS mrtrix-builder
RUN conda install -y -c conda-forge -c mrtrix3 mrtrix3 libstdcxx-ng \
    && conda clean -afy

# =============================================================================
# Stage 2: Build the Lacuna Wheel
# =============================================================================
FROM python:3.11-slim AS lacuna-builder
WORKDIR /build
# Install git for setuptools-scm versioning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install build && python -m build --wheel

# =============================================================================
# Stage 3: Final Production Image
# =============================================================================
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="Lacuna"
LABEL org.opencontainers.image.source="https://github.com/m-petersen/lacuna"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy MRtrix3
COPY --from=mrtrix-builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# Reference grids ship bundled in the wheel; the nonlinear warps are fetched
# from OSF at build time into a shared cache so the image is offline-complete.
ENV LACUNA_CACHE_DIR="/opt/lacuna-cache"

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 lacuna \
    && mkdir -p /data /output /work /scratch /connectomes \
    && chmod a+rwx /data /output /work /scratch /connectomes

ENV HOME="/home/lacuna"
ENV LACUNA_TMP_DIR="/tmp"

WORKDIR /app

# Install Lacuna from wheel
COPY --from=lacuna-builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Pre-fetch the 6Asym<->2009c nonlinear warps from OSF (sha256-verified) so the
# container works without network at runtime. World-readable for the lacuna user.
RUN python3 -c "from lacuna.assets.transforms import load_transform; \
    load_transform('MNI152NLin6Asym_to_MNI152NLin2009cAsym'); \
    load_transform('MNI152NLin2009cAsym_to_MNI152NLin6Asym')" \
    && chmod -R a+rX "$LACUNA_CACHE_DIR"

USER lacuna
WORKDIR /home/lacuna
ENTRYPOINT ["lacuna"]
CMD ["--help"]