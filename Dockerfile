# syntax=docker/dockerfile:1
# Lacuna Production Docker Image

# =============================================================================
# Stage 1: Install MRtrix3 via Conda
# =============================================================================
FROM continuumio/miniconda3:latest AS mrtrix-builder
RUN conda install -y -c conda-forge -c mrtrix3 mrtrix3 libstdcxx-ng \
    && conda clean -afy

# =============================================================================
# Stage 2: Fetch TemplateFlow templates & CLEAN UP
# =============================================================================
FROM python:3.10-slim AS templateflow-fetcher
ENV TEMPLATEFLOW_HOME="/templateflow"
RUN pip install --no-cache-dir templateflow

# 1. Fetch specific templates
RUN python3 <<EOF
import templateflow.api as tf
# MNI152NLin6Asym
tf.get('MNI152NLin6Asym', resolution=1, desc=None, suffix='T1w')
tf.get('MNI152NLin6Asym', resolution=2, desc=None, suffix='T1w')
tf.get('MNI152NLin6Asym', resolution=1, desc='brain', suffix='mask')
tf.get('MNI152NLin6Asym', resolution=2, desc='brain', suffix='mask')
# MNI152NLin2009cAsym
tf.get('MNI152NLin2009cAsym', resolution=1, desc=None, suffix='T1w')
tf.get('MNI152NLin2009cAsym', resolution=2, desc=None, suffix='T1w')
tf.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask')
tf.get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask')
# Transforms
tf.get('MNI152NLin6Asym', suffix='xfm', extension='.h5')
tf.get('MNI152NLin2009cAsym', suffix='xfm', extension='.h5')
EOF

# 2. WHITELIST CLEANUP: Delete everything that is NOT the two MNI templates
# This removes tpl-dhcp, tpl-fsLR, tpl-MouseIn, etc.
RUN find /templateflow -mindepth 1 -maxdepth 1 -type d -name "tpl-*" \
    -not -name "tpl-MNI152NLin6Asym" \
    -not -name "tpl-MNI152NLin2009cAsym" \
    -exec rm -rf {} +

# =============================================================================
# Stage 3: Build the Lacuna Wheel
# =============================================================================
FROM python:3.10-slim AS lacuna-builder
WORKDIR /build
# Install git for setuptools-scm versioning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install build && python -m build --wheel

# =============================================================================
# Stage 4: Final Production Image
# =============================================================================
FROM python:3.10-slim AS production

LABEL org.opencontainers.image.title="Lacuna"
LABEL org.opencontainers.image.source="https://github.com/lacuna/lacuna"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy MRtrix3
COPY --from=mrtrix-builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# Copy TemplateFlow (NOW CLEANED)
COPY --from=templateflow-fetcher /templateflow /templateflow
ENV TEMPLATEFLOW_HOME="/templateflow"

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

USER lacuna
WORKDIR /home/lacuna
ENTRYPOINT ["lacuna"]
CMD ["--help"]