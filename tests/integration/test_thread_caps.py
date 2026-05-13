"""
Regression tests for the joblib fork-after-BLAS deadlock fix.

These tests pin down the two guarantees that make the deadlock impossible on
many-core nodes. They do NOT reproduce the deadlock itself (it is a fork-time
race condition); they verify the contracts that the fix maintains.

If either guarantee silently regresses, one of these tests fails.

1. Importing ``lacuna`` caps the BLAS/OMP env vars to "1" before numpy/scipy/
   nilearn load their native libraries — unless the user explicitly set them.
   Tested via a subprocess so we can control the starting environment.

2. ``ParallelStrategy.execute`` runs each worker with BLAS thread pools
   capped to 1 (via ``parallel_backend(..., inner_max_num_threads=1)``).
   Tested by introspecting threadpoolctl from inside a real loky worker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData
from lacuna.analysis.base import BaseAnalysis
from lacuna.batch.strategies import ParallelStrategy

BLAS_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_PROBE_CODE = """\
import json, os
import lacuna  # noqa: F401
print(json.dumps({k: os.environ.get(k) for k in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)}))
"""


def _probe_env_after_import(overrides: dict[str, str]) -> dict[str, str | None]:
    """Spawn a fresh Python, import lacuna, return the BLAS env vars."""
    env = {k: v for k, v in os.environ.items() if k not in BLAS_VARS}
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_caps_blas_env_vars_when_unset():
    """With no BLAS env vars set, importing lacuna must default all four to '1'."""
    got = _probe_env_after_import(overrides={})
    for var in BLAS_VARS:
        assert got[var] == "1", f"{var} not capped on import: {got[var]!r}"


def test_import_preserves_user_set_blas_env_vars():
    """User-set values must survive (setdefault must not overwrite)."""
    got = _probe_env_after_import(overrides={"OMP_NUM_THREADS": "4"})
    assert got["OMP_NUM_THREADS"] == "4", "user override clobbered"
    for var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert got[var] == "1", f"{var} not capped: {got[var]!r}"


class _ThreadProbeAnalysis(BaseAnalysis):
    """Minimal analysis that reports the max BLAS thread-pool size in the worker."""

    TARGET_SPACE = None  # skip spatial transformation

    def _validate_inputs(self, mask_data):  # noqa: D401
        pass

    def _run_analysis(self, mask_data):
        import threadpoolctl

        info = threadpoolctl.threadpool_info()
        max_num_threads = max((entry["num_threads"] for entry in info), default=0)
        return {"max_num_threads": max_num_threads}


@pytest.fixture
def tiny_subjects():
    """Two minimal SubjectData objects suitable for parallel execution."""
    subjects = []
    for i in range(2):
        img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
        subjects.append(
            SubjectData(
                img,
                space="MNI152NLin6Asym",
                resolution=1.0,
                metadata={"subject_id": f"sub-{i:02d}"},
            )
        )
    return subjects


def test_parallel_strategy_caps_worker_blas_threads(tiny_subjects):
    """ParallelStrategy with loky must cap BLAS thread pools to 1 in workers."""
    strategy = ParallelStrategy(n_jobs=2, backend="loky")
    results = strategy.execute(inputs=tiny_subjects, analysis=_ThreadProbeAnalysis())

    counts = [r.results["_ThreadProbeAnalysis"]["max_num_threads"] for r in results]
    assert counts, "no worker results returned"
    assert all(c == 1 for c in counts), f"worker BLAS thread counts not capped: {counts}"
