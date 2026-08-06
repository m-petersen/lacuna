"""Regression: JSON exporters must serialize numpy values, not crash on them.
Analysis results and provenance routinely hold numpy scalars/arrays."""

import json

import nibabel as nib
import numpy as np

from lacuna.core.subject_data import SubjectData
from lacuna.io.export import export_provenance_to_json, export_results_to_json


def _subject():
    # eye(4) affine -> 1mm; resolution must match to satisfy SubjectData validation.
    img = nib.Nifti1Image(np.array([[[0, 1]]], np.uint8), np.eye(4))
    return SubjectData(mask_img=img, space="MNI152NLin6Asym", resolution=1)


def test_export_results_to_json_handles_numpy(tmp_path):
    sd = _subject().add_result(
        "SomeAnalysis",
        {"stat": np.float32(3.5), "count": np.int64(7), "vec": np.array([1.0, 2.0])},
    )
    out = tmp_path / "res.json"
    export_results_to_json(sd, out)  # previously TypeError on np.float32
    assert json.loads(out.read_text())  # wrote valid JSON


def test_export_provenance_to_json_handles_numpy(tmp_path):
    # Provenance is normally validated JSON-serializable at creation, so this is
    # a defensive check: inject a raw provenance record carrying numpy values and
    # confirm the exporter serializes it instead of crashing.
    img = nib.Nifti1Image(np.array([[[0, 1]]], np.uint8), np.eye(4))
    sd = SubjectData(
        mask_img=img,
        space="MNI152NLin6Asym",
        resolution=1,
        provenance=[
            {
                "function": "x",
                "parameters": {"threshold": np.float32(1.25), "ids": np.array([1, 2])},
                "timestamp": "t",
                "version": "1",
            }
        ],
    )
    out = tmp_path / "prov.json"
    export_provenance_to_json(sd, out)
    json.loads(out.read_text())  # valid JSON, no crash
