"""Unit tests for lacuna.cli.main."""

from argparse import Namespace
from pathlib import Path

import pytest

from lacuna.cli.main import RunConfig


def _base_ns(**overrides):
    base = dict(
        bids_dir=Path("/in"),
        output_dir=Path("/out"),
        analysis="lntf",
        participant_label=None,
        session_id=None,
        pattern=None,
        mask_space=None,
        nprocs=-1,
        batch_size=-1,
        tmp_dir=None,
        overwrite=False,
        keep_intermediate=False,
        on_empty="warn",
        verbose_count=0,
        ntatlas_dir=None,
        ace_dir=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_runconfig_export_provenance_defaults_false():
    cfg = RunConfig.from_args(_base_ns())
    assert cfg.export_provenance is False


def test_runconfig_export_provenance_when_flag_set():
    cfg = RunConfig.from_args(_base_ns(export_provenance=True))
    assert cfg.export_provenance is True
