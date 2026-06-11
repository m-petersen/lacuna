"""Regression test: setup_tutorial_data must produce a clean BIDS dataset,
not copy the Python package machinery (__init__.py / __pycache__)."""

from lacuna.data.tutorials import setup_tutorial_data


def test_setup_tutorial_data_excludes_package_files(tmp_path):
    target = setup_tutorial_data(tmp_path / "ds", overwrite=True)

    names = {p.name for p in target.rglob("*")}
    assert "__init__.py" not in names
    assert "__pycache__" not in names
    assert not any(n.endswith(".pyc") for n in names)

    # Real BIDS content is present
    assert (target / "dataset_description.json").exists()
    assert (target / "participants.tsv").exists()
    assert (target / "sub-01").is_dir()
