"""Regression test: extract() must not silently drop subjects that share an
identifier (e.g. multiple lesions for one subject); it disambiguates instead."""

import warnings

from lacuna.batch import extract


class _Stub:
    """Minimal stand-in for SubjectData with the attributes extract() reads."""

    def __init__(self, score):
        self.metadata = {"subject_id": "sub-001"}  # identical -> identifier collides
        self.results = {"MyAnalysis": {"score": score}}


def test_extract_disambiguates_colliding_identifiers():
    subjects = [_Stub(1.0), _Stub(2.0)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = extract(subjects, analysis="MyAnalysis", pattern="score")

    # Both subjects preserved (no silent overwrite): one under the id, one suffixed
    assert len(out) == 2
    assert "sub-001" in out
    assert any(k.startswith("sub-001#") for k in out)
    assert sorted(out.values()) == [1.0, 2.0]


def test_extract_warns_on_collision():
    subjects = [_Stub(1.0), _Stub(2.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract(subjects, analysis="MyAnalysis", pattern="score")
    assert any("Duplicate result identifier" in str(w.message) for w in caught)
