"""Regression test: multiple --session-id values must filter, not be ignored.

Previously _build_pattern only added a glob fragment for a single session, so
`--session-id 01 02` silently processed *all* sessions.
"""

from lacuna.cli.main import _filter_by_sessions


class _Stub:
    def __init__(self, session):
        self.metadata = {"session_id": session}


def _sessions(result):
    return sorted(s.metadata["session_id"] for s in result)


def test_multiple_sessions_are_filtered():
    subs = [_Stub("01"), _Stub("02"), _Stub("03")]
    assert _sessions(_filter_by_sessions(subs, ["01", "03"])) == ["01", "03"]


def test_single_session_filtered():
    subs = [_Stub("01"), _Stub("02")]
    assert _sessions(_filter_by_sessions(subs, ["02"])) == ["02"]


def test_session_prefix_insensitive():
    subs = [_Stub("01"), _Stub("02")]
    assert _sessions(_filter_by_sessions(subs, ["ses-01"])) == ["01"]


def test_no_match_returns_empty():
    subs = [_Stub("01"), _Stub("02")]
    assert _filter_by_sessions(subs, ["99"]) == []
