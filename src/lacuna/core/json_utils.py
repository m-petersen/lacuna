"""Shared JSON serialization helpers."""

import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that serializes numpy scalars and arrays.

    numpy types are not JSON-serializable by ``json`` out of the box, and
    analysis results and provenance routinely contain them (e.g. ``np.float32``
    summary stats, ``np.int64`` counts). Pass ``cls=NumpyJSONEncoder`` to
    ``json.dump``/``json.dumps`` to handle them.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
