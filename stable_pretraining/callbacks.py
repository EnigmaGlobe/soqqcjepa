"""Minimal callback factory shim for `stable_pretraining`.

Lightning attempts to import `stable_pretraining.callbacks` via entrypoints
to load external callback factories. Provide a minimal `load()` that returns
an empty list so training can proceed without the optional callbacks.
"""
from typing import List


def load() -> List:
    """Return an empty list of callbacks (no external callbacks)."""
    return []
