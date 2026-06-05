"""Factory shim returning no external callbacks.

Lightning expects callback factories under entrypoints; provide a minimal
`load()` function that returns an empty list.
"""
from typing import List


def load() -> List:
    return []


def default():
    """Compatibility alias expected by Lightning's registry entrypoint resolution."""
    return load()
