"""Tiny local shim for `stable_worldmodel` used by training script.

This implements the minimal `wm.dinowm.Embedder` used to create
action/proprio encoders in `get_world_model`.
"""
from . import wm

__all__ = ["wm"]
from . import wm
