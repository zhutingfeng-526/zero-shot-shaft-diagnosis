"""Numerical background functions."""
from __future__ import annotations

CONTEXT_FAMILIES = ("normal", "U", "M")

def _bind(modules):
    globals().update(modules)
