"""Energy estimation utilities."""
from __future__ import annotations

import jax.numpy as jnp


def energy_per_op(power_mw: float, ops: int, *, efficiency: float = 1.0) -> float:
    """Estimate energy in picojoules per operation."""
    energy_j = power_mw * 1e-3 / efficiency / ops
    return energy_j * 1e12

