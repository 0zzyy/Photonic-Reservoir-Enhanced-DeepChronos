from __future__ import annotations
"""2-D waveguide mesh reservoir.

Placeholder implementation of a mesh of Mach-Zehnder interferometers forming a
reconfigurable photonic reservoir [Shen2017]_.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MeshParams:
    """Parameters for the waveguide mesh."""

    size: int = 4
    phase_shift: float = 0.0


def mesh_propagate(state: jnp.ndarray, params: MeshParams) -> jnp.ndarray:
    """Propagate fields through a simple mesh with fixed phase shifts."""
    phase = jnp.exp(1j * params.phase_shift)
    return jnp.roll(state * phase, 1, axis=1)


def run_mesh(inputs: jnp.ndarray, params: MeshParams) -> jnp.ndarray:
    """Iteratively apply the mesh to the inputs."""

    def step(carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        new_carry = mesh_propagate(carry + x, params)
        return new_carry, new_carry

    init = jnp.zeros((params.size, params.size), dtype=jnp.complex64)
    _, states = jax.lax.scan(step, init, inputs)
    return states
