r"""Ikeda delay map reservoir.

Implements a single-node delay-based photonic reservoir following the Ikeda map
ODE [Ikeda1979]_. The state update is

.. math::
   E_{n+1} = \rho E_n \exp\bigl(j(\beta |E_n|^2 + \phi)\bigr) + \gamma u_n,

where :math:`E_n` is the complex field and :math:`u_n` the input.
"""
from __future__ import annotations
from jax.tree_util import register_pytree_node_class

import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import Tuple

@register_pytree_node_class

@dataclass
class IkedaParams:
    """Parameters for the Ikeda reservoir."""

    def tree_flatten(self):
        return ((self.rho, self.beta, self.phi, self.gamma), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        rho, beta, phi, gamma = children
        return cls(rho, beta, phi, gamma)

    rho: float = 0.9
    beta: float = 0.4
    phi: float = 0.0
    gamma: float = 0.2


@jit
def ikeda_step(state: jnp.ndarray, inp: jnp.ndarray, params: IkedaParams) -> jnp.ndarray:
    """One time-step update.

    Parameters
    ----------
    state:
        Complex state ``E_n``.
    inp:
        Real-valued input ``u_n``.
    params:
        Ikeda parameters.

    Returns
    -------
    jnp.ndarray
        Updated state ``E_{n+1}``.
    """
    phase = params.beta * jnp.abs(state) ** 2 + params.phi
    return params.rho * state * jnp.exp(1j * phase) + params.gamma * inp


def run_ikeda(inputs: jnp.ndarray, params: IkedaParams, *, init: complex = 0j) -> jnp.ndarray:
    """Run reservoir over a sequence using ``lax.scan``."""

    def step(carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        new_carry = ikeda_step(carry, x, params)
        return new_carry, new_carry

    _, states = jax.lax.scan(step, jnp.asarray(init, dtype=jnp.complex64), inputs)
    return states
