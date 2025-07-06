"""Reservoir layer wrapping the Ikeda map."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Tuple

from ..physics.ikeda import IkedaParams, run_ikeda


@dataclass
class ReservoirLayer:
    """Differentiable reservoir layer using custom VJP."""

    params: IkedaParams

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return reservoir_vjp(x, self.params)


def _forward(x: jnp.ndarray, params: IkedaParams) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, IkedaParams]]:
    states = run_ikeda(x, params)
    return states, (x, params)


def _backward(res: Tuple[jnp.ndarray, IkedaParams], g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    x, params = res
    # Straight-through estimator for simplicity
    return g, None


@jax.custom_vjp
def reservoir_vjp(x: jnp.ndarray, params: IkedaParams) -> jnp.ndarray:
    return run_ikeda(x, params)


reservoir_vjp.defvjp(_forward, _backward)

