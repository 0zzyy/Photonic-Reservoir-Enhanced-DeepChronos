from photonic_rc.physics.ikeda import run_ikeda, IkedaParams
import jax.numpy as jnp


def test_run_ikeda_shape():
    params = IkedaParams()
    inputs = jnp.ones(10)
    states = run_ikeda(inputs, params)
    assert states.shape == (10,)

