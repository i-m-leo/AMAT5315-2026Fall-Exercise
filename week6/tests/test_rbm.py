from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from rbm import cd_surrogate, energy, gibbs_steps, init_params, visible_free_energy
from complete_rbm import clamped_gibbs
from ais_rbm import ais_log_weights, softplus


def test_energy_and_visible_free_energy_shapes():
    params = (jnp.ones((4, 3)), jnp.zeros(4), jnp.zeros(3))
    v = jnp.ones((2, 4))
    h = jnp.zeros((2, 3))
    assert energy(v, h, params).shape == (2,)
    assert visible_free_energy(v, params).shape == (2,)


def test_binary_gibbs_reproducible_and_correct_shape():
    params = (jnp.zeros((4, 3)), jnp.zeros(4), jnp.zeros(3))
    v = jnp.zeros((2, 4))
    a = gibbs_steps(jax.random.PRNGKey(7), v, params, 2)
    b = gibbs_steps(jax.random.PRNGKey(7), v, params, 2)
    np.testing.assert_array_equal(a, b)
    assert a.shape == v.shape
    assert np.isin(np.asarray(a), [0, 1]).all()


def test_clamped_gibbs_preserves_observed_pixels():
    params = (jnp.zeros((4, 3)), jnp.zeros(4), jnp.zeros(3))
    current = jnp.zeros((2, 4))
    observed = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=jnp.float32)
    mask = jnp.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    sample = clamped_gibbs(jax.random.PRNGKey(9), current, observed, mask, params)
    np.testing.assert_array_equal(np.asarray(sample[:, :2]), np.asarray(observed[:, :2]))
    assert np.isin(np.asarray(sample), [0, 1]).all()


def test_ais_exact_for_uncoupled_rbm():
    w = jnp.zeros((4, 3))
    vb = jnp.array([-1.0, 0.0, 0.5, 2.0])
    hb = jnp.array([-0.5, 0.2, 1.1])
    logw = ais_log_weights(jax.random.PRNGKey(3), w, vb, hb, 5,
                           jnp.linspace(0.0, 1.0, 6)[1:])
    np.testing.assert_allclose(np.asarray(logw), np.zeros(5), atol=1e-6)
    exact_logz = jnp.sum(softplus(vb)) + jnp.sum(softplus(hb))
    estimate = exact_logz + jax.scipy.special.logsumexp(logw) - jnp.log(logw.size)
    np.testing.assert_allclose(float(estimate), float(exact_logz), atol=1e-6)


def test_cd_gradient_finite():
    x = np.random.default_rng(4).integers(0, 2, size=(3, 4)).astype(np.float32)
    params = init_params(jax.random.PRNGKey(2), 4, 3, x, 0.01, 0, (0.01, 0.99))
    grad = jax.grad(cd_surrogate)(params, jnp.asarray(x), jax.random.PRNGKey(1), 1)
    assert all(np.isfinite(np.asarray(part)).all() for part in grad)
