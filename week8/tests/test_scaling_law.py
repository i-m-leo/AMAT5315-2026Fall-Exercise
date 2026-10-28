"""The scaling law: the fit, the compute-optimal split and the printed parameters."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import scaling_law


def test_load_runs_drops_the_five_highest_losses():
    if not scaling_law.RUNS.exists():
        pytest.skip("the Chinchilla runs are a download, kept out of git")
    runs = scaling_law.load_runs()
    assert len(runs["N"]) == len(runs["all"]["N"]) - scaling_law.DROPPED
    assert runs["dropped"]["L"].min() >= runs["L"].max()
    np.testing.assert_allclose(runs["D"], runs["C"] / (6.0 * runs["N"]))


def test_the_law_reproduces_its_own_parameters():
    parameters = {"E": 1.8, "A": 400.0, "B": 400.0, "alpha": 0.34, "beta": 0.28}
    n = np.asarray([1e9, 1e10])
    d = np.asarray([1e10, 1e11])
    expected = (parameters["E"] + parameters["A"] / n ** parameters["alpha"]
                + parameters["B"] / d ** parameters["beta"])
    np.testing.assert_allclose(scaling_law.predict(parameters, n, d), expected)


def test_pack_and_unpack_round_trip():
    parameters = {"E": 1.8, "A": 400.0, "B": 2000.0, "alpha": 0.35, "beta": 0.36}
    restored = scaling_law.unpack(scaling_law.pack(parameters))
    for key, value in parameters.items():
        assert restored[key] == pytest.approx(value)


def test_huber_loss_is_quadratic_inside_delta_and_linear_outside():
    delta = scaling_law.HUBER_DELTA
    inside = scaling_law._huber(np.asarray([delta / 2.0]))
    outside = scaling_law._huber(np.asarray([10 * delta]))
    assert inside[0] == pytest.approx(0.5 * (delta / 2.0) ** 2)
    assert outside[0] == pytest.approx(delta * (10 * delta - 0.5 * delta))
    assert outside[0] < 0.5 * (10 * delta) ** 2


def test_compute_optimal_split_satisfies_its_first_order_condition():
    parameters = {"E": 1.8, "A": 400.0, "B": 2000.0, "alpha": 0.35, "beta": 0.36}
    for compute in (1e19, 1e21, 1e23):
        n_star, d_star = scaling_law.optimal_split(parameters, np.asarray([compute]))
        # The budget is spent exactly.
        assert 6.0 * n_star[0] * d_star[0] == pytest.approx(compute, rel=1e-9)
    # Both grow nearly like sqrt(C) when alpha and beta are close.
    compute = np.logspace(18, 24, 7)
    n_star, d_star = scaling_law.optimal_split(parameters, compute)
    slope_n = np.polyfit(np.log(compute), np.log(n_star), 1)[0]
    slope_d = np.polyfit(np.log(compute), np.log(d_star), 1)[0]
    assert slope_n + slope_d == pytest.approx(1.0, abs=1e-9)
    assert 0.45 < slope_n < 0.55


def test_a_smaller_beta_buys_far_more_tokens_per_parameter():
    """The printed law's beta = 0.28 implies an implausible split."""
    refit = {"E": 1.8, "A": 478.0, "B": 2144.0, "alpha": 0.347, "beta": 0.367}
    n_refit, d_refit = scaling_law.optimal_split(refit, np.asarray([5.76e23]))
    n_printed, d_printed = scaling_law.optimal_split(scaling_law.PRINTED,
                                                     np.asarray([5.76e23]))
    assert 10.0 < d_refit[0] / n_refit[0] < 30.0
    assert d_printed[0] / n_printed[0] > 50.0


def test_grid_fit_recovers_synthetic_parameters():
    truth = {"E": 1.8, "A": 400.0, "B": 2000.0, "alpha": 0.35, "beta": 0.36}
    n = np.logspace(8, 10, 12)
    d = np.logspace(9, 11, 12)
    measured = scaling_law.predict(truth, n, d)
    best, _ = scaling_law.grid_fit(n, d, measured)
    assert best["alpha"] == pytest.approx(truth["alpha"], abs=0.02)
    assert best["beta"] == pytest.approx(truth["beta"], abs=0.02)
    assert best["E"] == pytest.approx(truth["E"], abs=0.05)
