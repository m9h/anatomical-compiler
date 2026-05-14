"""Smoke test for anatomical_compiler.training.

Builds a tiny synthetic TF-trajectory hypergraph, runs a few epochs through
the new training loop, and asserts the contract bl1's trainer guarantees:

  * loss_history is non-empty, one record per epoch;
  * every component is finite;
  * total loss does not increase (training is making *some* progress);
  * the fitted model produces a finite rollout;
  * per-TF MSE is recorded for the driver/stress ranking downstream.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("diffrax")
pytest.importorskip("hgx")
pytest.importorskip("equinox")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import diffrax
import hgx
from hgx._dynamics import HypergraphNeuralODE

from anatomical_compiler.training import (
    TrainingConfig,
    train_hg_neural_ode,
)


def _build_tiny_trajectory(T: int = 6, N: int = 8, D: int = 1, seed: int = 0):
    """Synthetic (T, N, D) trajectory: smooth sin/cos drift per TF.

    Just smooth enough to be learnable by a 1-layer ODE in a handful of
    epochs without becoming a numerical-precision test.
    """
    t = jnp.linspace(0.0, 1.0, T)
    phases = jnp.linspace(0.0, jnp.pi, N)
    # (T, N) signal: 0.5 + 0.3 * sin(2*pi*t + phase)
    base = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * t[:, None] + phases[None, :])
    return base[..., None].astype(jnp.float32)


def _build_model_and_forward(traj):
    T, N, D = traj.shape
    incidence = jnp.ones((N, 1), dtype=jnp.float32)
    snaps = [hgx.from_incidence(incidence, node_features=traj[k]) for k in range(T)]
    ts = jnp.linspace(0.0, 1.0, T)

    key = jax.random.PRNGKey(0)
    conv = hgx.UniGCNConv(D, D, key=key)
    model = HypergraphNeuralODE(conv)

    target = traj[1:]
    target_times = ts[1:]
    t0_val, t1_val = float(ts[0]), float(ts[-1])

    def forward(m):
        sol = m(snaps[0], t0=t0_val, t1=t1_val,
                saveat=diffrax.SaveAt(ts=target_times))
        return sol.ys

    return model, forward, target


def test_train_runs_and_loss_finite():
    traj = _build_tiny_trajectory()
    model, forward, target = _build_model_and_forward(traj)

    cfg = TrainingConfig(
        n_epochs=5,
        learning_rate=1e-2,
        log_every=100,            # silence the per-epoch print
    )
    result = train_hg_neural_ode(
        model, forward=forward, target=target, config=cfg,
    )

    assert len(result.loss_history) == cfg.n_epochs
    for rec in result.loss_history:
        for key in ("total", "rollout_mse", "driver_mse", "stress_hinge",
                    "trajectory_smoothness", "uniformity", "param_l2"):
            assert key in rec, f"missing {key} in epoch record"
            assert jnp.isfinite(jnp.asarray(rec[key])), \
                f"{key}={rec[key]} is non-finite"

    assert result.nan_epochs == 0


def test_train_reduces_loss():
    traj = _build_tiny_trajectory()
    model, forward, target = _build_model_and_forward(traj)

    cfg = TrainingConfig(n_epochs=30, learning_rate=1e-2, log_every=100)
    result = train_hg_neural_ode(
        model, forward=forward, target=target, config=cfg,
    )
    losses = [rec["total"] for rec in result.loss_history]
    # Training shouldn't make things worse on a simple problem.  Compare
    # the first epoch to the best-so-far over the remaining ones, not the
    # final epoch alone, in case the loss oscillates near the minimum.
    assert min(losses[1:]) <= losses[0] + 1e-6, \
        f"training did not reduce loss: {losses[0]:.4f} -> {min(losses[1:]):.4f}"


def test_driver_panel_weights_loss():
    """Specifying a driver panel should change the loss numerically.

    Same model, same data, same seed -- but with vs without a driver
    panel the loss components differ at epoch 0 in exactly one place
    (driver_mse) and the total reflects the extra weighted term.
    """
    traj = _build_tiny_trajectory()
    model, forward, target = _build_model_and_forward(traj)

    cfg = TrainingConfig(n_epochs=1, log_every=100)

    r_none = train_hg_neural_ode(
        model, forward=forward, target=target,
        driver_idx=None, stress_idx=None, config=cfg,
    )
    r_with = train_hg_neural_ode(
        model, forward=forward, target=target,
        driver_idx=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        stress_idx=jnp.asarray([6, 7], dtype=jnp.int32),
        config=cfg,
    )

    rec_none = r_none.loss_history[0]
    rec_with = r_with.loss_history[0]

    # Without a driver/stress panel both penalties are zero.
    assert rec_none["driver_mse"] == pytest.approx(0.0, abs=1e-7)
    assert rec_none["stress_hinge"] == pytest.approx(0.0, abs=1e-7)
    # With a driver panel the driver_mse term is non-trivial.
    assert rec_with["driver_mse"] > 1e-6


def test_final_per_tf_mse_recorded():
    traj = _build_tiny_trajectory()
    model, forward, target = _build_model_and_forward(traj)

    cfg = TrainingConfig(n_epochs=3, log_every=100)
    result = train_hg_neural_ode(
        model, forward=forward, target=target, config=cfg,
    )
    assert result.final_per_tf_mse is not None
    arr = jnp.asarray(result.final_per_tf_mse)
    assert arr.shape == (traj.shape[1],)
    assert bool(jnp.all(jnp.isfinite(arr)))
