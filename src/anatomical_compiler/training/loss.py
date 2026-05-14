"""Differentiable loss for fitting a Hypergraph Neural ODE to a TF timecourse.

The pattern is ported from bl1.training.loss: build the scalar loss out of
named components, each gradient-friendly, and return a components dict
alongside the total so the training loop can log each piece per-epoch.

In bl1 the components are (firing-rate, burst-rate, synchrony, weight-reg)
tuned against Wagenaar 2006 culture targets.  Here the analogues are:

  1. **rollout MSE**     -- base reconstruction error against observed
                            timepoints (mandatory; analog of FR loss).
  2. **driver MSE**      -- weighted reconstruction error on a user-supplied
                            driver TF panel.  Pushes the ODE to capture
                            known regenerative drivers especially well
                            (analog of target-firing-rate matching).
  3. **stress hinge**    -- *penalises low* per-TF MSE on a stress-responder
                            panel (FOS/JUN/ATF3/...).  Encourages the ODE
                            to leave transient stress responses unmodelled
                            so the learned dynamics correspond to the
                            stable regenerative flow, not the transient.
                            Soft hinge:  relu(margin - mse_stress)^2.
                            (Analog of bl1's synchrony penalty.)
  4. **trajectory smoothness** -- L2 on consecutive predicted timepoint
                            differences; regularises against ODE rollouts
                            that overshoot between observed bins.
  5. **per-TF uniformity** -- variance of per-TF rollout MSE across the
                            panel.  Encourages the model to fit all TFs
                            equally well, which is the modularity-aware
                            regime (the alternative -- a small number of
                            TFs absorbing all reconstruction quality --
                            is exactly what driver/stress weighting wants
                            to *select*, not what we want the optimiser to
                            stumble into by accident).
  6. **parameter L2**    -- standard weight regularisation across all
                            array leaves of the eqx model.

All components are scalars on the same scale, combined with config-supplied
weights so the trainer can rebalance them at runtime.

Every function in this module is pure JAX (no NumPy, no Python loops on
trajectory data) and differentiable end-to-end so it can be wrapped by
``eqx.filter_value_and_grad``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# 1. Rollout reconstruction
# ---------------------------------------------------------------------------


def rollout_mse(pred: Array, target: Array, mask: Array | None = None) -> Array:
    """Masked MSE between predicted and observed trajectories.

    Args:
        pred: ``(T, N, D)`` predicted trajectory at the target timepoints.
        target: ``(T, N, D)`` observed trajectory.
        mask: Optional ``(T, N)`` boolean/float mask; masked-out
            (node, timepoint) pairs do not contribute.

    Returns:
        Scalar MSE.
    """
    sq = (pred - target) ** 2
    if mask is not None:
        sq = sq * mask[..., None]
        denom = jnp.sum(mask) * pred.shape[-1] + 1e-8
        return jnp.sum(sq) / denom
    return jnp.mean(sq)


# ---------------------------------------------------------------------------
# 2. Driver-panel weighted reconstruction
# ---------------------------------------------------------------------------


def driver_mse(
    pred: Array,
    target: Array,
    driver_idx: Array,
) -> Array:
    """MSE restricted to a known-driver TF panel.

    Args:
        pred: ``(T, N, D)`` predicted trajectory.
        target: ``(T, N, D)`` observed trajectory.
        driver_idx: 1-D int array of node indices considered drivers.

    Returns:
        Scalar MSE on the selected nodes, or 0.0 if the panel is empty.
    """
    if driver_idx.size == 0:
        return jnp.asarray(0.0, dtype=pred.dtype)
    pred_d = pred[:, driver_idx, :]
    target_d = target[:, driver_idx, :]
    return jnp.mean((pred_d - target_d) ** 2)


# ---------------------------------------------------------------------------
# 3. Stress-responder soft hinge -- penalise *fitting* transient stress
# ---------------------------------------------------------------------------


def stress_hinge(
    pred: Array,
    target: Array,
    stress_idx: Array,
    margin: float = 0.1,
) -> Array:
    """Soft hinge that penalises low MSE on stress responders.

    The learned ODE should describe the *stable* regenerative flow, not the
    transient FOS/JUN/ATF3 response.  We therefore push reconstruction of
    those TFs to be no better than ``margin``:  if their mean MSE drops
    below ``margin`` the loss grows quadratically.

    ``relu(margin - mse_stress)^2`` is smooth (one continuous derivative)
    and exactly zero once the stress MSE clears the margin, so it stops
    contributing to gradients in the regime we want.

    Args:
        pred: ``(T, N, D)`` predicted trajectory.
        target: ``(T, N, D)`` observed trajectory.
        stress_idx: 1-D int array of stress-responder node indices.
        margin: Threshold MSE below which the penalty activates.

    Returns:
        Scalar non-negative penalty.
    """
    if stress_idx.size == 0:
        return jnp.asarray(0.0, dtype=pred.dtype)
    sq = (pred[:, stress_idx, :] - target[:, stress_idx, :]) ** 2
    mse_stress = jnp.mean(sq)
    gap = jax.nn.relu(margin - mse_stress)
    return gap * gap


# ---------------------------------------------------------------------------
# 4. Trajectory smoothness (predicted, not observed)
# ---------------------------------------------------------------------------


def trajectory_smoothness(pred: Array) -> Array:
    """L2 on consecutive predicted-timepoint differences.

    Acts on the **predicted** rollout so it regularises ODE behaviour, not
    the data.  (``hgx.temporal_smoothness_loss`` covers the observed-data
    version.)

    Args:
        pred: ``(T, N, D)`` predicted trajectory with ``T >= 2``.

    Returns:
        Scalar mean squared finite-difference.
    """
    diffs = pred[1:] - pred[:-1]
    return jnp.mean(diffs ** 2)


# ---------------------------------------------------------------------------
# 5. Per-TF MSE uniformity (variance across panel)
# ---------------------------------------------------------------------------


def per_tf_mse_uniformity(pred: Array, target: Array) -> tuple[Array, Array]:
    """Variance of per-node rollout MSE across the panel.

    A low value means every TF is captured about equally well -- the regime
    where the *post hoc* driver/stress split is determined by domain
    weighting (driver_mse / stress_hinge), not by which TFs happen to be
    easier for the optimiser.

    Returns:
        ``(variance, per_node_mse)`` -- the scalar penalty plus the raw
        per-node MSE vector so the trainer can log it without recomputing.
    """
    sq = (pred - target) ** 2                  # (T, N, D)
    per_node = jnp.mean(sq, axis=(0, 2))       # (N,)
    return jnp.var(per_node), per_node


# ---------------------------------------------------------------------------
# 6. Parameter L2 regularisation (eqx model)
# ---------------------------------------------------------------------------


def parameter_l2(model: Any) -> Array:
    """Sum of squared array leaves under ``eqx.partition(model, is_array)``.

    Works for any eqx module (Hypergraph Neural ODE, plain conv, etc.).
    """
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        return jnp.asarray(0.0)
    return sum(jnp.sum(leaf ** 2) for leaf in leaves)


# ---------------------------------------------------------------------------
# Top-level composite loss
# ---------------------------------------------------------------------------


def hg_ode_loss(
    pred: Array,
    target: Array,
    model: Any,
    driver_idx: Array,
    stress_idx: Array,
    mask: Array | None,
    *,
    w_rollout: float,
    w_driver: float,
    w_stress: float,
    w_smooth: float,
    w_uniformity: float,
    w_reg: float,
    stress_margin: float,
) -> tuple[Array, dict[str, Array]]:
    """Combine the six components into a scalar loss + components dict.

    Returned ``components`` mirrors bl1's ``culture_loss`` output: every
    individual loss term plus a few non-differentiable "metrics" (per-TF
    MSE vector, mean stress MSE) that are useful for logging but should
    *not* be backpropagated through if you were to take a separate
    gradient.  Here they are computed from the same JAX arrays as the
    losses so they remain in the autodiff graph, but the trainer only
    backprops the scalar ``total``.

    Args:
        pred: ``(T, N, D)`` rollout from the model at the target times.
        target: ``(T, N, D)`` observed trajectory at the same times.
        model: The eqx model being trained (used for parameter_l2 only).
        driver_idx: int array of driver-panel node indices.
        stress_idx: int array of stress-responder node indices.
        mask: optional ``(T, N)`` target-validity mask.
        w_*: scalar weights on each loss component.
        stress_margin: margin for ``stress_hinge``.

    Returns:
        ``(total_loss, components_dict)``.
    """
    L_rollout = rollout_mse(pred, target, mask=mask)
    L_driver = driver_mse(pred, target, driver_idx)
    L_stress = stress_hinge(pred, target, stress_idx, margin=stress_margin)
    L_smooth = trajectory_smoothness(pred)
    L_uniform, per_node_mse = per_tf_mse_uniformity(pred, target)
    L_reg = parameter_l2(model)

    total = (
        w_rollout * L_rollout
        + w_driver * L_driver
        + w_stress * L_stress
        + w_smooth * L_smooth
        + w_uniformity * L_uniform
        + w_reg * L_reg
    )

    if stress_idx.size > 0:
        stress_mse_metric = jnp.mean(
            (pred[:, stress_idx, :] - target[:, stress_idx, :]) ** 2
        )
    else:
        stress_mse_metric = jnp.asarray(0.0, dtype=pred.dtype)

    components = {
        "rollout_mse": L_rollout,
        "driver_mse": L_driver,
        "stress_hinge": L_stress,
        "stress_mse": stress_mse_metric,
        "trajectory_smoothness": L_smooth,
        "uniformity": L_uniform,
        "param_l2": L_reg,
        "total": total,
        "per_tf_mse": per_node_mse,
    }
    return total, components
