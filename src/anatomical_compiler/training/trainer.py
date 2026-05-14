"""Differentiable training loop for Hypergraph Neural ODEs.

Ported from ``bl1.training.trainer``.  The differences:

  * the plant is an ``eqx.Module`` (e.g. ``hgx.HypergraphNeuralODE``)
    rather than a pair of dense weight matrices, so we use
    ``eqx.partition`` + ``eqx.combine`` to clamp parameters and to take
    gradients;
  * the inner forward pass calls ``model(hg, t0, t1, saveat=...)`` and
    extracts ``sol.ys`` instead of running a spiking-network ``scan``;
  * the multi-component loss is the one in :mod:`anatomical_compiler.training.loss`.

The training-loop structure is otherwise the same: JIT-compiled
``_train_step``, ``optax.chain(clip_by_global_norm, adam)`` wrapped in
``optax.apply_if_finite`` for NaN-protected updates, per-element
parameter clamping, per-epoch history dict, optional tracker.log,
early-stop on persistent non-finite epochs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from anatomical_compiler.training.loss import hg_ode_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """All hyperparameters for fitting a Hypergraph Neural ODE.

    Field names mirror ``bl1.training.trainer.TrainingConfig`` where
    semantically equivalent (n_epochs, learning_rate, grad_clip_norm,
    max_*_value, seed); the loss-weight fields correspond to the six
    components in :func:`anatomical_compiler.training.loss.hg_ode_loss`.
    """

    # Optimisation
    n_epochs: int = 200
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0

    # Loss weights
    w_rollout: float = 1.0
    w_driver: float = 1.0
    w_stress: float = 0.5
    w_smooth: float = 0.05
    w_uniformity: float = 0.1
    w_reg: float = 1e-4

    # Stress-hinge activation margin (penalise stress MSE < margin)
    stress_margin: float = 0.05

    # Parameter clamping (per-element absolute-value cap on array leaves)
    max_param_value: float = 5.0

    # apply_if_finite tolerates this many consecutive bad gradient batches
    # before refusing further updates -- bl1 uses an explicit nan_count + 10
    # early-stop; we use the same threshold here.
    max_consecutive_nan_steps: int = 10

    # Logging cadence (print every ``log_every`` epochs)
    log_every: int = 10

    seed: int = 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Output from :func:`train_hg_neural_ode`."""

    model: Any
    loss_history: list[dict] = field(default_factory=list)
    config: TrainingConfig = field(default_factory=TrainingConfig)
    driver_idx: Array | None = None
    stress_idx: Array | None = None
    final_per_tf_mse: Array | None = None
    nan_epochs: int = 0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_jax_idx(idx) -> Array:
    """Normalise a panel index argument to a 1-D int32 jax array."""
    if idx is None:
        return jnp.asarray([], dtype=jnp.int32)
    arr = jnp.asarray(idx, dtype=jnp.int32).reshape(-1)
    return arr


def _clamp_params(model: Any, max_val: float) -> Any:
    """Per-element clamp of every array leaf in ``model`` to [-max, max]."""
    params, static = eqx.partition(model, eqx.is_array)
    params = jax.tree.map(lambda p: jnp.clip(p, -max_val, max_val), params)
    return eqx.combine(params, static)


def _components_to_record(components: dict, exclude_keys=("per_tf_mse",)) -> dict:
    """Convert a JAX components dict into a float-only logging record.

    Per-element arrays (like ``per_tf_mse``) are dropped from the record so
    it can be appended to ``loss_history`` cheaply.
    """
    out = {}
    for k, v in components.items():
        if k in exclude_keys:
            continue
        out[k] = float(jnp.asarray(v))
    return out


# ---------------------------------------------------------------------------
# Single training step (JIT-compiled inside train_hg_neural_ode)
# ---------------------------------------------------------------------------


def _make_step_fn(
    *,
    forward: Callable[[Any], Array],
    target: Array,
    driver_idx: Array,
    stress_idx: Array,
    target_mask: Array | None,
    config: TrainingConfig,
    optimizer: optax.GradientTransformation,
):
    """Build the JIT-compilable single-step function.

    ``forward(model) -> pred`` so the training step is decoupled from
    diffrax / hypergraph plumbing -- the caller wires it in.
    """

    def loss_fn(model):
        pred = forward(model)
        return hg_ode_loss(
            pred=pred,
            target=target,
            model=model,
            driver_idx=driver_idx,
            stress_idx=stress_idx,
            mask=target_mask,
            w_rollout=config.w_rollout,
            w_driver=config.w_driver,
            w_stress=config.w_stress,
            w_smooth=config.w_smooth,
            w_uniformity=config.w_uniformity,
            w_reg=config.w_reg,
            stress_margin=config.stress_margin,
        )

    @eqx.filter_jit
    def step_fn(model, opt_state):
        (loss, components), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        new_model = _clamp_params(new_model, config.max_param_value)
        return new_model, new_opt_state, loss, components

    return step_fn


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_hg_neural_ode(
    model: Any,
    *,
    forward: Callable[[Any], Array],
    target: Array,
    driver_idx: Array | None = None,
    stress_idx: Array | None = None,
    target_mask: Array | None = None,
    config: TrainingConfig | None = None,
    tracker: Any = None,
) -> TrainingResult:
    """Fit ``model`` so that ``forward(model) ~= target``.

    The plumbing (building the temporal hypergraph, choosing ``t0``/``t1``,
    selecting ``SaveAt`` timepoints) is the caller's responsibility -- this
    function only needs a ``forward`` closure that returns the prediction
    given the current model.  That keeps the trainer agnostic to whether
    the model is a ``HypergraphNeuralODE``, an ``hgx.HypergraphNeuralSDE``,
    or something custom.

    Args:
        model: Initial eqx model (e.g. ``hgx.HypergraphNeuralODE(conv)``).
        forward: ``model -> pred`` closure.  ``pred`` should have shape
            ``(T, N, D)`` matching ``target``.
        target: ``(T, N, D)`` observed trajectory at the held-out times.
        driver_idx: optional 1-D index array selecting driver TFs in the
            node axis (``N``).
        stress_idx: optional 1-D index array selecting stress responders.
        target_mask: optional ``(T, N)`` validity mask for the target.
        config: :class:`TrainingConfig` (a fresh default is used if None).
        tracker: object with ``.log(dict)`` -- e.g. trackio.  Optional.

    Returns:
        :class:`TrainingResult` with the fitted model, full loss history,
        the panel indices used, and final per-TF MSE.
    """
    if config is None:
        config = TrainingConfig()

    driver_jax = _to_jax_idx(driver_idx)
    stress_jax = _to_jax_idx(stress_idx)

    # Optax stack matches bl1: clip global-norm, Adam, skip-on-NaN.
    base_optim = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.learning_rate),
    )
    optimizer = optax.apply_if_finite(
        base_optim,
        max_consecutive_errors=config.max_consecutive_nan_steps,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    step_fn = _make_step_fn(
        forward=forward,
        target=target,
        driver_idx=driver_jax,
        stress_idx=stress_jax,
        target_mask=target_mask,
        config=config,
        optimizer=optimizer,
    )

    if tracker is not None:
        tracker.log({
            "config/n_epochs": config.n_epochs,
            "config/learning_rate": config.learning_rate,
            "config/w_rollout": config.w_rollout,
            "config/w_driver": config.w_driver,
            "config/w_stress": config.w_stress,
            "config/w_smooth": config.w_smooth,
            "config/w_uniformity": config.w_uniformity,
            "config/w_reg": config.w_reg,
            "config/stress_margin": config.stress_margin,
            "config/max_param_value": config.max_param_value,
            "config/grad_clip_norm": config.grad_clip_norm,
            "panel/n_driver": int(driver_jax.size),
            "panel/n_stress": int(stress_jax.size),
        })

    print(
        f"Training HG-Neural-ODE: {config.n_epochs} epochs, "
        f"LR={config.learning_rate}, grad_clip={config.grad_clip_norm}, "
        f"|drivers|={driver_jax.size}, |stress|={stress_jax.size}"
    )
    print("-" * 80)

    loss_history: list[dict] = []
    last_components: dict | None = None
    nan_epochs = 0
    t0 = time.time()

    for epoch in range(config.n_epochs):
        epoch_t0 = time.time()
        model, opt_state, loss, components = step_fn(model, opt_state)

        loss_val = float(loss)
        if not jnp.isfinite(loss):
            nan_epochs += 1
        else:
            nan_epochs = 0  # reset on a clean epoch
            last_components = components

        record = _components_to_record(components)
        record["epoch"] = epoch
        record["wall_time_s"] = time.time() - epoch_t0
        loss_history.append(record)

        if tracker is not None:
            tracker.log({
                "train/loss": loss_val,
                "train/rollout_mse": record["rollout_mse"],
                "train/driver_mse": record["driver_mse"],
                "train/stress_hinge": record["stress_hinge"],
                "train/stress_mse": record["stress_mse"],
                "train/trajectory_smoothness": record["trajectory_smoothness"],
                "train/uniformity": record["uniformity"],
                "train/param_l2": record["param_l2"],
                "train/epoch_time_s": record["wall_time_s"],
                "train/nan_epochs": nan_epochs,
            })

        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            tail = " [NaN-protected]" if nan_epochs > 0 else ""
            print(
                f"Epoch {epoch:4d}/{config.n_epochs} | "
                f"loss={loss_val:9.5f} | "
                f"roll={record['rollout_mse']:8.5f} | "
                f"drv={record['driver_mse']:8.5f} | "
                f"strs={record['stress_hinge']:7.5f} "
                f"(mse={record['stress_mse']:6.4f}) | "
                f"smth={record['trajectory_smoothness']:7.5f} | "
                f"unif={record['uniformity']:7.5f} | "
                f"reg={record['param_l2']:6.3f} | "
                f"{record['wall_time_s']:.2f}s" + tail
            )

        if nan_epochs >= config.max_consecutive_nan_steps:
            print(
                f"\nWARNING: {nan_epochs} consecutive non-finite epochs. "
                "Stopping early."
            )
            break

    total_time = time.time() - t0
    print("-" * 80)
    print(f"Training complete in {total_time:.1f}s ({len(loss_history)} epochs)")

    final_per_tf_mse: Array | None = None
    if last_components is not None and "per_tf_mse" in last_components:
        final_per_tf_mse = jnp.asarray(last_components["per_tf_mse"])

    final = loss_history[-1] if loss_history else {}
    if final:
        print(f"Final loss: {final.get('total', float('nan')):.5f}")
        print(
            f"  rollout_mse={final.get('rollout_mse', float('nan')):.5f}  "
            f"driver_mse={final.get('driver_mse', float('nan')):.5f}  "
            f"stress_mse={final.get('stress_mse', float('nan')):.5f}"
        )

    if tracker is not None and final:
        tracker.log({
            "final/loss": final.get("total"),
            "final/rollout_mse": final.get("rollout_mse"),
            "final/driver_mse": final.get("driver_mse"),
            "final/stress_mse": final.get("stress_mse"),
            "final/uniformity": final.get("uniformity"),
            "final/n_epochs_completed": len(loss_history),
            "final/total_time_s": total_time,
        })

    return TrainingResult(
        model=model,
        loss_history=loss_history,
        config=config,
        driver_idx=driver_jax if driver_jax.size > 0 else None,
        stress_idx=stress_jax if stress_jax.size > 0 else None,
        final_per_tf_mse=final_per_tf_mse,
        nan_epochs=nan_epochs,
        wall_time_s=total_time,
    )
