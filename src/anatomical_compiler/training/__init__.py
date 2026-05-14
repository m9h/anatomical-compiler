"""Differentiable training for Hypergraph Neural ODEs on TF trajectories.

Mirrors the structure of bl1.training: a frozen ``TrainingConfig``, a
``TrainingResult`` container, a multi-component differentiable loss, and a
NaN-protected JIT-compiled training loop.
"""

from anatomical_compiler.training.loss import (
    hg_ode_loss,
    rollout_mse,
    driver_mse,
    stress_hinge,
    trajectory_smoothness,
    per_tf_mse_uniformity,
    parameter_l2,
)
from anatomical_compiler.training.trainer import (
    TrainingConfig,
    TrainingResult,
    train_hg_neural_ode,
)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "train_hg_neural_ode",
    "hg_ode_loss",
    "rollout_mse",
    "driver_mse",
    "stress_hinge",
    "trajectory_smoothness",
    "per_tf_mse_uniformity",
    "parameter_l2",
]
