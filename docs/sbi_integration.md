# SBI & CellFlow Integration

This document describes the Simulation-Based Inference (SBI) pipeline using [CellFlow](https://github.com/m9h/cellflow) to invert biological perturbations and infer regulatory mechanisms.

## Overview

Traditional GRN inference (like Pando) uses correlation or regression to find links. Here, we treat the regulatory network as a dynamical system:
1.  **Forward**: A TF knockout changes the velocity field of cell state.
2.  **Inverse**: We observe the change in cell distribution (e.g., via CRISPRi screen) and infer the underlying velocity field.

## Pipeline

### 1. Training (CellFlow)

We use **Flow Matching** to learn the generative mapping from a control population to a perturbed population.

```bash
# Example usage
uv run scripts/12_pollen_inverse_sbi.py --h5ad data/pollen/screen.h5ad
```

### 2. Jacobian Attribution

Once the model learns the velocity field $V(X, \text{condition})$, we compute the Jacobian $J = \frac{dV}{dX}$.

-   $J_{ij} > 0$: Gene $i$ positively regulates the rate of change of gene $j$.
    -   $J_{ij} < 0$: Gene $i$ inhibits the rate of change of gene $j$.

This provides a "live" regulatory matrix that reflects the system's response to specific perturbations.

## Validation

We validate the inferred Jacobian against the **Fleck et al. Pando GRN**:
-   Do the top entries in the Jacobian correspond to known Pando edges?
-   Does the sign of the Jacobian match the predicted direction from `validate_against_pando.py`?

## Demo

A simplified demonstration is available in `scripts/10_cellflow_sbi_demo.py`. This script simulates a GRN propagation using `hgx` and then learns to recover it using `CellFlow`.

```bash
uv run scripts/10_cellflow_sbi_demo.py
```

## Running Tests

To verify the integration, run:

```bash
uv run pytest tests/test_sbi.py
```
