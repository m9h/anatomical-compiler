"""PCA node features with automatic dimensionality estimation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PCAResult:
    features: np.ndarray       # (n_genes, d) float32
    eigenvalues: np.ndarray    # descending, all
    dim: int                   # chosen dimensionality
    method: str                # how dim was chosen
    explained_variance: float  # fraction explained by chosen dim
    details: dict              # method-specific details (k_aic, k_bic, etc.)

def compute_pca(
    expression: np.ndarray,
    *,
    n_subset: int = 5000,
    dim: int | None = None,
    dim_method: str = "ppca",
    variance_threshold: float = 0.90,
    max_dim: int = 200,
    seed: int = 42,
) -> PCAResult:
    """Compute gene-level PCA features from a (genes x cells) expression matrix.

    dim_method:
        "ppca" — PPCA/MELODIC AIC+BIC consensus (from neurojax)
        "variance" — choose k for variance_threshold explained variance
        "fixed" — use dim directly
    """
    n_genes, n_cells = expression.shape
    rng = np.random.default_rng(seed)

    # Subsample cells
    n_sub = min(n_subset, n_cells)
    sub_idx = rng.choice(n_cells, size=n_sub, replace=False)
    subset = expression[:, sub_idx].astype(np.float64)

    # Gene-gene covariance + eigendecomposition
    cov = np.cov(subset)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = eigvals[::-1]

    # Explained variance
    total_var = eigvals_desc.sum()
    cumvar = np.cumsum(eigvals_desc) / total_var

    details = {}

    if dim_method == "fixed":
        if dim is None:
            raise ValueError("dim must be specified for dim_method='fixed'")
        effective_dim = dim
        details["method"] = "fixed"

    elif dim_method == "variance":
        k_var = int(np.searchsorted(cumvar, variance_threshold) + 1)
        effective_dim = max(min(k_var, max_dim), 2)
        details["method"] = "variance"
        details["threshold"] = variance_threshold
        details["k_raw"] = k_var

    elif dim_method == "ppca":
        # PPCA/MELODIC dimensionality estimation
        d_max = len(eigvals_desc)
        lambdas = np.maximum(eigvals_desc, 1e-19)
        log_lam = np.log(lambdas)
        cs_log = np.cumsum(log_lam)
        cs_lam = np.cumsum(lambdas)
        sum_noise = cs_lam[-1] - cs_lam

        all_k = np.arange(1, d_max)
        m_noise = d_max - all_k
        v = np.maximum(sum_noise[all_k - 1] / m_noise, 1e-19)
        log_lik = -(n_sub / 2) * (cs_log[all_k - 1] + m_noise * np.log(v))
        m_params = d_max * all_k - 0.5 * all_k * (all_k + 1)

        aic = log_lik - m_params
        bic = log_lik - 0.5 * m_params * np.log(n_sub)
        k_aic = int(np.argmax(aic) + 1)
        k_bic = int(np.argmax(bic) + 1)
        k_consensus = int(round((k_aic + k_bic) / 2))

        effective_dim = max(min(k_consensus, max_dim), 2)
        details["method"] = "ppca"
        details["k_aic"] = k_aic
        details["k_bic"] = k_bic
        details["k_consensus"] = k_consensus
    else:
        raise ValueError(f"Unknown dim_method: {dim_method}")

    # Variance info
    k_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    k_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    details["k_90pct"] = k_90
    details["k_95pct"] = k_95

    node_features = eigvecs[:, -effective_dim:].astype(np.float32)
    explained = float(cumvar[effective_dim - 1]) if effective_dim <= len(cumvar) else 1.0

    return PCAResult(
        features=node_features,
        eigenvalues=eigvals_desc.astype(np.float32),
        dim=effective_dim,
        method=dim_method,
        explained_variance=explained,
        details=details,
    )
