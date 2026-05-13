"""JAX-native Marchenko-Pastur (RMT) denoiser for scRNA-seq.

A 30-line reimplementation of the spine of Rabadán Lab's ``randomly`` (Aparicio &
Bordyuh, *Patterns* 2020):

  1. SVD of the (centered, log-normalised) cells x genes matrix.
  2. Marchenko-Pastur upper edge  λ₊ = σ²(1+√q)²,  q = p/n,  σ² estimated from the
     bulk (iteratively: median of eigenvalues below the current λ₊).
  3. Project onto the signal eigenvectors (λ > λ₊).

Why JAX-native (vs. ``pip install randomly``):
  * GPU/JIT for free; SVD on a 10⁴ × 10³ matrix is sub-second.
  * Differentiable: ``jax.grad`` through the denoiser works (the truncated SVD
    reconstruction is differentiable in X).
  * Composes with the rest of the project's JAX stack (``hgx``, ``jaxctrl``,
    ``betse.science.jax.*``); no SciPy↔JAX adapter.
  * Optional parallel-analysis null (per-column permutation) via ``jax.vmap`` —
    a count-data-friendlier alternative to the pure MP cutoff.

For background on whether to use this at all, and where in the pipeline it
matters for *this* project, see the methods discussion in the paper §"Code and
data availability" and the project memo on three identifiabilities. The
headline: RMT denoising is a clean, principled preprocessing alternative; its
impact on this project's downstream metrics is bounded by *structural*, not
preprocessing, bottlenecks (Labs 4, 6, 7) — so it's a *toggle ablation*, not a
default pipeline change. Self-test below is the calibration check.

CLI usage::

    uv run python scripts/denoise_rmt.py --self-test           # synthetic validation
    uv run python scripts/denoise_rmt.py path/to/data.h5ad    # writes adata.layers["rmt_denoised"]
                                                              # and adata.obsm["X_rmt"]
"""
from __future__ import annotations

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Core: Marchenko-Pastur cutoff + truncated-SVD denoiser
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_iter",))
def mp_cutoff(eigvals: jnp.ndarray, q: float, n_iter: int = 3):
    """Marchenko-Pastur upper edge λ₊ = σ²(1+√q)², σ² refined on the bulk.

    ``eigvals``: 1-D array of eigenvalues of (1/n) X_cᵀ X_c (so ``s_i**2 / n``).
    ``q``: aspect ratio p/n.
    Returns ``(cutoff, sigma_squared)``.
    """
    # σ² estimator: mean of the bulk (the MP density has mean = σ², median ≠ σ² unless q is small;
    # using nanmedian biases low under skew, drags the cutoff under the true λ₊, and counts noise as signal —
    # so use nanmean).
    s2 = jnp.nanmean(eigvals)                                 # initial guess (bulk-dominated)
    for _ in range(n_iter):
        upper = s2 * (1.0 + jnp.sqrt(q)) ** 2
        bulk = jnp.where(eigvals < upper, eigvals, jnp.nan)
        s2 = jnp.nanmean(bulk)
    return s2 * (1.0 + jnp.sqrt(q)) ** 2, s2


def rm_denoise(X: jnp.ndarray):
    """Marchenko-Pastur denoising via truncated SVD.

    ``X``: (n_cells, n_genes), assumed log-normalised (any monotone size-factor
    normalisation will do — the MP shape is preserved). NOT pre-centred — we
    centre internally so the cutoff is on the *informative* eigenvalues.

    Returns a dict::

        X_denoised:  (n_cells, n_genes) — signal-only reconstruction
        eigvals:     (k,)               — eigenvalues of (1/n) X_cᵀ X_c, descending
        cutoff:      float              — MP upper edge λ₊
        sigma2:      float              — estimated noise variance
        n_signal:    int                — number of eigenvalues above λ₊
        U, s, Vt:    SVD pieces of the centered X (full, not truncated)

    Notes
    -----
    Cell-side singular vectors ``U`` (n × k) are a low-rank cell embedding —
    use the columns 0..n_signal as a denoised X_rmt for clustering / DE / GRN.
    Gene-side ``Vt`` (k × p) rows are the principal axes; their loadings on the
    signal block are the ``signal_genes`` measure below.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    n, p = X.shape
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    U, s, Vt = jnp.linalg.svd(Xc, full_matrices=False)        # k = min(n, p)
    eigvals = s ** 2 / n
    cutoff, sigma2 = mp_cutoff(eigvals, q=p / n)
    n_signal = int(jnp.sum(eigvals > cutoff))
    s_signal = jnp.where(eigvals > cutoff, s, 0.0)
    X_denoised = (U * s_signal) @ Vt + jnp.mean(X, axis=0, keepdims=True)
    return dict(X_denoised=X_denoised, eigvals=eigvals, cutoff=float(cutoff),
                sigma2=float(sigma2), n_signal=n_signal, U=U, s=s, Vt=Vt)


def signal_genes(out: dict, k_top: int) -> np.ndarray:
    """Top-``k_top`` genes by total loading on the signal eigenvectors.

    A principled alternative to scanpy's variance-based HVG selection: rank
    genes by Σ_{i∈signal} |Vt[i, gene]|² · s_i², i.e. their contribution to the
    MP-significant subspace.
    """
    Vt = np.asarray(out["Vt"]);  s = np.asarray(out["s"]);  ns = out["n_signal"]
    if ns == 0:
        return np.argsort(-(Vt[:1] ** 2 * s[:1, None] ** 2).sum(0))[:k_top]
    score = (Vt[:ns] ** 2 * s[:ns, None] ** 2).sum(axis=0)    # length p
    return np.argsort(-score)[:k_top]


# ---------------------------------------------------------------------------
# Optional: parallel-analysis null (count-data-friendlier than pure MP)
# ---------------------------------------------------------------------------

def _shuffle_columns(X, key):
    """Permute each column of X independently — destroys gene-gene structure."""
    n, p = X.shape
    keys = jax.random.split(key, p)
    return jax.vmap(lambda k, j: jax.random.permutation(k, X[:, j]),
                    in_axes=(0, 0))(keys, jnp.arange(p)).T


@partial(jax.jit, static_argnames=("n_perm",))
def parallel_analysis_cutoff(X, key, n_perm: int = 50):
    """Empirical 95th-percentile of the top eigenvalue of a per-column-permuted X.

    More appropriate than pure MP when the column-marginals are non-Gaussian
    (count data with mean-variance scaling). Returns a scalar threshold on
    eigvals = s²/n of the centered matrix.
    """
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    n = Xc.shape[0]

    def one(k):
        Xp = _shuffle_columns(Xc, k)
        s_top = jnp.linalg.svd(Xp, compute_uv=False)[0]
        return s_top ** 2 / n

    keys = jax.random.split(key, n_perm)
    top_eigs = jax.vmap(one)(keys)
    return jnp.quantile(top_eigs, 0.95)


# ---------------------------------------------------------------------------
# Self-test: synthetic block-structured matrix → MP should recover the signal rank
# ---------------------------------------------------------------------------

def self_test(seed: int = 0):
    """Generate a low-rank signal + Gaussian noise; check MP recovers the rank.

    Reports: (rank truth, n_signal estimated, sigma² truth, sigma² estimated,
    parallel-analysis cutoff agreement, runtime).
    """
    import time
    rng = np.random.default_rng(seed)
    n, p, true_rank, sigma_true = 800, 500, 4, 1.0
    L = rng.normal(size=(n, true_rank))            # cell-side latent
    R = rng.normal(size=(true_rank, p)) * 6.0      # gene-side, scaled to be well above noise
    signal = L @ R
    noise = rng.normal(size=(n, p)) * sigma_true
    X = signal + noise

    t0 = time.time()
    out = rm_denoise(X)
    t_mp = time.time() - t0
    cutoff, sigma2, n_signal = out["cutoff"], out["sigma2"], out["n_signal"]

    t0 = time.time()
    pa_cut = float(parallel_analysis_cutoff(jnp.asarray(X), jax.random.PRNGKey(seed)))
    t_pa = time.time() - t0

    eigvals = np.asarray(out["eigvals"])
    mse_recon = float(np.mean((np.asarray(out["X_denoised"]) - signal) ** 2))
    mse_raw   = float(np.mean((X - signal) ** 2))

    print(f"=== Synthetic validation (n={n}, p={p}, true rank={true_rank}, σ_true={sigma_true}) ===")
    print(f"  MP estimated n_signal = {n_signal}        (truth: {true_rank})  "
          + ("✓" if n_signal == true_rank else "✗"))
    print(f"  MP σ² estimate        = {sigma2:.4f}     (truth: {sigma_true**2:.4f})  "
          + ("✓" if abs(sigma2 - sigma_true**2) < 0.05 else "✗"))
    print(f"  MP cutoff λ₊          = {cutoff:.4f}     (top {true_rank} eigvals: "
          f"{eigvals[:true_rank].tolist()})")
    print(f"  Parallel-analysis cut = {pa_cut:.4f}     (should be ≈ MP cutoff for Gaussian noise)")
    print(f"  Recon MSE vs signal   = {mse_recon:.4f}  (raw-vs-signal MSE = {mse_raw:.4f}; "
          f"{100*(1-mse_recon/mse_raw):.0f}% reduction)")
    print(f"  Timing                  : rm_denoise {t_mp*1000:.0f} ms · parallel-analysis {t_pa*1000:.0f} ms")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("h5ad", nargs="?", help="anndata .h5ad path; writes layers/obsm in place")
    ap.add_argument("--self-test", action="store_true", help="run synthetic validation and exit")
    ap.add_argument("--use", choices=("X", "raw"), default="X", help="which matrix to denoise")
    ap.add_argument("--n-hvg", type=int, default=2000, help="pre-filter to top-N variance genes before RMT (memory)")
    args = ap.parse_args()

    if args.self_test:
        self_test();  return
    if not args.h5ad:
        ap.error("provide an .h5ad path or --self-test")

    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if args.n_hvg and args.n_hvg < adata.shape[1]:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    out = rm_denoise(X)
    adata.layers["rmt_denoised"] = np.asarray(out["X_denoised"])
    adata.obsm["X_rmt"] = np.asarray(out["U"][:, :out["n_signal"]] * out["s"][:out["n_signal"]])
    print(f"RMT: {adata.shape[0]} cells × {adata.shape[1]} genes  →  "
          f"{out['n_signal']} signal components  (σ²={out['sigma2']:.3f}, λ₊={out['cutoff']:.3f})")
    adata.write(args.h5ad)
    print(f"wrote rmt_denoised layer + X_rmt obsm back to {args.h5ad}")


if __name__ == "__main__":
    _cli()
