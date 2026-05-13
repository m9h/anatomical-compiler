"""Ablation: does RMT denoising change the Module Identifiability Index?

Mirrors ``benchmark_kidney_modularity.py``'s pipeline (gene-gene HVG-correlation
hypergraph → Hodge L0 spectrum → MII) on the same three systems
(bioprinted kidney / brain organoid / fetal kidney), but adds a ``--rmt`` arm
that replaces variance-based HVG selection + raw correlation with the
JAX-native Marchenko-Pastur denoiser (`scripts/denoise_rmt.py`):

  * select top-``n_hvg`` genes by **signal-eigenvector loading** (rather than
    by variance), on the MP-significant subspace;
  * compute gene-gene correlation on the **denoised** (signal-only) matrix,
    not on the raw normalised one.

Then compares MII / Fiedler value / n_signal-eigvals across both arms, on each
system. The user-stated drop threshold is **|ΔMII| < 0.05** → not worth
adopting; per the project note that this is a probable outcome (RMT works on
a clean noise model, the bottleneck for MII is structural not noise-driven).

Run::

    uv run python scripts/ablate_rmt_kidney_modularity.py            # both arms, all three systems
    uv run python scripts/ablate_rmt_kidney_modularity.py --n-cells 1500
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import scanpy as sc
import jax, jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
from denoise_rmt import rm_denoise, signal_genes


def _hodge_l0_from_corr(X_for_corr: np.ndarray, knn: int = 10) -> dict:
    """Given an (n_cells, n_genes) matrix, build the HVG top-knn correlation incidence,
    compute the Hodge L0 spectrum (dense; n_genes is small here). Returns ev0/fiedler/n_nodes."""
    corr = np.corrcoef(X_for_corr.T)
    n = corr.shape[0]
    incidence = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        top = np.argsort(np.abs(corr[i]))[-(knn + 1):]
        incidence[top, i] = 1.0
    # clique-expansion L0 (consistent with how Lab 4 computes it from the same incidence)
    A = incidence @ incidence.T
    np.fill_diagonal(A, 0.0)
    L0 = np.diag(A.sum(1)) - A
    ev0 = np.sort(np.linalg.eigvalsh(L0))
    fiedler = float(ev0[ev0 > 1e-6][0]) if np.any(ev0 > 1e-6) else 0.0
    return dict(ev0=ev0, fiedler=fiedler, n_nodes=n)


def mii_heuristic(spectrum: np.ndarray, n: int = 10) -> float:
    """The project's MII (Lab 4 / scripts/test_nitmb_modularity.py, verbatim)."""
    s = np.sort(spectrum)
    s = s[s > 1e-6]                                # drop the harmonic null (matches the kidney benchmark)
    if len(s) < 2: return 0.0
    return float(np.mean(np.diff(s)[:n]) / (np.std(s[:n]) + 1e-8))


def run_baseline(adata, n_hvg: int = 100) -> dict:
    """The original benchmark's path: variance-HVG → corr on raw normalised X → Hodge L0."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    X = adata[:, adata.var.highly_variable].X
    if hasattr(X, "toarray"): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    out = _hodge_l0_from_corr(X)
    out["arm"] = "baseline (variance-HVG, raw corr)"
    return out


def run_rmt(adata, n_hvg: int = 100, n_pool: int = 2000) -> dict:
    """RMT arm: top-n_pool variance pre-filter for memory, then RMT denoise, then top-n_hvg
    by signal-eigenvector loading, then corr on the *denoised* (signal-only) matrix."""
    # pre-filter to a manageable gene pool for the SVD (n_cells × n_pool)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_pool)
    Xp = adata[:, adata.var.highly_variable].X
    if hasattr(Xp, "toarray"): Xp = Xp.toarray()
    Xp = np.asarray(Xp, dtype=np.float32)
    out_rmt = rm_denoise(jnp.asarray(Xp))
    sel = signal_genes(out_rmt, n_hvg)
    X_den = np.asarray(out_rmt["X_denoised"])[:, sel]
    out = _hodge_l0_from_corr(X_den)
    out["arm"] = f"rmt (signal-eigvec-HVG, denoised corr; σ²≈{out_rmt['sigma2']:.3f}, n_signal={out_rmt['n_signal']})"
    out["rmt_n_signal"] = out_rmt["n_signal"]
    out["rmt_sigma2"] = float(out_rmt["sigma2"])
    return out


def _measure(adata, label: str, n_hvg: int, n_cells: int) -> dict:
    print(f"\n--- {label}  (subsample {n_cells}, n_hvg {n_hvg}) ---")
    a = adata[:n_cells].to_memory() if hasattr(adata, "is_view") else adata[:n_cells].copy()
    # log1p only if the matrix looks like counts; the *processed* h5ads here vary
    Xprobe = a.X[:200].toarray() if hasattr(a.X, "toarray") else np.asarray(a.X[:200])
    if Xprobe.min() >= 0.0 and (Xprobe > 20).any():
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        print("    [normalised + log1p]")
    else:
        print("    [matrix already normalised — skipping log1p]")
    t = time.time(); b = run_baseline(a.copy(), n_hvg=n_hvg);  t_b = time.time() - t
    t = time.time(); r = run_rmt(a.copy(),       n_hvg=n_hvg);  t_r = time.time() - t
    mii_b, mii_r = mii_heuristic(b["ev0"]), mii_heuristic(r["ev0"])
    print(f"    baseline : MII={mii_b:.4f}  fiedler={b['fiedler']:.3e}  ({t_b:.1f}s)")
    print(f"    rmt      : MII={mii_r:.4f}  fiedler={r['fiedler']:.3e}  "
          f"(n_signal={r.get('rmt_n_signal')}, σ²={r.get('rmt_sigma2'):.3f}; {t_r:.1f}s)")
    print(f"    ΔMII (rmt - baseline) = {mii_r - mii_b:+.4f}   |   threshold for adoption: |Δ| > 0.05")
    return dict(label=label, mii_baseline=mii_b, mii_rmt=mii_r, dMII=mii_r - mii_b,
                fiedler_baseline=b["fiedler"], fiedler_rmt=r["fiedler"],
                rmt_n_signal=r.get("rmt_n_signal"), rmt_sigma2=r.get("rmt_sigma2"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-cells", type=int, default=1000, help="cells per system (kidney benchmark uses 1000)")
    ap.add_argument("--n-hvg",   type=int, default=100,  help="HVGs in the final correlation graph")
    args = ap.parse_args()

    DATA = Path("data")
    rows = []
    # 1. bioprinted kidney
    p = DATA / "bioprinting" / "lawlor_2021_processed.h5ad"
    if p.exists():
        a = sc.read_h5ad(p)
        sc.pp.filter_cells(a, min_genes=200)
        rows.append(_measure(a, "Bioprinted Kidney (Lawlor 2021)", args.n_hvg, args.n_cells))
    else:
        print(f"[skip] {p} absent")
    # 2. brain organoid (Fleck)
    p = DATA / "zenodo" / "RNA_all_velo.h5ad"
    if p.exists():
        a = sc.read_h5ad(p, backed="r")
        rows.append(_measure(a, "Brain Organoid (Fleck 2023 RNA)", args.n_hvg, args.n_cells))
    else:
        print(f"[skip] {p} absent")
    # 3. fetal kidney reference
    p = DATA / "bioprinting" / "kidney_ref_processed.h5ad"
    if p.exists():
        a = sc.read_h5ad(p)
        rows.append(_measure(a, "Fetal Kidney (Ref)",             args.n_hvg, args.n_cells))
    else:
        print(f"[skip] {p} absent")

    print("\n" + "=" * 78)
    print(f"  {'system':>34}  {'MII (baseline)':>15}  {'MII (RMT)':>10}  {'ΔMII':>9}  {'verdict':>20}")
    print("-" * 78)
    for r in rows:
        verdict = "ADOPT" if abs(r["dMII"]) >= 0.05 else "drop (<0.05)"
        print(f"  {r['label']:>34}  {r['mii_baseline']:>15.4f}  {r['mii_rmt']:>10.4f}  {r['dMII']:>+9.4f}  {verdict:>20}")
    print("=" * 78)
    print("Committed reference (figures/nitmb_modularity_report.json, Lab 4 baseline):")
    print("  Brain organoid 0.3812 > Fetal kidney 0.3667 > Bioprinted kidney 0.3526")
    print("(Re-runs differ at the 0.01 level from subsampling / pp variation; the ΔMII row is the ablation signal.)")

    Path("figures").mkdir(exist_ok=True)
    out = {"rows": rows, "args": vars(args)}
    Path("figures/rmt_ablation_kidney_modularity.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote figures/rmt_ablation_kidney_modularity.json")


if __name__ == "__main__":
    main()
