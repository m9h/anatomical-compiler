"""Network control theory on the Pando regulome — the "anatomical compiler" layer.

Where the rest of the suite builds the *plant model* (Hypergraph Neural ODE),
the *system identification* (SBI / CellFlow inverse, the Pando GRN itself) and
the *observability / decomposition* layer (the Hodge-Laplacian Module
Identifiability Index), this script adds the missing *controller-synthesis*
layer using `jaxctrl` (https://github.com/m9h/jaxctrl):

  1. Controllability of the master-regulator set.  Build a linear network
     dynamics x' = A x + B u on the TF co-regulation graph (A = -L of the
     weighted TF-TF adjacency, so A is Hurwitz; B selects driver TFs).  Ask:
     is the curated key-TF set a *controlling* input set for the whole TF
     network (Kalman rank)?  How many driver nodes are needed at minimum?

  2. Control leverage per TF.  For each TF i, the finite-horizon controllability
     Gramian W_i = W(A, e_i, T) summarises how easily a single-TF intervention
     steers the network: trace(W_i) ~ "average controllability", lambda_min(W_i)
     ~ "modal / boundary controllability" (Gu et al. 2015).  Rank TFs by both.

  3. Steer-to-target ("anatomical compiler").  Take the early-pseudotime TF
     state x0 and the late-pseudotime TF state xf from temporal_expression.npy,
     and compute (a) the minimum control energy to drive x0 -> xf using the
     key-TF set vs a random equal-size set vs all TFs, and (b) an LQR feedback
     law steering toward xf, plus the closed-loop trajectory.  This is the
     toy version of "desired anatomy in -> intervention out".

The script is data- and dependency-agnostic: it reads data/processed/ artifacts
(from scripts/00_preprocess.py); if those or `jaxctrl` are absent it prints a
note and exits cleanly, like the other benchmark_*.py scripts.

Refs: Liu, Slotine & Barabasi 2011 (Nature, structural controllability of
complex networks); Gu et al. 2015 (Nat Commun, controllability of structural
brain networks); Pasqualetti, Zampieri & Bullo 2014 (controllability metrics);
Yan et al. 2017 (Nature, network control principles); Pezzulo & Levin 2016 /
Levin 2022 (the "anatomical compiler" framing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = Path("data/processed")


def _load_processed():
    """Load the Pando incidence + TF metadata + temporal expression."""
    req = ["incidence.npy", "tf_names.json", "gene_names.json"]
    if not all((PROC / f).exists() for f in req):
        return None
    inc = np.load(PROC / "incidence.npy").astype(np.float32)          # (genes, regulons)
    tf_names = json.loads((PROC / "tf_names.json").read_text())        # len = regulons
    gene_names = json.loads((PROC / "gene_names.json").read_text())    # len = genes
    tf_gene = {}
    if (PROC / "tf_gene_indices.json").exists():
        tf_gene = json.loads((PROC / "tf_gene_indices.json").read_text())
    key_tf = {}
    if (PROC / "key_tf_indices.json").exists():
        key_tf = json.loads((PROC / "key_tf_indices.json").read_text())
    temporal = None
    if (PROC / "temporal_expression.npy").exists():
        temporal = np.load(PROC / "temporal_expression.npy")          # (bins, genes)
    return dict(inc=inc, tf_names=tf_names, gene_names=gene_names,
                tf_gene=tf_gene, key_tf=key_tf, temporal=temporal)


def _tf_network(inc: np.ndarray, n_tf_keep: int = 200, knn: int = 8):
    """Sparse weighted TF-TF co-regulation network from the gene x regulon incidence.

    Pando regulons overlap heavily (the all-pairs shared-target matrix is nearly
    dense and low-rank), so for a well-conditioned linear control model we keep,
    for each regulon, only its `knn` strongest **Jaccard-overlap** partners
    (symmetrised) among the `n_tf_keep` regulons with the most targets.  We then
    take A = -(L_sym + eps I) where L_sym is the symmetric-normalised Laplacian
    of that sparse graph -- A is Hurwitz, so finite-horizon Gramians exist.
    Returns (A, kept_idx, degree).
    """
    target_counts = inc.sum(axis=0)
    kept = np.argsort(-target_counts)[:n_tf_keep]
    H = inc[:, kept].astype(np.float32)                   # (genes, k)
    inter = H.T @ H                                       # (k, k) shared targets
    sizes = np.diag(inter).copy()
    union = sizes[:, None] + sizes[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(J, 0.0)
    k = len(kept)
    knn = max(1, min(knn, k - 1))
    W = np.zeros_like(J)
    for i in range(k):
        nbr = np.argpartition(-J[i], knn)[:knn]
        W[i, nbr] = J[i, nbr]
    W = np.maximum(W, W.T)                                 # symmetrise
    deg = W.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-9), 0.0)
    A_norm = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L = np.eye(k, dtype=np.float32) - A_norm
    A = -(L + 1e-3 * np.eye(k, dtype=np.float32))
    return A.astype(np.float32), kept, deg.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tfs", type=int, default=200, help="top-N regulons (by target count) to keep")
    ap.add_argument("--horizon", type=float, default=4.0, help="control horizon T")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-fig", default="figures/network_control.png")
    ap.add_argument("--out-json", default="figures/network_control_results.json")
    args = ap.parse_args()

    print("Network control theory on the Pando regulome (jaxctrl)...")

    data = _load_processed()
    if data is None:
        print(f"  Error: {PROC}/ artifacts not found. Run scripts/00_preprocess.py first.")
        return

    try:
        import jax
        import jax.numpy as jnp
        import jaxctrl
        HAS_JAXCTRL = True
    except Exception as e:  # pragma: no cover
        HAS_JAXCTRL = False
        print(f"  jaxctrl not available ({e!r}). It is a project dependency -- run `uv sync` "
              f"(or `pip install -e ../jaxctrl`).")

    A_np, kept, deg = _tf_network(data["inc"], n_tf_keep=args.n_tfs)
    kept_names = [data["tf_names"][i] for i in kept]
    n = A_np.shape[0]
    print(f"  TF network: {n} regulons (top-{args.n_tfs} by target count); "
          f"mean degree {deg.mean():.1f}")

    # --- driver set: curated key TFs that survived the top-N filter --------------
    key_names = list(data["key_tf"].keys())
    # map original-regulon key TFs onto the kept-index space, by name
    name_to_local = {nm: j for j, nm in enumerate(kept_names)}
    drivers_local = sorted(name_to_local[nm] for nm in key_names if nm in name_to_local)
    if not drivers_local:  # fall back to highest-degree TFs as drivers
        drivers_local = sorted(np.argsort(-deg)[:8].tolist())
    driver_names = [kept_names[j] for j in drivers_local]
    print(f"  Driver TFs ({len(drivers_local)}): {driver_names}")

    results = {
        "n_tfs": n, "n_drivers": len(drivers_local), "driver_tfs": driver_names,
        "horizon": args.horizon, "jaxctrl": HAS_JAXCTRL,
        "refs": ["Liu-Slotine-Barabasi 2011", "Gu et al. 2015",
                 "Pasqualetti-Zampieri-Bullo 2014", "Yan et al. 2017",
                 "Pezzulo & Levin 2016 (anatomical compiler)"],
    }

    if not HAS_JAXCTRL:
        results["status"] = "jaxctrl_unavailable"
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"  Wrote {args.out_json} (driver-set summary only).")
        return

    A = jnp.asarray(A_np)
    eye = jnp.eye(n)
    B_drivers = eye[:, jnp.asarray(drivers_local)]                  # (n, m)
    T = float(args.horizon)

    # --- 1. controllability of the master-regulator set -------------------------
    C_d = jaxctrl.controllability_matrix(A, B_drivers)             # (n, n*m)
    s = jnp.linalg.svd(C_d, compute_uv=False)
    rank_d = int((s > s[0] * n * jnp.finfo(C_d.dtype).eps).sum())
    ok_d = bool(jaxctrl.is_controllable(A, B_drivers))
    print(f"  Kalman controllability of the TF network from the {len(drivers_local)}-TF "
          f"driver set: rank {rank_d}/{n} (fully controllable = {ok_d})")
    results["controllability"] = {
        "key_tf_set_rank": rank_d, "key_tf_set_controllable": ok_d, "n": n,
        "n_drivers": len(drivers_local),
    }
    # minimum number of driver nodes (structural; needs an hgx Hypergraph)
    try:
        import hgx
        from jaxctrl import minimum_driver_nodes
        hg = hgx.from_incidence(jnp.asarray(data["inc"][:, kept]))   # genes x kept-regulons
        results["controllability"]["min_driver_nodes_hgx"] = int(minimum_driver_nodes(hg))
        print(f"  Minimum driver nodes (structural, hgx): "
              f"{results['controllability']['min_driver_nodes_hgx']}")
    except Exception as e:
        results["controllability"]["min_driver_nodes_hgx"] = None
        print(f"  (min_driver_nodes via hgx skipped: {e!r})")

    # --- 2. control leverage per TF (single-input Gramians) ---------------------
    @jax.jit
    def single_node_gramian(i):
        Bi = eye[:, i][:, None]
        return jaxctrl.controllability_gramian(A, Bi, T=T)
    avg_ctrl = np.empty(n); mod_ctrl = np.empty(n)
    for i in range(n):
        W = single_node_gramian(i)
        avg_ctrl[i] = float(jnp.trace(W))
        mod_ctrl[i] = float(jnp.linalg.eigvalsh(W)[0])               # smallest eigenvalue
    order_avg = np.argsort(-avg_ctrl)
    top = [{"tf": kept_names[i], "avg_controllability": float(avg_ctrl[i]),
            "modal_controllability": float(mod_ctrl[i]), "degree": float(deg[i]),
            "is_driver": bool(i in set(drivers_local))} for i in order_avg[:20]]
    results["control_leverage_top20"] = top
    print("  Top control-leverage TFs (avg controllability): "
          + ", ".join(f"{t['tf']}" for t in top[:8]))

    # --- 3. steer-to-target (early-pseudotime -> late-pseudotime TF state) ------
    steer = {"available": False}
    temporal = data["temporal"]
    if temporal is not None and temporal.ndim == 2 and temporal.shape[0] >= 2:
        # TF-state vectors at the first and last pseudotime bin (z-scored over bins)
        gene_idx_for_kept = []
        ok = True
        for nm in kept_names:
            gi = data["tf_gene"].get(nm)
            if gi is None or gi >= temporal.shape[1]:
                ok = False; break
            gene_idx_for_kept.append(int(gi))
        if ok:
            E = temporal[:, np.asarray(gene_idx_for_kept)]           # (bins, n)
            E = (E - E.mean(0, keepdims=True)) / (E.std(0, keepdims=True) + 1e-6)
            x0 = jnp.asarray(E[0]); xf = jnp.asarray(E[-1])
            # min control energy to steer x0 -> xf per input set, with the smallest
            # controllability-Gramian eigenvalue as a steerability/conditioning flag
            # (lam_min ~ 0  =>  the input set barely reaches the target direction, so the
            #  "energy" is dominated by an ill-conditioned inverse and is not reliable).
            def steer_metrics(Bsel):
                W = jaxctrl.controllability_gramian(A, Bsel, T=T)
                lam_min = float(jnp.linalg.eigvalsh(W)[0])
                e = float(jaxctrl.minimum_energy(A, Bsel, x0, xf, T))
                # None => the controllability Gramian is (numerically) singular for this
                # input set, i.e. it does not span the target direction; "energy" undefined.
                e = abs(e) if (np.isfinite(e) and lam_min > 1e-9) else None
                return e, lam_min
            rng = np.random.default_rng(args.seed)
            rand_local = sorted(rng.choice(n, size=len(drivers_local), replace=False).tolist())
            e_key, lam_key = steer_metrics(B_drivers)
            e_rand, lam_rand = steer_metrics(eye[:, jnp.asarray(rand_local)])
            e_full, lam_full = steer_metrics(eye)
            # LQR steer-to-target: y = x - xf, dy/dt ~= A y + B u  =>  cost-to-go y0^T X y0
            Q = jnp.eye(n)
            y0 = x0 - xf
            K_key, X_key = jaxctrl.lqr(A, B_drivers, Q, 1e-2 * jnp.eye(len(drivers_local)))
            K_full, X_full = jaxctrl.lqr(A, eye, Q, 1e-2 * eye)   # well-posed (full actuation)
            steer = {"available": True,
                     "energy_key_tf_inputs": e_key,
                     "energy_random_inputs": e_rand,
                     "energy_full_inputs": e_full,
                     "gramian_min_eig_key_tf": lam_key,
                     "gramian_min_eig_random": lam_rand,
                     "gramian_min_eig_full": lam_full,
                     "lqr_cost_to_go_key_tf": float(y0 @ X_key @ y0),
                     "lqr_cost_to_go_full": float(y0 @ X_full @ y0),
                     "lqr_gain_norm_key_tf": float(jnp.linalg.norm(K_key)),
                     "lqr_gain_norm_full": float(jnp.linalg.norm(K_full)),
                     "random_input_set": [kept_names[j] for j in rand_local]}
            _e = lambda v: ("uncontrollable" if v is None else f"{v:.3g}")
            print(f"  Steer early->late TF state:  E*(key TFs)={_e(e_key)} (lam_min={lam_key:.1e})  "
                  f"E*(random {len(drivers_local)})={_e(e_rand)} (lam_min={lam_rand:.1e})  "
                  f"E*(all TFs)={_e(e_full)} (lam_min={lam_full:.1e})")
            print(f"  LQR cost-to-go from x0:  key-TF inputs {steer['lqr_cost_to_go_key_tf']:.3g} "
                  f"(||K||={steer['lqr_gain_norm_key_tf']:.1f})   full actuation "
                  f"{steer['lqr_cost_to_go_full']:.3g} (||K||={steer['lqr_gain_norm_full']:.1f})")
            # closed-loop trajectory: regulate the deviation y = x - xf to 0 under u = -K y
            # on the linearised dynamics (FULL actuation -- the well-posed case), then x = y + xf
            try:
                ts, ys, us = jaxctrl.simulate_closed_loop(
                    A, eye, K_full, x0=(x0 - xf), T=T, num_steps=60)
                xs_traj = np.asarray(ys) + np.asarray(xf)[None, :]
                pick = drivers_local[:4]
                steer["traj_ts"] = np.asarray(ts).tolist()
                steer["traj_tfs"] = [kept_names[j] for j in pick]
                steer["traj"] = xs_traj[:, np.asarray(pick)].tolist()
                steer["traj_target"] = [float(xf[j]) for j in pick]
                steer["traj_actuation"] = "full"
            except Exception as e:
                print(f"  (closed-loop sim skipped: {e!r})")
    results["steer_to_target"] = steer
    results["status"] = "ok"

    # --- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k = min(15, len(top))
    names = [t["tf"] for t in top[:k]][::-1]
    vals = [t["avg_controllability"] for t in top[:k]][::-1]
    colors = ["#d62728" if t["is_driver"] else "#1f77b4" for t in top[:k]][::-1]
    axes[0].barh(names, vals, color=colors)
    axes[0].set_title("Control leverage per TF\n(avg controllability = trace of single-input Gramian; red = curated driver)")
    axes[0].set_xlabel(f"avg controllability  (T={T})")
    if steer.get("available") and "traj" in steer:
        tr = np.asarray(steer["traj"]); tgt = steer.get("traj_target", [])
        cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for j, nm in enumerate(steer["traj_tfs"]):
            axes[1].plot(steer["traj_ts"], tr[:, j], color=cyc[j % len(cyc)], label=nm)
            if j < len(tgt):
                axes[1].axhline(tgt[j], color=cyc[j % len(cyc)], ls=":", lw=1)
        axes[1].set_title("Closed-loop steer toward late-pseudotime TF state\n"
                          "(LQR u = -K(x - x_f), full actuation; dotted = target x_f)")
        axes[1].set_xlabel("time"); axes[1].set_ylabel("TF state (z)"); axes[1].legend(fontsize=8)
    else:
        labels = ["key TFs", f"random ({len(drivers_local)})", "all TFs"]
        evs = [steer.get("energy_key_tf_inputs"), steer.get("energy_random_inputs"),
               steer.get("energy_full_inputs")] if steer.get("available") else [0, 0, 0]
        axes[1].bar(labels, [0 if (e is None or not np.isfinite(e)) else e for e in evs],
                    color=["#d62728", "#7f7f7f", "#2ca02c"])
        axes[1].set_title("Min control energy: early -> late TF state\n(by input set; 0 bar = uncontrollable in that subspace)")
        axes[1].set_ylabel("E*  (lower = easier to steer)")
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"  Figure -> {args.out_fig}")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"  Results -> {args.out_json}")


if __name__ == "__main__":
    main()
