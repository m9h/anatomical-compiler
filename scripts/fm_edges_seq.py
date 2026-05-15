"""fm_edges_seq — sequence-grounded edge priors for the regulome.

Step 3 of docs/foundation-models.md. Companion to scripts/fm_embed.py:
where fm_embed.py extracts *node* features (per-cell or per-gene
embeddings), fm_edges_seq.py scores the *edges* of the regulome graph
from promoter sequence + TF binding likelihood.

Three subcommands, three intentionally complementary substrates:

motif     PWM/motif-scan baseline. Real-mode uses JASPAR / CIS-BP motifs
          and a sliding-window scan; stub-mode is a deterministic hash-
          based score that reproduces dimensionally. The cheapest, most
          honest sequence prior — no neural net at all.
evo       Evo 2 (Arc Institute; Nguyen, Hie et al.). DNA language model;
          scores TF-binding likelihood as a conditional probability on
          the promoter sequence with the TF motif as a prompt. Real-mode
          loads the HF checkpoint; stub-mode is a different deterministic
          projection (scaled to look like a log-likelihood).
borzoi    Borzoi (Linder, Kelley et al.; Google + Kundaje, Stanford).
          Sequence → per-cell-type expression track predictor. Real-mode
          scores edges as "predicted expression change when TF binding
          site is masked vs unmasked"; stub-mode mimics with deterministic
          noise.

Output contract (same as fm_embed.py): writes <stem>_<model>.npy with
shape (n_edges,) and a matching manifest JSON. Consumer code does not
distinguish between real and stub mode beyond the manifest field.

Usage
-----
    python scripts/fm_edges_seq.py motif  --edges edges.csv --output cache/
    python scripts/fm_edges_seq.py evo    --edges edges.csv --promoters prom.fa --output cache/
    python scripts/fm_edges_seq.py borzoi --edges edges.csv --promoters prom.fa --output cache/

`edges.csv` has columns ``tf,target`` (gene symbols or Ensembl IDs).
`promoters.fa` is required for real-mode `evo` and `borzoi`; not needed for
the stub paths nor for `motif --mode stub`.

See also
--------
- docs/foundation-models.md §3 — the intervention this implements.
- scripts/fm_embed.py — node-feature priors.
- scripts/ablate_edge_priors.py — ablation on a synthetic-truth benchmark.
- scripts/run_fm_real_dgx.sh — real-mode driver for DGX Spark (128 GB GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------
# Configuration — per-model provenance and shape
# ----------------------------------------------------------------------------

MODELS = {
    "motif": {
        "scale": "log-odds",
        "citation": "JASPAR / CIS-BP PWM scan (Kheradpour & Kellis; Mathelier et al.)",
        "checkpoint": "JASPAR2024_CORE_vertebrates_non-redundant.meme",
        "dgx_gb": 0,
    },
    "evo": {
        "scale": "log-likelihood",
        "citation": "Nguyen, Hie et al. 2024 (Arc Institute, Stanford-adjacent)",
        "checkpoint": "togethercomputer/evo-1-131k-base",  # or evo2-* once released
        "dgx_gb": 32,  # FP16 1B param Evo on a ~2kb promoter context
    },
    "borzoi": {
        "scale": "delta-track-prediction",
        "citation": "Linder, Kelley et al. 2025 (Google + Kundaje, Stanford)",
        # HF mirror by johahi (Calico-permitted port). `calico/borzoi-v1`
        # doesn't exist on HF.
        "checkpoint": "johahi/borzoi-replicate-0",
        "dgx_gb": 16,  # Borzoi v1 in FP16
    },
}


# ----------------------------------------------------------------------------
# Edge list loading
# ----------------------------------------------------------------------------


def _load_edges(path: Path):
    """Load an edges CSV with columns tf, target. Returns a list of (tf, target) tuples."""
    import csv

    edges: list[tuple[str, str]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tf = row.get("tf") or row.get("TF") or row.get("regulator")
            tg = row.get("target") or row.get("gene")
            if not tf or not tg:
                continue
            edges.append((tf.strip(), tg.strip()))
    if not edges:
        raise ValueError(f"no (tf, target) edges parsed from {path}")
    return edges


def _load_promoters(path: Path | None) -> dict[str, str]:
    """Load a FASTA of promoter sequences. Keys = gene symbol; values = nucleotide string.

    Returns empty dict if path is None (stub-mode tolerates missing sequences).
    """
    if path is None:
        return {}
    seqs: dict[str, str] = {}
    with open(path, "r") as f:
        current_id = None
        current_chunks: list[str] = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_chunks).upper()
                current_id = line[1:].split()[0]
                current_chunks = []
            else:
                current_chunks.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(current_chunks).upper()
    return seqs


# ----------------------------------------------------------------------------
# Stub backends — deterministic, reproducible, dimensionally correct
# ----------------------------------------------------------------------------


def _edge_seed(tf: str, target: str, salt: int) -> int:
    """Deterministic 32-bit seed from (tf, target, salt). Stable across runs / machines."""
    h = hashlib.sha256(f"{tf}|{target}|{salt}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _stub_motif(edges, promoters, seed: int) -> np.ndarray:
    """Stub PWM-scan score: a deterministic per-edge log-odds-shaped score.

    Real motif scanning produces a small number of high-scoring matches per
    edge (sparse, right-skewed distribution). The stub mimics the *shape*
    of that distribution — Gumbel-distributed scores seeded from the edge
    identity — without doing any actual scanning.
    """
    scores = np.empty(len(edges), dtype=np.float32)
    for i, (tf, target) in enumerate(edges):
        rng = np.random.default_rng(_edge_seed(tf, target, seed))
        # Gumbel(0, 1) — heavy right tail, matches real PWM hit distribution.
        u = rng.uniform(1e-9, 1.0)
        scores[i] = -np.log(-np.log(u))
    return scores


def _stub_evo(edges, promoters, seed: int) -> np.ndarray:
    """Stub Evo log-likelihood: deterministic, on a different scale from motif.

    Real Evo log-likelihoods range roughly [-12, -4] per 100bp of context;
    the stub samples Gaussian on that scale so consumers can see the
    distinct distributional shape vs `motif`.
    """
    scores = np.empty(len(edges), dtype=np.float32)
    for i, (tf, target) in enumerate(edges):
        rng = np.random.default_rng(_edge_seed(tf, target, seed) ^ 0xE0)
        scores[i] = rng.normal(loc=-8.0, scale=1.5)
    return scores


def _stub_borzoi(edges, promoters, seed: int) -> np.ndarray:
    """Stub Borzoi Δ-track score: deterministic, signed.

    Real Borzoi delta-expression scores are signed (TF binding can up- or
    down-regulate). Stub uses Laplace(0, 0.5) which matches the empirical
    distribution shape.
    """
    scores = np.empty(len(edges), dtype=np.float32)
    for i, (tf, target) in enumerate(edges):
        rng = np.random.default_rng(_edge_seed(tf, target, seed) ^ 0xB0)
        scores[i] = rng.laplace(loc=0.0, scale=0.5)
    return scores


_STUBS = {"motif": _stub_motif, "evo": _stub_evo, "borzoi": _stub_borzoi}


# ----------------------------------------------------------------------------
# Real backends — guarded; raise informatively if deps / weights missing
# ----------------------------------------------------------------------------


def _real_motif(edges, promoters, seed: int) -> np.ndarray:
    """Real PWM scan: best-hit log-odds over the target promoter.

    Lazy-imports `Bio.motifs` (Biopython) and reads JASPAR motifs from
    $JASPAR_MEME (or a sensible default location). Falls back to a clear
    error if anything's missing.
    """
    try:
        from Bio import motifs  # type: ignore
        from Bio.Seq import Seq  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "motif real-mode requires Biopython: pip install biopython. "
            "Fall back to --mode stub for tutorial use."
        ) from e
    import os
    meme_path = os.environ.get(
        "JASPAR_MEME", "/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme"
    )
    if not Path(meme_path).exists():
        raise RuntimeError(
            f"motif real-mode needs JASPAR MEME file at {meme_path} (or $JASPAR_MEME env)."
        )
    # JASPAR's *.meme.txt is the MEME minimal text format. Biopython's
    # "minimal" parser captures the PWM but only stores the accession
    # (e.g. "MA0004.1") as motif.name — the gene symbol on the same MOTIF
    # line (e.g. "Arnt" in "MOTIF MA0004.1 Arnt") is dropped. Parse that
    # mapping separately and index motifs by gene symbol so edges' tf field
    # (which holds the gene symbol) can match.
    parsed = list(motifs.parse(open(meme_path), "minimal"))
    acc_to_gene: dict[str, str] = {}
    with open(meme_path) as f:
        for line in f:
            if line.startswith("MOTIF "):
                tok = line.split()
                if len(tok) >= 3:
                    acc_to_gene[tok[1]] = tok[2]
    jdb: dict[str, Any] = {}
    for m in parsed:
        gene = acc_to_gene.get(m.name, "")
        # Heterodimer motifs are written e.g. "RXRA::VDR"; index each component
        for sym in gene.replace("::", " ").split():
            jdb.setdefault(sym.upper(), m)

    scores = np.full(len(edges), -np.inf, dtype=np.float32)
    n_scored = 0
    n_missing_prom = 0
    n_missing_motif = 0
    n_calc_error = 0
    last_err = ""
    for i, (tf, target) in enumerate(edges):
        if target not in promoters:
            n_missing_prom += 1
            continue
        if tf.upper() not in jdb:
            n_missing_motif += 1
            continue
        # Drop ambiguous bases — Bio.motifs PWM scanning can't handle Ns;
        # replace with A so scoring proceeds (these positions just won't be PWM hits).
        clean_seq = promoters[target].translate(str.maketrans("Nn", "AA"))
        seq = Seq(clean_seq)
        # Score with PSSM (log-odds) — PWM is just the matrix; PSSM is what
        # has .calculate() in biopython's API.
        pssm = jdb[tf.upper()].pssm
        try:
            arr = pssm.calculate(seq)
            best = float(np.nanmax(arr)) if hasattr(arr, "__len__") and len(arr) else float("-inf")
        except Exception as e:
            n_calc_error += 1
            last_err = f"{type(e).__name__}: {e}"
            continue
        scores[i] = best
        n_scored += 1
    print(
        f"_real_motif: matched {len(jdb)} gene symbols to motifs; "
        f"scored {n_scored}/{len(edges)} edges "
        f"(missing_prom={n_missing_prom}, missing_motif={n_missing_motif}, "
        f"calc_err={n_calc_error}{'; last_err=' + last_err if last_err else ''})",
        file=sys.stderr,
    )
    return scores


def _real_evo(edges, promoters, seed: int) -> np.ndarray:
    """Real Evo conditional log-likelihood of TF-motif consensus given promoter.

    Scores each edge with log P(motif_consensus | promoter) under Evo-1, where
    motif_consensus is the per-position argmax of the JASPAR PWM for the TF.
    A more positive score means the model assigns higher likelihood to the
    motif sequence given the promoter context — interpretable as "this promoter
    is consistent with this TF's binding motif under the DNA LM."

    Requires:
        - $JASPAR_MEME (same file as the motif backend) to look up TF consensus
        - GPU with ≥ 32 GB for the 7B parameter model in bf16
    """
    try:
        from evo import Evo  # type: ignore
        import torch  # type: ignore
        from Bio import motifs  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Evo real-mode requires `evo-model`, `torch`, and `biopython`. "
            "Available in the FM container (Dockerfile.fm)."
        ) from e

    if not promoters:
        raise RuntimeError("Evo real-mode requires --promoters FASTA")

    import os
    meme_path = os.environ.get(
        "JASPAR_MEME", "/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme"
    )
    if not Path(meme_path).exists():
        raise RuntimeError(
            f"Evo real-mode needs JASPAR MEME at {meme_path} (for TF motif consensus)"
        )

    # Reuse the same JASPAR → gene-symbol mapping as _real_motif.
    parsed = list(motifs.parse(open(meme_path), "minimal"))
    acc_to_gene: dict[str, str] = {}
    with open(meme_path) as f:
        for line in f:
            if line.startswith("MOTIF "):
                tok = line.split()
                if len(tok) >= 3:
                    acc_to_gene[tok[1]] = tok[2]
    consensus: dict[str, str] = {}
    for m in parsed:
        gene = acc_to_gene.get(m.name, "")
        cons = str(m.consensus)
        for sym in gene.replace("::", " ").split():
            consensus.setdefault(sym.upper(), cons)

    # Load Evo
    device = os.environ.get("EVO_DEVICE", "cuda")
    model_name = os.environ.get("EVO_MODEL", "evo-1-131k-base")
    print(f"_real_evo: loading {model_name} on {device}…", file=sys.stderr)
    evo_obj = Evo(model_name)
    model = evo_obj.model.to(device)
    tokenizer = evo_obj.tokenizer
    model.eval()

    scores = np.full(len(edges), np.nan, dtype=np.float32)
    n_scored = 0
    n_skipped = 0
    # Cap promoter to a reasonable context so we don't blow GPU memory.
    max_ctx = int(os.environ.get("EVO_MAX_CTX", "2048"))
    print(f"_real_evo: scoring {len(edges)} edges (max_ctx={max_ctx})…", file=sys.stderr)
    for i, (tf, target) in enumerate(edges):
        if target not in promoters or tf.upper() not in consensus:
            n_skipped += 1
            continue
        prompt = promoters[target][-max_ctx:]  # last max_ctx bp ending at TSS
        motif_seq = consensus[tf.upper()]
        full = prompt + motif_seq
        try:
            ids = torch.tensor(tokenizer.tokenize(full), dtype=torch.int, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(ids)  # (1, L, V)
            # Conditional log-likelihood of motif tokens given the promoter prefix.
            # logits[:, t-1, :] predicts token at position t; so log P(token_t | prefix)
            # = log_softmax(logits[t-1])[token_t].
            n_prompt = len(tokenizer.tokenize(prompt))
            motif_ids = ids[0, n_prompt:]
            shift_logits = logits[0, n_prompt - 1 : n_prompt - 1 + len(motif_ids), :]
            lp = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
            ll = lp.gather(-1, motif_ids.long().unsqueeze(-1)).squeeze(-1)
            scores[i] = float(ll.mean().item())
            n_scored += 1
        except Exception as e:
            if i < 3:
                print(f"_real_evo: edge {i} ({tf}->{target}) error: {type(e).__name__}: {e}",
                      file=sys.stderr)
            continue
        if (i + 1) % 1000 == 0:
            print(f"_real_evo: {i+1}/{len(edges)} done (scored {n_scored})",
                  file=sys.stderr)

    print(
        f"_real_evo: scored {n_scored}/{len(edges)} edges; skipped {n_skipped}",
        file=sys.stderr,
    )
    return scores


def _real_borzoi(edges, promoters, seed: int) -> np.ndarray:
    """Real Borzoi Δ-track score: |predicted_expression(with TF motif) − predicted_expression(masked)|.

    Lazy-imports `borzoi_pytorch`. On DGX Spark, sequence-context up to ~196 kb;
    ~10 edges/sec for a single track because each edge requires two forward passes.
    """
    try:
        import borzoi_pytorch  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Borzoi real-mode requires the `borzoi_pytorch` package "
            "(https://github.com/johahi/borzoi-pytorch). Fall back to --mode stub."
        ) from e
    if not promoters:
        raise RuntimeError("Borzoi real-mode requires --promoters FASTA")
    import os
    checkpoint = os.environ.get("BORZOI_CHECKPOINT", "johahi/borzoi-replicate-0")
    # TODO(real-mode): borzoi-pytorch's actual API is Borzoi.from_pretrained(<HF
    # repo>) → model takes one-hot encoded sequences of fixed length and returns
    # predictions over many cell-type tracks. A proper delta-track score would
    # mask the TF motif in the promoter sequence, run two forward passes, and
    # compute the |Δexpression| over a chosen track set. This is non-trivial
    # to do correctly. Below is a placeholder that raises so a stub doesn't
    # silently take over — finish the implementation against borzoi-pytorch's
    # actual API (johahi/borzoi-pytorch on GitHub).
    raise RuntimeError(
        f"Borzoi real-mode not yet implemented against borzoi-pytorch's actual API. "
        f"Checkpoint {checkpoint} is fetchable; the masked-vs-unmasked Δ-track "
        f"scoring still needs porting. See TODO in scripts/fm_edges_seq.py:_real_borzoi. "
        f"Fall back to --mode stub for now."
    )


_REAL = {"motif": _real_motif, "evo": _real_evo, "borzoi": _real_borzoi}


# ----------------------------------------------------------------------------
# Public API — what consumers import
# ----------------------------------------------------------------------------


def extract_edge_scores(
    edges: list[tuple[str, str]],
    promoters: dict[str, str] | None = None,
    model: str = "motif",
    mode: str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract sequence-grounded edge scores.

    Parameters
    ----------
    edges
        List of (tf_symbol, target_symbol) pairs.
    promoters
        Dict gene_symbol → nucleotide sequence. Required for real-mode `evo` /
        `borzoi`; optional for real-mode `motif` and all stub modes.
    model
        One of "motif", "evo", "borzoi".
    mode
        "real" | "stub" | "auto". Default "auto": try real, fall back to stub.
    seed
        Stub-mode reproducibility seed.

    Returns
    -------
    scores : np.ndarray, shape (n_edges,)
    manifest : dict
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    spec = MODELS[model]
    promoters = promoters or {}

    used_mode = mode
    if mode == "real":
        scores = _REAL[model](edges, promoters, seed)
    elif mode == "stub":
        scores = _STUBS[model](edges, promoters, seed)
    elif mode == "auto":
        try:
            scores = _REAL[model](edges, promoters, seed)
            used_mode = "real"
        except RuntimeError as e:
            warnings.warn(f"fm_edges_seq: falling back to stub mode ({model}): {e}")
            scores = _STUBS[model](edges, promoters, seed)
            used_mode = "stub"
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from real/stub/auto")

    manifest = {
        "model": model,
        "mode": used_mode,
        "scale": spec["scale"],
        "n_edges": int(len(edges)),
        "n_promoters_provided": int(len(promoters)),
        "citation": spec["citation"],
        "checkpoint": spec["checkpoint"],
        "dgx_estimated_gb": spec["dgx_gb"],
        "seed": seed,
        "fm_edges_seq_version": "0.1.0",
    }
    return scores, manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cmd_extract(args) -> int:
    edges_path = Path(args.edges)
    if not edges_path.exists():
        print(f"error: edges CSV not found: {edges_path}", file=sys.stderr)
        return 2
    promoters_path = Path(args.promoters) if args.promoters else None
    if promoters_path is not None and not promoters_path.exists():
        print(f"error: promoters FASTA not found: {promoters_path}", file=sys.stderr)
        return 2
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"fm_edges_seq: loading edges from {edges_path}")
    edges = _load_edges(edges_path)
    print(f"fm_edges_seq: {len(edges)} edges loaded")

    print(f"fm_edges_seq: loading promoters from {promoters_path or '(none)'}")
    promoters = _load_promoters(promoters_path)
    print(f"fm_edges_seq: {len(promoters)} promoters loaded")

    scores, manifest = extract_edge_scores(
        edges, promoters=promoters, model=args.model, mode=args.mode, seed=args.seed
    )
    manifest["edges_path"] = str(edges_path)
    if promoters_path is not None:
        manifest["promoters_path"] = str(promoters_path)

    stem = edges_path.stem
    out_npy = output_dir / f"{stem}_{args.model}.npy"
    out_manifest = output_dir / f"{stem}_{args.model}_manifest.json"
    np.save(out_npy, scores)
    out_manifest.write_text(json.dumps(manifest, indent=2))

    finite = np.isfinite(scores)
    print(
        f"fm_edges_seq: wrote {out_npy}  shape={scores.shape}  "
        f"finite={finite.sum()}/{scores.size}  mean={scores[finite].mean():.3f}  "
        f"std={scores[finite].std():.3f}"
    )
    print(f"fm_edges_seq: wrote {out_manifest}  mode={manifest['mode']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="model", required=True)
    for name in MODELS:
        sp = sub.add_parser(name, help=f"{name} — {MODELS[name]['citation']}")
        sp.add_argument("--edges", required=True, help="CSV with tf,target columns")
        sp.add_argument(
            "--promoters", default=None, help="FASTA of promoter sequences (real-mode for evo/borzoi)"
        )
        sp.add_argument("--output", required=True, help="output directory")
        sp.add_argument("--mode", choices=["real", "stub", "auto"], default="auto")
        sp.add_argument("--seed", type=int, default=0)
        sp.set_defaults(func=_cmd_extract)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
