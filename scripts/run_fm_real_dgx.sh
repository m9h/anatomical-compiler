#!/usr/bin/env bash
# run_fm_real_dgx.sh — orchestrate real-mode FM extractor runs on DGX Spark.
#
# Runs the three foundation-model directions end-to-end on real checkpoints,
# producing a single cache directory the downstream notebooks consume.
#
# Step 1+2: per-gene and per-cell node priors (scripts/fm_embed.py)
#   uce ............ 1280-d cell embedding (Stanford+CZI; ~32 GB FP16)
#   geneformer ..... 512-d gene embedding  (UCSF/Gladstone; ~4 GB FP16)
#   scgpt .......... 512-d cell embedding  (Toronto/Vector; ~2 GB FP16)
#
# Step 3: sequence-grounded edge priors (scripts/fm_edges_seq.py)
#   motif .......... JASPAR/CIS-BP PWM scan (no GPU)
#   evo ............ Evo log-likelihood    (Arc Institute; ~32 GB FP16)
#   borzoi ......... Borzoi Δ-track score  (Google + Kundaje; ~16 GB FP16)
#
# Total GPU peak: ~32 GB at any one time (each tool serialised). Easily
# fits on DGX Spark's 128 GB GPU. The script runs each tool sequentially
# and writes errors to per-step logs; failure of one tool does not abort.
#
# Prerequisites — see docs/dgx-spark-setup.md.
#
# Usage:
#   ./scripts/run_fm_real_dgx.sh <h5ad> <edges.csv> <promoters.fa> [OUT_DIR]
#
# Example:
#   ./scripts/run_fm_real_dgx.sh data/pollen.h5ad data/fleck_edges.csv \
#       data/promoters_hg38_2kb.fa cache/dgx_real_pollen

set -euo pipefail

H5AD="${1:?usage: run_fm_real_dgx.sh <h5ad> <edges.csv> <promoters.fa> [OUT_DIR]}"
EDGES="${2:?missing edges.csv}"
PROMOTERS="${3:?missing promoters.fa}"
OUT="${4:-cache/dgx_real_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT/logs"
echo "fm-real-dgx: H5AD=$H5AD"
echo "fm-real-dgx: EDGES=$EDGES"
echo "fm-real-dgx: PROMOTERS=$PROMOTERS"
echo "fm-real-dgx: OUT=$OUT"
echo

# Path to project root regardless of where the script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

run_step() {
    local name="$1"; shift
    echo "── $name ────────────────────────────────────────"
    if "$@" 2>&1 | tee "$OUT/logs/${name}.log"; then
        echo "✓ $name done"
    else
        echo "✗ $name FAILED — see $OUT/logs/${name}.log"
    fi
    echo
}

# Step 1+2 — node features (each model gets its own GPU pass)
for model in uce geneformer scgpt; do
    run_step "embed_${model}" python3 "$PROJECT_ROOT/scripts/fm_embed.py" \
        "$model" --input "$H5AD" --output "$OUT" --mode real
done

# Step 3 — edge priors (motif first, no GPU; then GPU models)
for model in motif evo borzoi; do
    run_step "edges_${model}" python3 "$PROJECT_ROOT/scripts/fm_edges_seq.py" \
        "$model" --edges "$EDGES" --promoters "$PROMOTERS" \
        --output "$OUT" --mode real
done

# Step 3b — ablation on whatever the run produced
run_step "ablation" python3 "$PROJECT_ROOT/scripts/ablate_edge_priors.py" \
    --output "$OUT/ablation"

# Summary
echo "── summary ──────────────────────────────────────"
echo "outputs in $OUT:"
ls -la "$OUT" | grep -v '^d' | grep -v '^total'
echo
echo "to consume from notebooks/11_foundation_model_pipeline.ipynb,"
echo "set the cache path to '$OUT' and re-run."
