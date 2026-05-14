#!/usr/bin/env bash
# run_fm_real_dgx.sh — orchestrate real-mode FM extractor runs on DGX Spark.
#
# Runs each extractor inside the anatomical-compiler/fm:26.04 container
# (built from Dockerfile.fm), which carries geneformer, scgpt, UCE, evo,
# and borzoi-pytorch on top of nvcr.io/nvidia/pytorch:26.04-py3. The host
# venv stays clean — the JAX pipeline downstream only consumes the .npy
# cache the container writes.
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
# Step 4 (optional): scGPT in-silico KD perturbation prior — Lab 6/8 BO loop.
#
# Total GPU peak: ~32 GB at any one time (each tool serialised). Easily
# fits on DGX Spark's 128 GB GPU. Each tool gets its own per-step log;
# failure of one does not abort the rest.
#
# Prerequisites — see docs/dgx-spark-setup.md and Dockerfile.fm.
#
# Usage:
#   ./scripts/run_fm_real_dgx.sh <h5ad> <edges.csv> <promoters.fa> [TFS.txt] [OUT_DIR]
#
# Example:
#   ./scripts/run_fm_real_dgx.sh data/pollen.h5ad data/fleck_edges.csv \
#       /data/mhough/refs/promoters_hg38_2kb.fa data/tfs.txt cache/dgx_real_pollen
#
# Environment overrides (all optional):
#   FM_IMAGE           container image tag (default: anatomical-compiler/fm:26.04)
#   HF_HOME            HuggingFace cache on host (default: /data/mhough/cache/huggingface)
#   GENEFORMER_MODEL   model dir/handle (default: ctheodoris/Geneformer)
#   SCGPT_MODEL_DIR    path to scGPT checkpoint dir (required for scgpt real)
#   UCE_MODEL_LOC      path to UCE .torch checkpoint (default: auto-downloaded)
#   UCE_SPECIES        species string for UCE (default: human)
#   JASPAR_MEME        path to JASPAR MEME file (required for motif real)

set -euo pipefail

H5AD="${1:?usage: run_fm_real_dgx.sh <h5ad> <edges.csv> <promoters.fa> [TFS.txt] [OUT_DIR]}"
EDGES="${2:?missing edges.csv}"
PROMOTERS="${3:?missing promoters.fa}"
TFS="${4:-}"  # optional; if absent, step 4 is skipped
OUT="${5:-cache/dgx_real_$(date +%Y%m%d-%H%M%S)}"

IMAGE="${FM_IMAGE:-anatomical-compiler/fm:26.04}"
HF_HOME_HOST="${HF_HOME:-/data/mhough/cache/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

mkdir -p "$OUT/logs" "$HF_HOME_HOST"

echo "fm-real-dgx: IMAGE=$IMAGE"
echo "fm-real-dgx: H5AD=$H5AD"
echo "fm-real-dgx: EDGES=$EDGES"
echo "fm-real-dgx: PROMOTERS=$PROMOTERS"
[[ -n "$TFS" ]] && echo "fm-real-dgx: TFS=$TFS"
echo "fm-real-dgx: OUT=$OUT"
echo "fm-real-dgx: HF_HOME=$HF_HOME_HOST"
echo

# fm_run — invoke the FM container with the standard bind-mounts + env.
# Identity-mounts both the project tree and /data/mhough so paths inside
# the container match the host (no path translation needed in args).
fm_run() {
    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --user "$(id -u):$(id -g)" \
        -v "$PROJECT_ROOT":"$PROJECT_ROOT" \
        -v /data/mhough:/data/mhough \
        -w "$PROJECT_ROOT" \
        -e HOME=/tmp \
        -e HF_HOME="$HF_HOME_HOST" \
        -e UCE_REPO="${UCE_REPO:-/opt/UCE}" \
        -e UCE_MODEL_LOC="${UCE_MODEL_LOC:-}" \
        -e UCE_SPECIES="${UCE_SPECIES:-human}" \
        -e GENEFORMER_MODEL="${GENEFORMER_MODEL:-ctheodoris/Geneformer}" \
        -e SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-}" \
        -e JASPAR_MEME="${JASPAR_MEME:-}" \
        "$IMAGE" \
        "$@"
}

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
    run_step "embed_${model}" fm_run python3 scripts/fm_embed.py \
        "$model" --input "$H5AD" --output "$OUT" --mode real
done

# Step 3 — edge priors (motif first, no GPU; then GPU models)
for model in motif evo borzoi; do
    run_step "edges_${model}" fm_run python3 scripts/fm_edges_seq.py \
        "$model" --edges "$EDGES" --promoters "$PROMOTERS" \
        --output "$OUT" --mode real
done

# Step 3b — edge-prior ablation (host-side, no GPU; runs in container anyway
# to keep all numpy/scipy versions consistent with the producers)
run_step "ablation_edges" fm_run python3 scripts/ablate_edge_priors.py \
    --output "$OUT/ablation_edges"

# Step 4 — scGPT in-silico KD perturbation prior (Lab 6/8 BO loop)
if [[ -n "$TFS" ]]; then
    run_step "perturb_scgpt" fm_run python3 scripts/fm_perturb_scgpt.py \
        --input "$H5AD" --tfs "$TFS" --output "$OUT" --mode real
    run_step "ablation_perturb_eig" fm_run python3 scripts/ablate_perturb_eig.py \
        --output "$OUT/ablation_perturb_eig"
else
    echo "skipping step 4 (no TFs file passed)"
fi

# Summary
echo "── summary ──────────────────────────────────────"
echo "outputs in $OUT:"
ls -la "$OUT" | grep -v '^d' | grep -v '^total'
echo
echo "to consume from notebooks/11_foundation_model_pipeline.ipynb,"
echo "set the cache path to '$OUT' and re-run."
