#!/usr/bin/env bash
#SBATCH --job-name=borzoi_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:%j_will_be_replaced%
#
# borzoi_smoke — Borzoi real-mode smoke test on the first 100 edges.
# Validates the rewrite in scripts/fm_edges_seq.py:_real_borzoi (johahi
# HF mirror, marginal-promoter-activity scoring). Runs only after the
# perturb job's GPU window frees (set via --dependency=afterany:<jobid>
# at submit time).
#
# Submit (after the perturb job is queued):
#   sbatch --dependency=afterany:$PERTURB_JOBID slurm/borzoi_smoke.sbatch

set -euo pipefail

mkdir -p slurm/logs cache/borzoi_smoke
echo "[$(date +%H:%M:%S)] slurm job $SLURM_JOB_ID on $SLURMD_NODENAME"

# 100-edge smoke set
head -101 data/pollen_edges.csv > /tmp/edges_100.csv

cd /home/mhough/dev/anatomical-compiler
exec docker run --rm \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -v "$PWD":"$PWD" -v /data/mhough:/data/mhough -v /tmp:/host_tmp \
    -w "$PWD" -e HOME=/tmp \
    -e HF_HOME=/data/mhough/cache/huggingface \
    -e BORZOI_CHECKPOINT=johahi/borzoi-replicate-0 \
    anatomical-compiler/fm:26.04 \
    python3 scripts/fm_edges_seq.py borzoi \
        --edges /host_tmp/edges_100.csv \
        --promoters /data/mhough/refs/promoters_hg38_2kb.fa \
        --output cache/borzoi_smoke \
        --mode real
