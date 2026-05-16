#!/usr/bin/env bash
#SBATCH --job-name=fm_perturb_scgpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x.%j.out
#SBATCH --error=slurm/logs/%x.%j.err
#SBATCH --time=00:30:00
#
# fm_perturb_scgpt — scGPT in-silico knockdown for Lab 6 / Lab 8 priors.
# Waits for the out-of-band Evo run (PID set via $WAIT_FOR_PID) to release
# the GPU before claiming it. Allocated by slurm but sleeps via kill -0 poll
# until the gate condition is met; then runs the FM container.
#
# Submit:
#   WAIT_FOR_PID=$(pgrep -f 'fm_edges_seq.py evo' | head -1) sbatch slurm/fm_perturb_scgpt.sbatch
# or without gate (run immediately):
#   sbatch slurm/fm_perturb_scgpt.sbatch

set -euo pipefail

mkdir -p slurm/logs

echo "[$(date +%H:%M:%S)] slurm job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "[$(date +%H:%M:%S)] gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"

PID="${WAIT_FOR_PID:-}"
if [ -n "$PID" ]; then
    echo "[$(date +%H:%M:%S)] waiting for PID $PID to release the GPU…"
    while kill -0 "$PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] PID $PID exited; proceeding"
fi

cd /home/mhough/dev/anatomical-compiler
exec docker run --rm \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -v "$PWD":"$PWD" -v /data/mhough:/data/mhough \
    -w "$PWD" -e HOME=/tmp \
    -e HF_HOME=/data/mhough/cache/huggingface \
    -e SCGPT_MODEL_DIR=/data/mhough/cache/huggingface/hub/models--perturblab--scgpt-human/snapshots/571a0445d68fa48381f863ff75dd4f6d0eae3dfc \
    anatomical-compiler/fm:26.04 \
    python3 scripts/fm_perturb_scgpt.py \
        --input data/pollen_slice_geneformer.h5ad \
        --tfs data/pollen_tfs.txt \
        --output cache/real_run \
        --mode real
