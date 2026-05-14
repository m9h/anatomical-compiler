# anatomical-compiler — reproducible container per CURE-R (Sauro 2025, ref. 99a)
#
# Two stages:
#   baseline  — CPU-only, project + scientific stack + JAX/diffrax/optax/hgx/jaxctrl.
#               Runs every stub-mode tutorial notebook and every ablation in seconds.
#               This is the CURE-Reproducible baseline.
#   fm        — extends baseline with the foundation-model dependencies (Geneformer,
#               scGPT, UCE, Evo, Borzoi, biopython, JASPAR) so the real-mode pipeline
#               from docs/dgx-spark-setup.md is one `docker build --target fm` away.
#
# Both stages are multi-arch (linux/amd64 + linux/arm64 — DGX Spark is GH200 ARM64,
# typical CI is x86_64); uv's base image is multi-arch upstream.
#
# Build:
#   docker build -t anatomical-compiler:baseline .
#   docker build -t anatomical-compiler:fm --target fm .
#
# Run:
#   docker run --rm -it anatomical-compiler:baseline bash
#   docker run --rm --gpus all anatomical-compiler:fm bash   # for the DGX Spark side

# ----------------------------------------------------------------------------
# Stage 1 — baseline (CPU, no FM deps)
# ----------------------------------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS baseline

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# git for the sister-repo clones; build tools for any package that needs to compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The pyproject pins hgx and jaxctrl as relative paths (`../hgx`, `../jaxctrl`)
# for development convenience. In the container we pre-fetch them as sibling
# directories so the same uv.sources resolves cleanly inside the build.
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/m9h/hgx.git      hgx \
 && git clone --depth 1 https://github.com/m9h/jaxctrl.git  jaxctrl

# Copy the project last; this is the layer that changes most often, so keep it
# below the sister-repo clones for cache efficiency.
WORKDIR /workspace/anatomical-compiler
COPY pyproject.toml uv.lock ./
# Resolve deps before copying the rest — maximises cache reuse on code-only changes.
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen

# Convenience: every script + notebook expects `python3` to mean `uv run python3`.
ENV PATH="/workspace/anatomical-compiler/.venv/bin:${PATH}"

# Sanity check baked into the build: the project's headline ablation must run
# clean. If it doesn't, the image is broken and the build fails — which is
# exactly the kind of CURE-Reproducible guarantee the audit doc calls for.
RUN python3 scripts/ablate_edge_priors.py --output /tmp/dockersmoke_edges \
 && python3 scripts/ablate_perturb_eig.py --output /tmp/dockersmoke_eig \
 && rm -rf /tmp/dockersmoke_*

CMD ["bash"]

# ----------------------------------------------------------------------------
# Stage 2 — fm (real-mode foundation-model deps; for DGX Spark)
# ----------------------------------------------------------------------------

FROM baseline AS fm

# transformers + torch for the HF model loaders (Geneformer / scGPT / UCE / Evo / Borzoi all
# go through HF + a thin wrapper). torch is the heaviest dep here.
RUN uv pip install --system \
        'torch >= 2.4' \
        'transformers >= 4.42' \
        'huggingface-hub >= 0.24' \
        biopython

# The real-mode FM packages. These are pip-installable but pull tens of GB of
# checkpoint metadata; the actual weights are fetched at runtime via huggingface-cli
# (see docs/dgx-spark-setup.md §2). Each is pinned to >= a version known to work.
RUN uv pip install --system \
        'geneformer'             \
        'evo-model >= 1.1'       \
        'borzoi-pytorch'

# scGPT and UCE are editable / GitHub-only:
RUN uv pip install --system \
        'git+https://github.com/bowang-lab/scGPT'   \
        'git+https://github.com/snap-stanford/UCE'

# JASPAR motif database for the `motif` edge-prior real-mode path.
# Downloading at image-build time bakes the database into the image so air-gapped
# clusters don't need internet for the motif scan step.
RUN mkdir -p /opt/jaspar \
 && curl -fsSL 'https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt' \
        -o /opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme
ENV JASPAR_MEME=/opt/jaspar/JASPAR2024_CORE_vertebrates_non-redundant.meme

# Don't bake the HF checkpoints into the image — they're 50+ GB and you want
# them in a mounted volume that survives image rebuilds. See docs/dgx-spark-setup.md
# §2 for the `huggingface-cli download` pre-fetch block.

CMD ["bash"]
