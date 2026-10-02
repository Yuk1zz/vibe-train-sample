# vibe-serve / vibe-train orchestrator
# Includes CUDA + PyTorch (for inner training sandbox), uv (Python deps),
# and Node.js + Claude Code CLI (for the cli implementer agent).
FROM nvcr.io/nvidia/pytorch:25.04-py3

# ── system deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 22 + Claude Code CLI ───────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# ── uv (system-wide so the non-root user can reach it) ─────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

# ── non-root user (Claude Code CLI refuses to run as root) ─────────────────
# Pass host UID/GID at build time so mounted volumes are writable without
# needing --user at runtime:
#   docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t vibe-train .
ARG UID=1001
ARG GID=1001
RUN groupadd -g $GID vibe && useradd -m -u $UID -g $GID vibe

# ── Python deps (run as vibe so .venv is owned correctly — no chown -R) ────
WORKDIR /app
RUN chown vibe:vibe /app && mkdir -p /app/.torch_cache && chown vibe:vibe /app/.torch_cache

# Copy manifests with correct ownership so uv can write the lockfile if needed.
# README.md is required by setuptools to build the package metadata.
COPY --chown=vibe:vibe pyproject.toml uv.lock README.md ./

USER vibe

# Install all non-dev dependencies (torch comes from the NGC base, not uv).
RUN uv sync --frozen --no-dev --no-install-project --extra train

# ── project source ─────────────────────────────────────────────────────────
COPY --chown=vibe:vibe src/ src/
COPY --chown=vibe:vibe examples/ examples/
COPY --chown=vibe:vibe resources/ resources/

# Install the project itself (fast — deps are already cached above).
RUN uv sync --frozen --no-dev --extra train

# ── runtime ────────────────────────────────────────────────────────────────
# Credentials and workspace are supplied at runtime via env vars / mounts.
# Example:
#   docker run --gpus all \
#     -v /path/to/weights:/app/examples/Llama-3.1-8B-Training/reference/model:ro \
#     -v $(pwd)/resources:/app/resources \
#     -v $(pwd)/exp_env:/app/exp_env \
#     --env-file .env \
#     vibe-train --exp-name run-001

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_XET_TRANSFER=1
ENV TORCH_HOME="/app/.torch_cache"

ENTRYPOINT ["/app/.venv/bin/vibe-train"]
CMD ["--help"]
