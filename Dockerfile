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
RUN curl -Ls https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local sh
ENV PATH="/usr/local/bin:$PATH"

# ── non-root user (Claude Code CLI refuses to run as root) ─────────────────
RUN useradd -m vibe

# ── Python deps ────────────────────────────────────────────────────────────
WORKDIR /app
# Copy manifests first so dep installation is cached separately from source.
# README.md is required by setuptools to build the package metadata.
COPY pyproject.toml uv.lock README.md ./
# Install all non-dev dependencies (torch comes from the NGC base, not uv).
RUN uv sync --frozen --no-dev --no-install-project --extra train

# ── project source ─────────────────────────────────────────────────────────
COPY src/ src/
COPY examples/ examples/
COPY resources/ resources/
# Now install the project itself (fast — deps are already cached above).
RUN uv sync --frozen --no-dev --extra train

# Hand /app to the non-root user so vibe-train can write exp_env/, logs, etc.
RUN chown -R vibe:vibe /app

USER vibe

# ── runtime ────────────────────────────────────────────────────────────────
# Credentials and workspace are supplied at runtime via env vars / mounts.
# Example:
#   docker run --gpus all \
#     -v /path/to/weights:/app/examples/Llama-3.1-8B-Training/reference/model:ro \
#     -v $(pwd)/resources:/app/resources \
#     -v $(pwd)/exp_env:/app/exp_env \
#     --env-file .env \
#     vibe-train --exp-name run-001

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_XET_TRANSFER=1

ENTRYPOINT ["/app/.venv/bin/vibe-train"]
CMD ["--help"]
