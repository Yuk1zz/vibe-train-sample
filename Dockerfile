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

# ── uv ─────────────────────────────────────────────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Python deps ────────────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Sync without torch (already provided by the NGC base image)
RUN uv sync --frozen --no-dev \
    && uv pip install -e ".[train]" --no-build-isolation

# ── project source ─────────────────────────────────────────────────────────
COPY src/ src/
COPY examples/ examples/

# Reinstall in editable mode so entry points resolve correctly
RUN uv pip install -e ".[train]" --no-build-isolation

# ── runtime ────────────────────────────────────────────────────────────────
# Credentials and workspace are supplied at runtime via env vars / mounts.
# Example:
#   docker run --gpus all \
#     -v $(pwd)/examples:/app/examples \
#     -v $(pwd)/exp_env:/app/exp_env \
#     --env-file .env \
#     vibe-train vibe-train --exp-name run-001

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_XET_TRANSFER=1

ENTRYPOINT ["uv", "run"]
CMD ["vibe-train", "--help"]
