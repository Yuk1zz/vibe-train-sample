"""Backend-agnostic agent runner protocol.

Two implementations live alongside this module:

- :mod:`vibe_serve.agents.deepagents_runner` wraps the existing
  ``deepagents`` + ``langchain`` stack used by every loop today.
- :mod:`vibe_serve.agents.cli_runner` wraps an
  ``agentshim``-backed ``vibe_serve._agent_cli`` compatibility layer, which drives
  external coding-agent CLIs
  (Claude Code, Gemini, Codex, Opencode).

The simple loop calls a single ``invoke()`` method per (iteration × phase).
There is intentionally no separate ``Session`` type — today's loops always
build a fresh agent for each call (clean context window) and ``vibe_serve._agent_cli``
is one-shot at the Python layer, so a reusable session would either be a thin
struct or would lie about reuse semantics on one of the backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from langchain_core.tools import BaseTool
from vibe_train._agent_cli.base import MCPServerSpec
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AgentRunner(Protocol):
    """Backend-agnostic agent invoker. One instance per loop run."""

    backend_name: str
    """Diagnostic identifier — ``"deepagents"`` or ``"cli"``."""

    def invoke(
        self,
        *,
        # static per-task config
        kind: str,
        workspace: Path,
        system_prompt: str,
        env: dict[str, str] | None = None,
        # dynamic per-call config
        user_prompt: str,
        response_cls: type[T],
        fallback_factory: Callable[[], T],
        round_label: str,
        mcp_servers: list[MCPServerSpec] | None = None,
        tools: list[BaseTool] | None = None,
    ) -> T:
        """Run an agent and return a structured response.

        Args:
            kind: One of ``"implementer"``, ``"judge"``, ``"perf_eval"``.
            workspace: Workspace root for this phase.
            system_prompt: Rendered Jinja2 system prompt (per phase).
            env: Optional environment overrides (e.g. ``CUDA_VISIBLE_DEVICES``).
            user_prompt: Rendered Jinja2 user prompt (per iteration).
            response_cls: Pydantic model class the agent should produce.
            fallback_factory: Constructs a default ``response_cls`` instance
                used when the agent fails to produce a parseable response.
            round_label: Short label used in log headers (e.g. ``"judge #3"``).
            mcp_servers: Optional list of stdio MCP servers to install for
                the duration of this call.
            tools: Optional list of in-process LangChain tools to expose to
                the agent for the duration of this call.

        Returns:
            An instance of ``response_cls``, either parsed from the agent's
            output or produced by ``fallback_factory()``.
        """
        ...
