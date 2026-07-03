"""Jinja2 prompt rendering.

Two concepts:

- **Template** — a full prompt the LLM sees as one document
  (e.g. ``vibe_train/templates/implementer/system.j2``). Has structure:
  headers, task description, constraints. Lives in a per-mode
  directory.
- **Fragment** — a small reusable snippet meant to be composed *into*
  a template, not rendered standalone. Lives at
  ``vibe_train/templates/_backend/<backend>/<name>.j2``. The
  ``_backend/`` prefix marks "fragment directory, not a place to find
  full templates".

:class:`ComputeBackendFragment` is the Python contract for backend fragments:
its :attr:`~ComputeBackendFragment.NAMES` class attribute is the canonical
list of fragment names, and concrete subclasses
(:class:`CudaComputeBackendFragment`, :class:`MetalComputeBackendFragment`) anchor
each backend in the ``_FRAGMENT_IMPLS`` registry. Adding a fragment
name requires updating ``NAMES`` and creating a ``<name>.j2`` file
under every backend dir (an empty file is a deliberate skip).

:class:`Prompt` validates the bound backend's fragment files exist at
construction time and auto-injects every fragment as a kwarg keyed by
filename stem on every ``render(...)`` call. Templates can therefore
reference ``{{ device_dtype }}`` regardless of which backend the run
targets.
"""

from abc import ABC
from pathlib import Path
from typing import ClassVar

from jinja2 import Environment, FileSystemLoader

from vibe_train.constants import ComputeBackend

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Cache of Jinja2 environments keyed by template directory path
_env_cache: dict[str, Environment] = {str(_TEMPLATES_DIR): _env}


def _build_env(template_dir: Path | str | None = None) -> Environment:
    """Return a Jinja2 Environment for the given template directory.

    Per-mode template directories also fall back to the shared
    ``vibe_train/templates/`` root, so ``{% include
    "_backend/<name>/foo.j2" %}`` from a per-mode template (and
    fragment lookups via :class:`ComputeBackendFragment`) resolve from the
    shared root.
    """
    if template_dir is None:
        return _env
    key = str(template_dir)
    if key not in _env_cache:
        search_paths = [key]
        if key != str(_TEMPLATES_DIR):
            search_paths.append(str(_TEMPLATES_DIR))
        _env_cache[key] = Environment(
            loader=FileSystemLoader(search_paths),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _env_cache[key]


def render_template(
    name: str,
    *,
    template_dir: Path | str | None = None,
    **kwargs: object,
) -> str:
    """Render a Jinja2 template (no fragment auto-injection)."""
    env = _build_env(template_dir)
    return env.get_template(name).render(**kwargs)


class ComputeBackendFragment(ABC):
    """Provides backend-specific Jinja fragments under
    ``vibe_train/templates/_backend/<backend>/``.
    """

    NAMES: ClassVar[frozenset[str]] = frozenset({
        "device_dtype",
        "judge_device_correctness",
        "profiling_workflow",
    })
    backend: ClassVar[ComputeBackend]  # set by subclasses

    def __init__(self, env: Environment) -> None:
        self._env = env

    def render(self, name: str) -> str:
        """Render a single fragment by name."""
        if name not in self.NAMES:
            raise ValueError(
                f"Unknown fragment {name!r}; valid: {sorted(self.NAMES)}"
            )
        return self._env.get_template(
            f"_backend/{self.backend.value}/{name}.j2"
        ).render().rstrip("\n")

    def render_all(self) -> dict[str, str]:
        """Render every fragment in :attr:`NAMES` keyed by name."""
        return {name: self.render(name) for name in self.NAMES}

    @classmethod
    def validate(cls) -> None:
        """Verify a ``.j2`` file exists for every fragment in :attr:`NAMES`."""
        backend_dir = _TEMPLATES_DIR / "_backend" / cls.backend.value
        missing = [
            n for n in cls.NAMES if not (backend_dir / f"{n}.j2").is_file()
        ]
        if missing:
            raise ValueError(
                f"{cls.__name__}: missing fragment files under {backend_dir}: "
                f"{', '.join(f'{n}.j2' for n in sorted(missing))}. "
                f"Use an empty file for a deliberate skip."
            )


class CudaComputeBackendFragment(ComputeBackendFragment):
    """Fragments for the CUDA backend (NVIDIA GPUs)."""
    backend = ComputeBackend.CUDA


class MetalComputeBackendFragment(ComputeBackendFragment):
    """Fragments for the Metal backend (Apple Silicon, MPS)."""
    backend = ComputeBackend.METAL


_FRAGMENT_IMPLS: dict[ComputeBackend, type[ComputeBackendFragment]] = {
    ComputeBackend.CUDA: CudaComputeBackendFragment,
    ComputeBackend.METAL: MetalComputeBackendFragment,
}


def get_backend_fragment(backend: ComputeBackend, env: Environment) -> ComputeBackendFragment:
    """Construct the :class:`ComputeBackendFragment` impl for the given backend."""
    if backend not in _FRAGMENT_IMPLS:
        raise ValueError(
            f"No ComputeBackendFragment registered for {backend!r}. "
            f"Registered: {sorted(_FRAGMENT_IMPLS.keys(), key=lambda b: b.value)}"
        )
    return _FRAGMENT_IMPLS[backend](env)


class Prompt:
    """Render templates from a per-mode directory, with backend fragments
    auto-injected as kwargs.

    Parameters
    ----------
    template_dir:
        Per-mode directory the renderer searches first (e.g.
        ``vibe_train/templates/``). Falls back to the shared
        ``vibe_train/templates/`` root, where backend fragments live.
    backend:
        Hardware backend the run targets. Selects the
        :class:`ComputeBackendFragment` impl whose fragments get
        auto-injected.
    """

    def __init__(self, template_dir: Path | str, backend: ComputeBackend) -> None:
        self._env = _build_env(template_dir)
        self._fragments = get_backend_fragment(backend, self._env)
        type(self._fragments).validate()

    def render(self, name: str, **kwargs: object) -> str:
        """Render a full template.

        ComputeBackend fragments are auto-injected as kwargs keyed by
        filename stem; explicit kwargs override.
        """
        auto = self._fragments.render_all()
        return self._env.get_template(name).render(**(auto | kwargs))

    def fragment(self, name: str) -> str:
        """Render a single backend fragment by name."""
        return self._fragments.render(name)
