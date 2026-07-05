"""Tests for examples/Llama-3.1-8B-Training/ bundle structure and reference script.

AST checks verify reference.py delegates to TorchTitan (not a hand-rolled model).
Functional tests stub out torchtitan so make_batch / TrainingConfig / _make_model_args
can be exercised locally without PyTorch >= 2.5 or a GPU.
"""

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

_requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

BUNDLE = Path("examples/Llama-3.1-8B-Training")
REF_DIR = BUNDLE / "reference"


# ---------------------------------------------------------------------------
# Bundle layout
# ---------------------------------------------------------------------------

class TestBundleLayout:
    def test_reference_py_exists(self):
        assert (REF_DIR / "reference.py").is_file()

    def test_config_json_exists(self):
        assert (REF_DIR / "config.json").is_file()

    def test_meta_json_exists(self):
        assert (REF_DIR / "meta.json").is_file()

    def test_accuracy_checker_exists(self):
        assert (BUNDLE / "accuracy_checker" / "checker.py").is_file()

    def test_benchmark_exists(self):
        assert (BUNDLE / "benchmark" / "benchmark.py").is_file()

    def test_objective_exists(self):
        assert (BUNDLE / "OBJECTIVE.md").is_file()


# ---------------------------------------------------------------------------
# config.json correctness
# ---------------------------------------------------------------------------

class TestConfigJson:
    @pytest.fixture(scope="class")
    def cfg(self):
        return json.loads((REF_DIR / "config.json").read_text())

    def test_model_type(self, cfg):
        assert cfg["model_type"] == "llama"

    def test_llama_31_8b_architecture(self, cfg):
        assert cfg["hidden_size"] == 4096
        assert cfg["num_hidden_layers"] == 32
        assert cfg["num_attention_heads"] == 32
        assert cfg["num_key_value_heads"] == 8
        assert cfg["vocab_size"] == 128256
        assert cfg["intermediate_size"] == 14336

    def test_rope_theta(self, cfg):
        assert cfg["rope_theta"] == pytest.approx(500000.0)

    def test_training_section_present(self, cfg):
        tr = cfg["training"]
        assert tr["seq_len"] == 4096
        assert tr["local_batch_size"] == 2

    def test_training_optimizer_hyperparams(self, cfg):
        tr = cfg["training"]
        assert tr["learning_rate"] == pytest.approx(3e-4)
        assert tr["weight_decay"] == pytest.approx(0.1)
        assert tr["beta1"] == pytest.approx(0.9)
        assert tr["beta2"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# reference.py AST checks — uses TorchTitan, no hand-rolled model
# ---------------------------------------------------------------------------

class TestReferenceStructure:
    @pytest.fixture(scope="class")
    def tree(self):
        return ast.parse((REF_DIR / "reference.py").read_text())

    @pytest.fixture(scope="class")
    def source(self):
        return (REF_DIR / "reference.py").read_text()

    def _class_names(self, tree):
        return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    def _import_modules(self, tree):
        return [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]

    def test_imports_torchtitan_transformer(self, tree):
        assert any("torchtitan" in m for m in self._import_modules(tree))

    def test_imports_transformer_block_from_torchtitan(self, source):
        assert "TransformerBlock" in source
        assert "torchtitan" in source

    def test_no_hand_rolled_llama_model(self, tree):
        classes = self._class_names(tree)
        for hand_rolled in ("LlamaModel", "Attention", "MLP", "FeedForward"):
            assert hand_rolled not in classes, (
                f"reference.py should use TorchTitan's {hand_rolled}, not reimplement it"
            )

    def test_no_hand_rolled_rms_norm(self, tree):
        assert "RMSNorm" not in self._class_names(tree)

    def test_no_hand_rolled_transformer_block(self, tree):
        # TransformerBlock must be imported from torchtitan, not defined here
        classes = self._class_names(tree)
        assert "TransformerBlock" not in classes

    def test_calls_init_weights(self, source):
        assert "init_weights" in source, (
            "reference.py must call TorchTitan's model.init_weights() for canonical initialization"
        )

    def test_uses_fsdp_summon_full_params(self, source):
        assert "summon_full_params" in source, (
            "gradient capture must use FSDP.summon_full_params to gather shards before saving"
        )


# ---------------------------------------------------------------------------
# Functional tests — stub torchtitan so the module loads on any PyTorch version
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
@_requires_torch
def ref_module():
    """Load reference.py with torchtitan imports stubbed.

    TorchTitan requires PyTorch >= 2.5; this fixture lets the pure-Python
    functions (make_batch, TrainingConfig, _make_model_args) be tested locally
    without a GPU or a matching PyTorch version.
    """
    tt_subpaths = [
        "torchtitan",
        "torchtitan.models",
        "torchtitan.models.llama3",
        "torchtitan.models.llama3.model",
        "torchtitan.models.llama3.model.model",
        "torchtitan.models.llama3.model.args",
    ]
    # Install stubs only for paths not already present (real install wins)
    added = []
    for path in tt_subpaths:
        if path not in sys.modules:
            sys.modules[path] = types.ModuleType(path)
            added.append(path)

    model_mod = sys.modules["torchtitan.models.llama3.model.model"]
    model_mod.Transformer = MagicMock(name="Transformer")
    model_mod.TransformerBlock = MagicMock(name="TransformerBlock")

    args_mod = sys.modules["torchtitan.models.llama3.model.args"]
    args_mod.TransformerModelArgs = MagicMock(name="TransformerModelArgs")
    args_mod.RoPEScalingArgs = MagicMock(name="RoPEScalingArgs")

    spec = importlib.util.spec_from_file_location("_ref_under_test", str(REF_DIR / "reference.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ref_under_test"] = mod  # dataclass needs this during class construction
    spec.loader.exec_module(mod)

    yield mod

    for path in added:
        sys.modules.pop(path, None)


@_requires_torch
class TestMakeBatch:
    def test_output_shape(self, ref_module):
        inp, tgt = ref_module.make_batch(2, 16, 128256, 42, 1, 0)
        assert inp.shape == (2, 16)
        assert tgt.shape == (2, 16)

    def test_deterministic_same_args(self, ref_module):
        a, _ = ref_module.make_batch(2, 16, 128256, 42, 1, 0)
        b, _ = ref_module.make_batch(2, 16, 128256, 42, 1, 0)
        assert torch.equal(a, b)

    def test_different_steps_produce_different_batches(self, ref_module):
        a, _ = ref_module.make_batch(2, 16, 128256, 42, 1, 0)
        b, _ = ref_module.make_batch(2, 16, 128256, 42, 2, 0)
        assert not torch.equal(a, b)

    def test_different_ranks_produce_different_batches(self, ref_module):
        a, _ = ref_module.make_batch(2, 16, 128256, 42, 1, 0)
        b, _ = ref_module.make_batch(2, 16, 128256, 42, 1, 1)
        assert not torch.equal(a, b)

    def test_input_and_labels_are_shifted_sequence(self, ref_module):
        # inp = ids[:, :-1], tgt = ids[:, 1:] from the same seeded sequence
        g = torch.Generator()
        g.manual_seed(42 + 1 * 1000 + 0)  # seed=42, step=1, rank=0
        ids = torch.randint(0, 128256, (1, 9), generator=g)
        inp, tgt = ref_module.make_batch(1, 8, 128256, 42, 1, 0)
        assert torch.equal(inp, ids[:, :-1])
        assert torch.equal(tgt, ids[:, 1:])

    def test_tokens_within_vocab_range(self, ref_module):
        inp, tgt = ref_module.make_batch(4, 32, 128256, 42, 5, 3)
        assert int(inp.min()) >= 0 and int(inp.max()) < 128256
        assert int(tgt.min()) >= 0 and int(tgt.max()) < 128256


@_requires_torch
class TestTrainingConfig:
    def test_from_json_parses_seq_len_and_batch_size(self, ref_module):
        cfg = ref_module.TrainingConfig.from_json(REF_DIR / "config.json")
        assert cfg.seq_len == 4096
        assert cfg.local_batch_size == 2

    def test_from_json_optimizer_hyperparams(self, ref_module):
        cfg = ref_module.TrainingConfig.from_json(REF_DIR / "config.json")
        assert cfg.learning_rate == pytest.approx(3e-4)
        assert cfg.weight_decay == pytest.approx(0.1)
        assert cfg.beta1 == pytest.approx(0.9)
        assert cfg.beta2 == pytest.approx(0.95)
        assert cfg.max_grad_norm == pytest.approx(1.0)

    def test_missing_keys_fall_back_to_defaults(self, ref_module, tmp_path):
        minimal = tmp_path / "config.json"
        minimal.write_text('{"training": {"seq_len": 512}}')
        cfg = ref_module.TrainingConfig.from_json(minimal)
        assert cfg.seq_len == 512
        assert cfg.local_batch_size == 2   # default
        assert cfg.seed == 42              # default


@_requires_torch
class TestMakeModelArgs:
    """Verify _make_model_args passes the correct Llama 3.1 8B values to TorchTitan."""

    @pytest.fixture(autouse=True)
    def reset_mocks(self, ref_module):
        TransformerModelArgs = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs
        RoPEScalingArgs = sys.modules["torchtitan.models.llama3.model.args"].RoPEScalingArgs
        TransformerModelArgs.reset_mock()
        RoPEScalingArgs.reset_mock()

    def test_model_dimensions(self, ref_module):
        ref_module._make_model_args(seq_len=4096)
        kw = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs.call_args.kwargs
        assert kw["dim"] == 4096
        assert kw["n_layers"] == 32
        assert kw["n_heads"] == 32
        assert kw["n_kv_heads"] == 8
        assert kw["vocab_size"] == 128256

    def test_ffn_sizing(self, ref_module):
        ref_module._make_model_args(seq_len=4096)
        kw = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs.call_args.kwargs
        # ffn_dim_multiplier=1.3 + multiple_of=1024 → SwiGLU hidden = 14336
        assert kw["multiple_of"] == 1024
        assert kw["ffn_dim_multiplier"] == pytest.approx(1.3)

    def test_rope_and_norm(self, ref_module):
        ref_module._make_model_args(seq_len=4096)
        kw = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs.call_args.kwargs
        assert kw["rope_theta"] == pytest.approx(500000.0)
        assert kw["norm_eps"] == pytest.approx(1e-5)

    def test_attention_type_is_sdpa(self, ref_module):
        ref_module._make_model_args(seq_len=4096)
        kw = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs.call_args.kwargs
        assert kw["attn_type"] == "sdpa"

    def test_seq_len_forwarded_to_max_seq_len(self, ref_module):
        ref_module._make_model_args(seq_len=2048)
        kw = sys.modules["torchtitan.models.llama3.model.args"].TransformerModelArgs.call_args.kwargs
        assert kw["max_seq_len"] == 2048

    def test_rope_scaling_llama31_values(self, ref_module):
        ref_module._make_model_args(seq_len=4096)
        kw = sys.modules["torchtitan.models.llama3.model.args"].RoPEScalingArgs.call_args.kwargs
        assert kw["scaling_factor"] == pytest.approx(8.0)
        assert kw["low_freq_factor"] == pytest.approx(1.0)
        assert kw["high_freq_factor"] == pytest.approx(4.0)
        assert kw["original_max_position_embeddings"] == 8192
