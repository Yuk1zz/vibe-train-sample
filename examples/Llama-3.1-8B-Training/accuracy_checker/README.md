# Accuracy Checker — Gradient Correctness Verifier

Compares accumulated gradients before `optimizer.step()` between the candidate and reference systems using `torch.allclose` in FP32.

## Correctness criterion (from proposal)

```python
torch.allclose(grad_candidate_fp32, grad_reference_fp32, rtol=1e-3, atol=1e-5)
```

All parameter gradients must pass.

## Usage

### Step 1 — Generate reference gradients (once)

```bash
torchrun --nproc_per_node=8 ../reference/reference.py \
    --steps 10 --save-grads /tmp/ref_grads.pt --grad-step 10
```

### Step 2 — Generate candidate gradients

Your candidate training script must accept `--save-grads PATH --grad-step N` and save a dict of `{param_name: grad_tensor_fp32}` using `torch.save`.

```bash
torchrun --nproc_per_node=8 train.py \
    --steps 10 --save-grads /tmp/cand_grads.pt --grad-step 10
```

### Step 3 — Compare

```bash
python checker.py --ref /tmp/ref_grads.pt --candidate /tmp/cand_grads.pt
```

Exit code 0 = PASS, 1 = FAIL.

## Gradient save contract

The `.pt` file must be a dict loadable with `torch.load(..., weights_only=True)`:
```python
{
  "model.embed_tokens.weight": tensor([...]),   # float32, CPU
  "model.layers.0.attn.q_proj.weight": tensor([...]),
  ...
}
```

Parameter names must match those used in the reference (FSDP `use_orig_params=True` preserves original names).
