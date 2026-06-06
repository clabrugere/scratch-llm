# KV Cache Integration Plan

## Context

The model currently recomputes the full attention pass over the growing sequence at every generation step — O(T²) work per token. `model/cache.py` contains a `KVCache` class that pre-allocates the K/V buffer but is not yet wired into the model. This plan fixes the one bug in `cache.py` and threads it through the attention stack so `generate()` uses prefill + single-token decode.

---

## Bug in `cache.py`

`update()` has a dimension error:
```python
seq_len = key_new.size(-1)  # wrong: -1 is head_dim
```
Keys/values are `(bs, num_heads, seq_len, head_dim)`, so it should be `key_new.size(-2)`.

Also: `torch.empty(...)` always allocates on CPU. A `device` parameter is needed.

---

## RoPE Constraint

`RotaryPositionalEncoding.forward()` always applies rotations for positions `0..seq_len-1`. With a cache, a new token at step T must be rotated at position T, not 0. A `start_pos: int = 0` parameter is needed; cached K values already have the correct rotations embedded.

---

## Changes (bottom-up)

### 1. `model/cache.py`
- Add `device=None` to `__init__`, store as `self.device`
- Pass `device=self.device` to `torch.empty` in `reset()`
- Fix: `key_new.size(-1)` → `key_new.size(-2)`

### 2. `model/transformer.py` — `RotaryPositionalEncoding.forward`
```python
def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
    seq_len = x.size(2)
    cos = self.position_cos[:, :, start_pos : start_pos + seq_len, :]
    sin = self.position_sin[:, :, start_pos : start_pos + seq_len, :]
    return (x * cos) + (self._rotate_half(x) * sin)
```
Default `start_pos=0` keeps all existing call sites unchanged.

### 3. `model/transformer.py` — move `causal_mask` to `TransformerStack`

**The refactor**: each `MultiHeadAttention` currently registers an identical `causal_mask` buffer. Move it to `TransformerStack.__init__` (one copy), compute the unified mask there, and pass it down as a plain `attn_mask`. This makes `MultiHeadAttention` mask-agnostic — it just applies whatever mask it receives.

**`TransformerStack`**:
```python
class TransformerStack(Module):
    def __init__(self, num_layers, seq_len, dim_emb, attn_num_heads, ffn_hidden_dim):
        super().__init__()
        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        self.blocks = ModuleList(...)

    def forward(self, x: Tensor, attn_mask: Tensor = None, cache=None) -> Tensor:
        q_len = x.size(1)
        if q_len > 1:  # training or prefill — apply causal mask
            causal = self.causal_mask[None, None, :q_len, :q_len]
            mask = causal if attn_mask is None else (causal | attn_mask)
        else:  # single-token decode — causality enforced by cache
            mask = attn_mask

        for layer_idx, block in enumerate(self.blocks):
            x, _ = block(x, attn_mask=mask, cache=cache, layer_idx=layer_idx)
        return x
```

`q_len > 1` is the right discriminant:

| Phase | cache | q_len | causal mask applied? |
|---|---|---|---|
| Training | None | > 1 | Yes ✓ |
| Prefill | set | prompt_len > 1 | Yes ✓ |
| Decode | set | 1 | No ✓ (cache enforces causality) |

**`MultiHeadAttention`**:
- Remove `causal_mask` buffer from `__init__`, remove `causal` param and its masking branch
- `forward` adds `cache=None, layer_idx: int = 0`; applies only `attn_mask`:
```python
start_pos = cache.current_seq_len if cache is not None else 0
q = self.positional_encoding(q, start_pos=start_pos)
k = self.positional_encoding(k, start_pos=start_pos)

if cache is not None:
    k, v = cache.update(layer_idx, k, v)

attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_emb**-0.5
if attn_mask is not None:
    attn_scores.masked_fill_(attn_mask, -torch.inf)
```

### 4. `model/transformer.py` — `TransformerBlock.forward`
Add `cache=None, layer_idx: int = 0`; pass both to `multihead_attn`. Remove `causal` from `__init__` (no longer passed to MHA).

### 6. `model/llm.py` — `LLM.forward`
Add `cache=None`; pass to `self.transformer(x, cache=cache)`.

### 7. `model/llm.py` — `LLM.generate`
Two-phase, external interface unchanged:

```python
# build cache from model internals
device = next(self.parameters()).device
cache = KVCache(num_layers, batch_size, num_heads, self.seq_len, dim_head, device=device)

# Phase 1: prefill — run full prompt, populate cache
inputs_cond = inputs[:, -self.seq_len:]  # truncate if needed
logits = self(inputs_cond, cache=cache)[:, -1, :]
cache.step(inputs_cond.size(1))

# sample first token from prefill logits
next_token = sample_top_p(F.softmax(logits / temperature, dim=-1), top_p)
if stop: return ...
inputs = torch.cat((inputs, next_token), dim=-1)

# Phase 2: decode — one token at a time
for _ in range(max_seq_len - 1):
    if cache.current_seq_len >= self.seq_len:
        break  # context window full
    logits = self(next_token, cache=cache)[:, -1, :]
    cache.step(1)
    next_token = sample_top_p(F.softmax(logits / temperature, dim=-1), top_p)
    if stop: break
    inputs = torch.cat((inputs, next_token), dim=-1)
```

`cache.step()` is called in `generate()` — after all layers complete — not inside `MultiHeadAttention`. This keeps `current_seq_len` stable across all layers during one forward pass.

---

## Signature Changes Summary

| File | What changes |
|---|---|
| `cache.py` | `__init__` +`device=None`; `reset` uses device; `update` fixes dim |
| `transformer.py` | `RoPE.forward` +`start_pos=0`; `TransformerStack.__init__` gains `causal_mask` buffer; `TransformerStack.forward` +`cache`, builds unified mask; `MHA.__init__` drops `causal_mask`/`causal`; `MHA.forward` +`cache`, `layer_idx`, drops causal branch; `TransformerBlock.forward` +`cache`, `layer_idx`, drops `causal` from `__init__` |
| `llm.py` | `LLM.forward` +`cache=None`; `LLM.generate` two-phase |

All new parameters have defaults — training and the notebook require zero changes.

---

## Verification

1. Run existing notebook (shakespeare.ipynb) — training and generation must produce the same output quality as before (cache=None path unchanged).
2. Manually call `model.generate(prompt, max_seq_len=50)` and verify output is coherent.
3. Optionally time: generation should be measurably faster for long sequences.
