# 03 — Attention inside your decoder

[Learning home](README.md) · Previous: [Math](02-math-with-intuition.md) · Next: [Encoder–decoder](04-encoder-decoder.md)

Open [model.py](../model.py) alongside this chapter. The explanation follows Bob's implementation, which is a small modern-style, pre-normalized decoder-only Transformer, not an exact reproduction of the original 2017 model.

## Follow the dimensions

Use `B` for batch size, `T` for sequence length, `D` for model width, `H` for head count, and `dh = D/H` for each head's width. Vocabulary size is `Vocab = 256`.

| Stage | Shape | Local config |
| --- | --- | --- |
| Input IDs | `[B, T]` | `[8, 128]` |
| Token + position embeddings | `[B, T, D]` | `[8, 128, 128]` |
| Combined Q/K/V projection | `[B, T, 3D]` | `[8, 128, 384]` |
| Each of Q, K, V after head split | `[B, H, T, dh]` | `[8, 4, 128, 32]` |
| Conceptual attention scores | `[B, H, T, T]` | `[8, 4, 128, 128]` |
| Concatenated head outputs | `[B, T, D]` | `[8, 128, 128]` |
| MLP hidden features | `[B, T, 4D]` | `[8, 128, 512]` |
| Final logits | `[B, T, Vocab]` | `[8, 128, 256]` |

The optimized attention kernel need not store the entire conceptual score matrix at once. Shapes describe the mathematics, not a guarantee about intermediate GPU allocations.

## Embeddings give the model something to work with

`self.tokens(tokens)` looks up a learned vector for each byte. `self.positions(positions)` supplies a learned vector for its position within the current window. Their sum enters the Transformer blocks.

Position matters: “dog bites person” and “person bites dog” use the same words but different order. Bob uses a learned position table with `context_length` entries. It cannot simply accept a longer context by loading the same checkpoint and changing one argument. Extending context requires a design and training change.

The word examples in this chapter are conceptual. In your model the actual units are bytes, including spaces and punctuation.

## Attention mixes information between positions

Each position begins with a vector. Attention computes a new vector using a weighted mixture of vectors from allowed positions. The mixing weights depend on the current content, rather than being fixed for every input.

Three learned projections give each position three roles:

- **Query (Q):** what features should this position match against?
- **Key (K):** what features does another position expose for matching?
- **Value (V):** what information can that position contribute?

These are useful interpretations, not literal questions or a database lookup. All three are numeric arrays learned through the prediction objective. Bob computes them together with `self.qkv(...)`, then separates them with `.chunk(3, dim=-1)`.

## The attention equation, read in four steps

```text
scores  = Q Kᵀ / sqrt(dh)
weights = softmax(scores + mask)     # across key positions
result  = weights V
```

1. `Q Kᵀ` scores each query position against each key position.
2. Dividing by `sqrt(dh)` helps control score magnitudes as head width grows.
3. The mask blocks disallowed positions; softmax turns the remaining scores into nonnegative weights that sum to one per query row.
4. The weighted sum of value vectors gives an output vector at each query position.

For example, suppose a query assigns weights `[0.1, 0.7, 0.2]` to three allowed positions. Its output is `0.1*v₀ + 0.7*v₁ + 0.2*v₂`. The second position contributes most in this example. You can understand the operation without manually expanding its 32 coordinates.

## The causal mask prevents cheating

For three input tokens, the allowed positions are:

| Query position ↓ / Key position → | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | Allow | Block | Block |
| 1 | Allow | Allow | Block |
| 2 | Allow | Allow | Allow |

Think of blocked scores as negative infinity before softmax, which makes their weights zero. Bob asks PyTorch for this behavior with `is_causal=True`.

If the mask were removed while retaining shifted next-byte targets, earlier positions could read later bytes—including their answers. Training loss could look excellent while generation failed. Full-sequence training works because every position gets its own restricted view, even though those calculations run in parallel.

## Why multiple heads?

Bob has four heads. Each uses a different learned projection into a 32-dimensional space. Their outputs are concatenated, then transformed by `self.projection`.

Several heads allow different content-dependent mixtures to be computed in parallel. A head is not guaranteed to specialize in a tidy human category such as “grammar” or “facts.” Adding heads at fixed width also reduces the dimensions per head; it does not automatically increase total capacity in a useful way.

Attention weights are intermediate quantities, not the model's learned weight matrices. They change with the prompt. A heatmap can show where information is mixed, but it is not a complete explanation of a prediction: value vectors, output projections, residual paths, and later blocks also matter.

## What makes it a Transformer block?

Bob's block is approximately:

```text
u   = x + Attention(LayerNorm(x))
out = u + MLP(LayerNorm(u))
```

The additions are **residual connections**. Each sublayer modifies an existing representation rather than having to replace it entirely. They also provide direct paths for gradients. Normalizing before each sublayer makes this a **pre-norm** layout.

Four blocks repeat this process. Later blocks receive representations already enriched by earlier ones. A final layer norm and linear output layer turn those representations into next-byte logits. There is no cross-attention in this model.

## See the real tensor shapes yourself

This reads your checkpoint and performs one forward pass; it does not train or modify it:

```bash
python - <<'PY'
import torch
from model import TinyLM
saved = torch.load('runs/local/checkpoint.pt', map_location='cpu', weights_only=True)
model = TinyLM(saved['config']).eval()
model.load_state_dict(saved['model'])
x = torch.tensor([list(b'hello')], dtype=torch.long)
with torch.no_grad():
    logits = model(x)
print('input:', tuple(x.shape))
print('logits:', tuple(logits.shape))
print('first QKV matrix:', tuple(model.blocks[0].qkv.weight.shape))
PY
```

For the local checkpoint, expect `[1, 5]`, `[1, 5, 256]`, and `[384, 128]` (Python prints tuples with parentheses). A short sequence can fit within the larger configured context.

## Why attention costs more with longer context

Every query can compare with many keys. The conceptual score table grows roughly as `T²`, with additional dependence on batch size, heads, and head width. Doubling context can roughly quadruple this pairwise attention work; it does not imply that all end-to-end training time quadruples because other computations scale differently.

Bob's generation recomputes previous positions. Key/value caching is a later performance improvement: retain useful past attention projections instead of recomputing them. It does not add learned facts or change the training objective. Bob's learned, window-relative positions make sliding-window cache management a design detail that needs care.

## Check yourself

1. Are Q, K, and V saved as fixed per-prompt arrays in the checkpoint?
2. Which dimension does attention softmax normalize over?
3. Would changing `is_causal=True` to `False` create a full encoder–decoder model?

**Answer hints:** the projection parameters are saved; Q/K/V activations are recomputed from input. Normalize over allowed key positions for each query. Removing the mask does not create an encoder or cross-attention, and would invalidate Bob's current objective.

For visual reinforcement after this explanation, try [3Blue1Brown's attention lesson](https://www.3blue1brown.com/lessons/attention/). Its diagrams provide another way to follow query/key matching and value mixing.
