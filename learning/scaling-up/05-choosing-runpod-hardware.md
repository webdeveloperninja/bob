# 05 — Choosing RunPod hardware for training and serving

[Scaling Up](README.md) · Next: [Good inference on a smaller budget](06-lower-cost-inference.md)

**For Bob today, start with your Mac or a modest GPU—not an H100 by default.** A larger GPU becomes worthwhile when measurements show that memory, throughput, or turnaround time limits your next experiment.

The tables below are planning recommendations based on memory accounting, not measured RunPod benchmarks. They assume compatible software and NVIDIA GPUs. Check the actual model, context length, batch/concurrency, GPU variant, and rental offer before choosing. “Fits in memory” does not mean “meets the response-time target.”

## A short GPU shortlist

These examples and memory sizes appear in [RunPod's GPU catalog](https://docs.runpod.io/references/gpu-types), checked September 17, 2026. Availability varies. Check [current pricing](https://www.runpod.io/pricing) when renting; hourly prices are deliberately not frozen into these notes.

| Memory class | Example GPU choices | How to think about the class |
| --- | --- | --- |
| 16 GB | RTX A4000, RTX 4080 | Small models and constrained experiments |
| 24 GB | RTX 3090, RTX 4090, L4 | Common starting point for small-model serving and adapter training |
| 32 GB | RTX 5090 | More memory than a 24 GB card; verify the runtime supports this generation |
| 48 GB | A40, RTX A6000, L40S | Larger weights, longer contexts, or more concurrent requests |
| 80 GB | A100 80GB, H100 PCIe/SXM | Large training/serving workloads with substantial memory needs |
| 96–141 GB | RTX PRO 6000 96GB, H200 SXM 141GB | Large single-GPU deployments; compare cost against multiple GPUs |

Memory capacity is only one property. GPUs with equal VRAM can differ greatly in compute, bandwidth, kernels, and price. H100 PCIe and SXM are distinct offers. A catalog entry also does not guarantee immediate stock or the same availability through Serverless.

The newer and older cards may need different compatible CUDA/runtime images. These guides focus on NVIDIA CUDA; an AMD rental would require a separately validated ROCm/software path.

## Inference memory: weights plus running state

Start with a weights-only estimate:

```text
weight bytes ≈ number of parameters × bits per weight / 8
```

| Dense model size | FP32 | BF16 / FP16 | INT8 ideal | 4-bit ideal |
| --- | --- | --- | --- | --- |
| 1B | 4 GB | 2 GB | 1 GB | 0.5 GB |
| 3B | 12 GB | 6 GB | 3 GB | 1.5 GB |
| 8B | 32 GB | 16 GB | 8 GB | 4 GB |
| 14B | 56 GB | 28 GB | 14 GB | 7 GB |
| 32B | 128 GB | 64 GB | 32 GB | 16 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |

B means one billion parameters. These are decimal GB and ideal payload sizes. Quantization scales, some higher-precision layers, alignment, runtime buffers, and tensor layouts add overhead. Actual GPU capacity accounting may use different units.

Add KV cache, activations/workspace, runtime reserve, and operating headroom. Weight quantization alone does not quantize the KV cache. Host RAM and storage also matter for downloading, converting, and loading model shards.

### Worked cache example

For an illustrative decoder with 32 layers, 8 KV heads, head width 128, and 2-byte cache values:

```text
cache per request = 2 × 32 × total_cached_tokens × 8 × 128 × 2
8,192 cached tokens = 1,073,741,824 bytes = 1 GiB
four such active requests ≈ 4 GiB of KV data
```

The first factor of 2 accounts for keys and values. Tokens include both prompt and generated continuation. This example uses grouped-query attention; a model with more KV heads can need substantially more cache at the same parameter count. It is not Bob's architecture or a universal 8B-model memory estimate.

## What to rent for language-model inference

Assumptions: a compatible serving engine, dense models, initial concurrency 1, and short-to-moderate contexts around 2k–4k subword tokens. Test larger context and concurrency separately. The recommendations intentionally leave more room than the raw weight size.

| Workload | Initial candidate | When to move up |
| --- | --- | --- |
| Bob 0.875M or 10.9M, current FP32 code | Mac/CPU first; 16–24 GB GPU only if latency/load warrants it | Benchmark before spending more; model loading/service design may dominate |
| Educational Bob around 86M | CPU for low-rate tests; 16–24 GB GPU for faster experiments | Current uncached generation, not VRAM alone, can limit speed |
| Dense 1–3B text model, BF16/FP16 | 16 GB GPU; 24 GB for extra room | Longer prompts, batch size, or simultaneous users |
| Dense 7–8B, good 4-bit checkpoint | 24 GB RTX 3090/4090 or L4 | Need more cache/concurrency or measured latency misses target |
| Dense 7–8B, BF16 | 24 GB is a bounded starting test; 48 GB for more headroom | Long context and multi-user traffic can exhaust the remaining memory |
| Dense 13–14B, 4-bit | 24 GB candidate; 48 GB for serving headroom | Memory or latency under real traffic |
| Dense 13–14B, BF16 | 48 GB | Larger context/concurrency or higher throughput requirement |
| Dense 30–32B, 4-bit | 48 GB conservative starting point | 24 GB may fit selected setups, but is not a broad recommendation |
| Dense 30–32B, BF16 | 80 GB, initially constrained concurrency | Cache demand may require larger/multiple GPUs |
| Dense 70B, 4-bit | 80 GB conservative starting point | 48 GB can work in constrained supported setups; benchmark before relying on it |
| Dense 70B, BF16 | Multiple 80 GB GPUs with supported sharding, or a larger-memory option with enough headroom | A 141 GB card is too close to the ideal 140 GB weight payload to assume a usable fit |

For throughput, two replicas of a model that fits on one GPU may be more useful than splitting it across two GPUs. For a model that cannot fit, tensor/model parallelism can distribute weights, but adds communication and requires engine support. Two 24 GB GPUs do **not** automatically behave like one 48 GB GPU. Interconnect and topology matter; check the specific deployment.

These size classes can apply to text/chat or code-specialized dense models, but the task affects required quality and output length. A code model may need longer outputs, raising resource use. A mixture-of-experts model's **total** parameter count matters for resident weights; “active parameters per token” alone is not a memory budget.

## What about other types of models?

| Type | How to size the first test |
| --- | --- |
| Small embeddings, classifiers, or rerankers | Try CPU first for low traffic, then 16–24 GB GPU for batching; input length and batch size matter |
| Vision-language model | Count the language model plus vision components; image count/resolution creates extra tokens and activations; start above the comparable text-only budget |
| Speech recognition/synthesis | Benchmark the actual architecture with realistic audio duration and simultaneous streams; text-token throughput is not an adequate metric |
| Image diffusion | Resolution, batch size, model components, and inference steps matter; many smaller pipelines are candidates for 24 GB, but use the selected pipeline's own requirements |
| Video generation | Frame count and resolution can dominate memory; do not extend a text-model parameter table to video |

These are sizing approaches, not compatibility guarantees for every model in each category. Bob implements none of the image/audio pipelines.

## Training needs a different table

Full-parameter training needs gradients, optimizer state, and stored/recomputed activations. A useful first estimate for Bob-style FP32 parameters with AdamW is **16 bytes per parameter before activations and overhead**. Other precision/optimizer/sharding schemes have different accounting.

```text
1B parameters:  about 16 GB before activations
3B parameters:  about 48 GB before activations
8B parameters: about 128 GB before activations
```

So a GPU that serves an 8B model does not necessarily fully train it. Scratch pretraining also needs appropriate data and many updates; fitting one batch says little about the budget for learning useful capabilities.

| Training task | Initial hardware candidate | Conditions |
| --- | --- | --- |
| Bob local config | Your Mac or CPU | Already appropriate for learning |
| Bob 10.9M config | 24 GB GPU such as 3090/4090 | Start with a small batch; compare speed/cost before choosing H100 |
| Proposed Bob 85.8M | 24 GB candidate at short context and small batch; 48 GB for more room | Profile; no fit guarantee at arbitrary settings |
| Full training/fine-tuning around 0.1–0.5B | 24–48 GB | Batch, context, and activation memory determine actual fit |
| Full training around 1B | 48 GB conservative test; 80 GB if needed | 24 GB can be tight; activation checkpointing changes tradeoffs |
| Full training around 3B | 80 GB or supported sharded training | The approximate 48 GB state subtotal is not total usage |
| Full training around 7–8B | Supported multi-GPU sharding/offload, commonly multiple 80 GB cards | One 80 GB card cannot hold the conventional 16-byte-per-parameter subtotal |
| QLoRA adaptation of 7–8B | 24 GB candidate | Compatible 4-bit base + adapters; modest context/microbatch, often activation checkpointing |
| QLoRA adaptation of 13–14B | 48 GB conservative candidate | Some constrained 24 GB recipes exist; do not assume every recipe fits |
| QLoRA adaptation of 30–32B | 80 GB conservative candidate | 48 GB may work with tighter settings; depends strongly on sequence length |
| Adapter training around 70B | Large-memory GPU or supported multi-GPU recipe | Use a validated recipe for the exact model rather than this rough table |

QLoRA uses a quantized base with trainable adapters; it does not update every quantized base weight like ordinary full fine-tuning. See [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization). Bob currently supports neither QLoRA nor distributed/sharded training.

Reduce microbatch size first when activations exceed memory. Gradient accumulation can preserve a larger effective batch, and activation checkpointing trades recomputation for activation memory. Neither reduces the number of base parameters; neither feature is implemented in Bob yet.

## Choose by a short measured trial

Record exact model/revision, dtype/quantization, context, batch/concurrency, GPU variant, runtime image, and hourly rate. Verify one realistic workload, then compare two candidate GPU types using the same workload and quality criteria.

For training, compare cost to reach the same held-out target, not just steps per second. For serving, compare cost per successful request that meets quality and latency targets. A cheaper hourly GPU can cost more per useful answer if it is too slow or requires too many replicas.

A practical starting decision: keep learning Bob locally; trial a 24 GB GPU for a larger learning run or a compatible quantized 7–8B application; use 48 GB when cache/concurrency becomes the constraint; move to 80 GB+ when model memory or measured throughput justifies it.
