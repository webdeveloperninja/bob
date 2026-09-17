# 03 — Efficient inference after training

[Scaling Up](README.md) · Previous: [Saving weights](02-checkpoints-and-releases.md) · Next: [Multi-tenant serving](04-multi-tenant-serving.md)

## First use the checkpoint as it is

On a Pod with your trained Bob model:

```bash
python generate.py --checkpoint runs/h100/checkpoint.pt --device cuda --prompt "The future of AI" --bytes 300
```

On your Mac, use the downloaded checkpoint with `--device auto`. The architecture is reconstructed from the checkpoint config. A GPU is not required for loading; whether generation is fast enough depends on model size and your machine.

Inference does not require the original training binaries. It needs weights, matching architecture, tokenization, and runtime. The tokenizer here is UTF-8 bytes with no chat template or special end token.

## What the current script costs

`generate.py` starts a process, loads a checkpoint, builds the model, generates one sequence, and exits. Calling it as a subprocess for every web request repeats initialization and model loading.

It also recomputes the entire retained context for every new byte. It uses FP32 model weights and `no_grad()`; it does not automatically apply the CUDA BF16 training autocast settings during generation. Your training precision and serving precision are separate choices.

An inference service should instead load one fixed model version at worker startup, call `eval()`, warm it up, then keep it resident. Use `torch.inference_mode()` for a dedicated inference path when appropriate. This avoids autograd overhead; it does not implement caching, batching, or tenant authorization for you.

## Prefill and decode

For a typical cached causal language model, **prefill** processes the prompt. **Decode** generates subsequent tokens using saved attention keys and values. A KV cache trades GPU memory for less repeated computation.

Bob has no KV cache. Adding one requires changes to attention, position handling, and generation—not merely wrapping the function in a web endpoint. Its learned window-relative positions make cache rollover particularly important: evicting an old token and shifting position assignments can invalidate cached calculations. Test cached versus uncached logits before relying on the optimization.

For ordinary multi-head attention, a rough cache size is:

```text
2 × layers × cached_tokens × KV_heads × head_width × bytes_per_value
```

The factor 2 is for keys and values. Sum over active requests. In Bob-style attention, `KV_heads × head_width` equals model width. Other architectures can share KV heads, changing this estimate. Runtime allocation and metadata add overhead. This formula describes a future cached implementation, not memory already allocated by Bob.

## Batching is where sharing can help

A batch lets a GPU process several requests together. Fixed batching waits for a group and often wastes work on padded/finished sequences. **Continuous batching** lets an inference scheduler add and remove active sequences as requests arrive and finish.

A first custom Bob service can use one loaded model and a small bounded queue with serial execution. This is simpler to validate, though it limits throughput. Add batching only after correctness and isolation tests. Multiple web-server processes each loading the model can multiply VRAM use; more HTTP workers are not automatically more GPU capacity.

Set limits on prompt bytes, output bytes, active requests, and queue length. Bob's model currently retains only its configured context and generates a fixed number of bytes. A service must document truncation or reject oversized prompts deliberately, rather than letting customers assume a long prompt was fully used.

## The production-engine path

For a model family supported by a specialized runtime, a serving engine such as vLLM provides a more practical starting point for efficient request scheduling and serving APIs. Check the [supported-model list](https://docs.vllm.ai/en/latest/models/supported_models/) and the [server documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) for the exact model and runtime version you choose.

**Bob's custom `TinyLM` checkpoint is not a drop-in vLLM model.** Its code/config/tokenization must be represented by a supported implementation or integrated explicitly. Saving it as `.safetensors` would change storage, not solve that compatibility problem.

Two reasonable routes are:

| Route | Appropriate when | Work required |
| --- | --- | --- |
| Keep Bob and build a small custom worker | Your main goal is learning inference systems | Persistent model, validation, queue, API, and eventually caching/batching |
| Adapt a supported pretrained model in a separate project | Your main goal is useful application behavior | Compatible training/export pipeline, tokenizer/template, evaluation, serving setup |

An API following a familiar request format does not automatically turn a completion model into a chat model. Chat requires the correct model behavior and input formatting too.

## Pods versus Serverless

A continuously running Pod gives you direct control over the process and warm model. It can suit steady traffic or debugging. RunPod Serverless manages workers/endpoints for inference workloads; study its [overview](https://docs.runpod.io/serverless/overview) before choosing queue-based jobs versus an interactive endpoint.

In either case, load the model once per worker lifecycle. Worker startup and model loading can create cold-start delays. Keeping workers warm can reduce those delays but changes the cost tradeoff. A Serverless handler or endpoint is not included in this repository, and provider autoscaling does not implement your application's tenant policies.

Separate training capacity from inference capacity when you need predictable latency. Sharing a GPU with an active optimizer can introduce memory contention and long response delays.

## Measure the user experience

Record at least:

- Time to first output: queue wait + initialization/prefill + first decode work.
- End-to-end latency, including the slow tail such as p95.
- Output tokens per second and completed requests per second.
- Active requests, queue depth, failures, GPU memory, and cost per completed request.

Bob counts **bytes as tokens**. A production model may count subword tokens; their rates are not directly comparable. Test realistic prompt/output lengths and concurrent request counts, not only a one-word prompt.

Test at concurrency 1, then a small higher value, then increase until latency or errors exceed your target. Streaming makes partial results visible earlier, but does not itself make the GPU generate faster. Client cancellation should stop the corresponding generation and free its resources.

## Precision and deployment changes need evaluation

BF16/FP16 inference and quantization can reduce some memory costs, but support and quality depend on the hardware, model, and runtime. Check outputs and task metrics after changing dtype or quantization. A smaller artifact is not proof that a deployment is faster.

**Performance exercise:** decide whether your bottleneck is repeated model loading, prompt processing, per-token generation, queueing, or network transfer before selecting an optimization.
