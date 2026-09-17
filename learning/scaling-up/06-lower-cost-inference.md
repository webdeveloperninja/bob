# 06 — Good inference on a smaller budget

[Scaling Up](README.md) · Previous: [Choosing hardware](05-choosing-runpod-hardware.md)

A smaller GPU can produce the same useful result as a larger one if the model, serving format, context, and load fit. Hardware does not directly determine answer quality. However, compromises made to fit it—smaller models, lossy quantization, or shortened context—can affect quality.

Use a task-specific evaluation set to decide which compromises are acceptable. The goal is **quality and latency within budget**, not the lowest bit count or cheapest hourly rental in isolation.

## Optimize in this order

| Change | Likely benefit | What to watch |
| --- | --- | --- |
| Load the model once per worker | Avoid repeated startup and disk I/O | Cold starts still happen when workers are created |
| Use a compatible optimized runtime | Better kernels, batching, memory management | Exact architecture and format must be supported |
| Choose a smaller model that passes your tasks | Reduce weights and compute | Lost capability or weaker reliability on hard cases |
| Test lower-precision/quantized weights | Reduce weight memory and sometimes latency | Quality changes; unsupported or slow kernels |
| Limit unnecessary context/output | Reduce cache and work per request | Dropping required evidence or truncating useful answers |
| Bound and schedule concurrency | Keep the worker within memory/latency limits | Queues can still become too long |
| Cache repeated work with isolation | Avoid recomputing safe reusable results/prefixes | Correct keys, model versions, tenant access, and invalidation |

Do not start by reducing accuracy before checking whether your application reloads the model on every request. Bob's CLI is useful interactively, but it is not a persistent serving worker.

## Quantization: smaller numbers, measured tradeoffs

Quantization approximates some model values using fewer bits plus scaling information. Weight-only 4-bit quantization can leave computations and other tensors at higher precision. “4-bit model” does not mean every byte of runtime memory shrinks eightfold compared with FP32.

Consider a dense 8B model: ideal BF16 weights take 16 GB, while ideal 4-bit weights take 4 GB before overhead. The difference can leave much more room on a 24 GB GPU for cache and requests. But speed depends on kernels and memory traffic; dequantization overhead can erase gains in some workloads.

Use a quantization format and engine implementation supported on the exact GPU. AWQ/GPTQ-style checkpoints, bitsandbytes formats, FP8, and GGUF are not universally interchangeable. Check the [vLLM quantization compatibility documentation](https://docs.vllm.ai/en/latest/features/quantization/) for that engine. Do not assume an older GPU accelerates the same low-precision operations as a newer GPU.

[Transformers' bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes) describes its supported loading/quantization path. This is a separate ecosystem from Bob's current `TinyLM`; adding a generic `--load-in-4bit` flag to Bob would not implement that integration.

Start from a trusted, supported quantized artifact when learning serving. If doing your own calibration-based quantization, use representative calibration data and keep a separate evaluation set. Compare against the original model at the same prompts and decoding settings, including longer contexts and difficult examples. There is no universal promise that 4-bit is lossless or that 8-bit always meets your task's needs.

KV-cache quantization is a **separate** optimization. It can reduce memory pressure from long contexts/concurrency but needs its own runtime support and quality checks. Weight quantization does not apply it automatically.

## Use context deliberately

Long prompts cost prefill work and occupy cache. Retrieve only relevant passages; remove duplicated boilerplate; cap unnecessary generated output. Keep task-critical instructions and evidence. Summarization can save context but may omit details, so evaluate it as part of the application.

RAG can make a smaller model useful on a narrow knowledge task by supplying the right evidence. It does not guarantee reasoning accuracy or faithful use of that evidence. Check answer correctness, source use, and appropriate refusal when the retrieved material lacks the answer.

For coding, aggressive output caps may truncate working solutions. For structured extraction, a compact response schema can cut unnecessary text without losing the task result. Tune output limits to the product, not a single global number.

## Share a smaller GPU well

A 24 GB worker serving a compatible quantized 7–8B model is a reasonable starting experiment for modest traffic. Measure at concurrency 1, 2, 4, and higher only while meeting latency and memory targets. Do not translate “100 registered customers” into “100 simultaneous generations.”

If the model fits with adequate headroom, adding replicas can improve total capacity and fault tolerance. If it does not fit, replicas cannot fix that; each still needs to hold its own model. Splitting a model across GPUs is a different strategy with communication costs.

Prioritize interactive requests fairly and batch offline jobs separately when possible. Continuous batching may improve throughput, but a batch size that maximizes tokens/second can worsen individual response latency. Each tenant still needs quotas and access checks as described in [multi-tenant serving](04-multi-tenant-serving.md).

Prefix/response caches must include appropriate tenant scope, model/adapter version, and generation/request identity. Never mix conversation state to gain batching efficiency. Share read-only weights; keep independent prompts, buffers, and ownership metadata.

## CPU and offloading are options, with different goals

For a tiny model such as Bob, CPU inference can be entirely reasonable at low request rates. For supported larger models, [llama.cpp](https://github.com/ggml-org/llama.cpp) offers CPU and GPU-oriented inference paths and quantized model support. Its supported formats/architectures still matter; Bob's checkpoint is not directly interchangeable.

Moving layers or runtime state between CPU RAM and GPU memory can make an otherwise impossible model load, but transfers and CPU work may severely hurt latency. Offloading is often better for experimentation or low-rate batch work than a responsive multi-user service. Sufficient host RAM matters, and disk swapping is not a useful substitute for GPU bandwidth.

A fast 24 GB GPU that fits a smaller model may serve your task better than a huge model constantly moving data over PCIe. Benchmark the full application result rather than preferring the larger parameter count automatically.

## Adaptation, routing, and distillation

A carefully adapted smaller model may perform well on a narrow task. Fine-tuning does not guarantee it will match a larger model generally; test the tasks you actually offer. Distillation trains a smaller model using outputs or signals from a stronger one and is another separate training project, subject to the source model/data terms.

A two-tier service can send easy cases to a cheaper model and escalate difficult cases to a stronger model. The routing policy itself needs evaluation. A model saying “I am confident” is not automatically a calibrated signal. Account for the extra call's cost and latency, and track missed escalations as well as unnecessary ones.

For a first deployment, one model with a clear quality target is usually easier to reason about than a complex cascade.

## A quality-preserving benchmark recipe

1. Assemble representative held-out tasks: typical, difficult, long-context, and edge cases. Use task-specific scoring—tests for code, field accuracy for extraction, grounded-answer review for document QA.
2. Establish a reference with the original model, adequate precision, and low concurrency. Record decoding settings and runtime version.
3. Change one variable: GPU, quantization, model size, context policy, or concurrency.
4. Re-run quality checks and realistic load tests. Measure failures and tail latency, not just average speed.
5. Accept a change only if it meets your predeclared quality and latency limits at an improved total cost. Recheck on new traffic over time.

For initial learning, 50–100 thoughtfully chosen tasks can reveal obvious regressions, but it is not a statistical guarantee or sufficient coverage for every production use. Do not repeatedly tune against the only final test set.

Suggested comparison table:

| Candidate | Quality score | p95 latency | Peak VRAM | Successful requests/hour | Total cost/hour | Cost per successful request |
| --- | --- | --- | --- | --- | --- | --- |
| Reference precision | Measure | Measure | Measure | Measure | Actual rate | Calculate |
| Quantized model | Measure | Measure | Measure | Measure | Actual rate | Calculate |
| Smaller model | Measure | Measure | Measure | Measure | Actual rate | Calculate |

Define success to include the application's quality and latency requirements. In production, some quality measurements may be estimated through sampling rather than scored automatically per request.

```text
cost per successful request = total workload cost / successful requests
```

Include idle time, warm workers, storage, and applicable transfer charges. As an invented arithmetic example—not RunPod pricing—a $0.60/hour worker completing 600 acceptable requests costs $0.001/request. A $1.20/hour worker completing 2,400 costs $0.0005/request. The higher hourly price can be cheaper per result.

## A concrete progression for you

**Now:** keep Bob on your Mac to understand training. Test a 24 GB GPU when you need faster iterations; the existing config name `h100.json` is a preset name, not a hardware requirement.

**First useful text application:** choose a supported small pretrained model that passes your task tests, then compare original precision versus a supported quantized version on a 24 GB candidate. Limit context/concurrency at first and keep a fixed quality baseline.

**Growing demand:** compare a 48 GB worker with several 24 GB replicas using your actual request mix. Move to 80 GB or larger when weights/cache requirements or measured economics justify it.

**What Bob still needs:** persistent serving, caching/batching, lower-precision integration, and tenant controls are future work. These docs do not change `generate.py`, convert its weights, or enable automatic production scaling.
