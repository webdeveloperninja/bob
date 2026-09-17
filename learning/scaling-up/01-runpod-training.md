# 01 — RunPod and larger training runs

[Scaling Up](README.md) · Next: [Checkpoints and releases](02-checkpoints-and-releases.md)

## Decide what “bigger” should achieve

There are two different goals:

| Goal | Approach | What you gain |
| --- | --- | --- |
| Understand model training | Increase Bob's width, layers, context, and data gradually | Visibility into architecture and optimization |
| Build a useful specialized application sooner | Evaluate a pretrained model, then fine-tune if needed | Existing language/code capability rather than random initialization |

A bigger Bob model remains a byte-level next-token model. It will not become a chat assistant simply because it runs on an H100. Define a task and an evaluation set before allocating more compute.

For adaptation, full fine-tuning updates all parameters. LoRA adds smaller trainable updates to selected layers of a base model; its artifact depends on that base model and revision. It can reduce training-state requirements, but base weights and activations still need memory. See the [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). Neither full pretrained-model loading nor LoRA is implemented in Bob.

## Move the existing project to RunPod

These steps use a **Pod**, a machine you operate, for an interactive single-GPU training job. Choose an H100 PyTorch/CUDA template with a supported Python version, preferably 3.12. Check the listed GPU memory, available disk, and current price in the console; do not assume all offerings or templates are identical.

Choose a network volume at Pod creation if you want `/workspace` to persist independently of that Pod. A normal Pod volume survives stops but is deleted on termination; container storage is temporary. See [RunPod storage options](https://docs.runpod.io/pods/storage/types). Keep an independent backup of valuable checkpoints.

Upload project source to `/workspace/bob` using the console's available connection details and [RunPod's file-transfer guide](https://docs.runpod.io/pods/storage/transfer-files). Do not upload the Mac `.venv` as a Linux runtime.

One source-packaging option on your Mac is:

```bash
cd /Users/robertsmith/Desktop/bob
git archive --format=tar.gz --output=/tmp/bob-source.tar.gz HEAD
```

This includes **committed files only**. Commit any changes you intend to ship first. Transfer that archive by your chosen method, then extract it on the Pod:

```bash
mkdir -p /workspace/bob
tar -xzf /workspace/bob-source.tar.gz -C /workspace/bob
cd /workspace/bob
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

The archive command does not include Git history, ignored datasets, or checkpoints. Transfer prepared `data/` and the desired checkpoint separately if resuming. Record the source commit with `git rev-parse HEAD` on the Mac. For reproducibility, record the template/container image and installed versions as well. The broad version ranges in `requirements.txt` are convenient for learning, not an exact environment lock.

## Prove the small workflow before a long run

If you copied the original prepared data, skip preparation. Otherwise:

```bash
python prepare.py
python train.py --config configs/h100.json --steps 20 --device cuda --out runs/h100-check
python generate.py --checkpoint runs/h100-check/checkpoint.pt --device cuda --prompt "The future of AI"
```

Then run a new 30-minute experiment:

```bash
python train.py --config configs/h100.json --minutes 30 --device cuda --out runs/h100
```

If `tmux` is available in the template, start the run inside `tmux new -s bob-training`. Detach with Ctrl+B, then D; reattach with `tmux attach -t bob-training`. This helps a job survive an SSH disconnect. It does not survive machine loss or replace checkpointing.

Resume the saved run for another 30 minutes:

```bash
python train.py --resume runs/h100/checkpoint.pt --minutes 30 --device cuda --out runs/h100
```

For code data, prepare with `prepare_code.py` and pass the matching `--data` directory to every training/resume command. The first code shard begins with CSS; model size does not change the source distribution.

## How large is the next step?

Actual parameter counts from Bob's implementation:

| Configuration | Width / layers / heads | Context | Parameters |
| --- | --- | --- | --- |
| Local, existing | 128 / 4 / 4 | 128 bytes | 875,264 |
| H100, existing | 384 / 6 / 6 | 256 bytes | 10,942,464 |
| Proposed next experiment | 768 / 12 / 12 | 512 bytes | 85,842,432 |

The proposed config is an educational suggestion, **not an H100 benchmark or fit/performance guarantee**. To try it later, copy `configs/h100.json` to `configs/larger.json`, set those four architecture values, and initially reduce `batch_size` to 8. Start with `--steps 20 --out runs/larger-check`, measure memory, and increase the batch only after it works. A changed architecture starts a new model; resume cannot enlarge a checkpoint.

Parameter count in this model is dominated by matrices scaling roughly with `layers × width²`. Attention adds context-dependent compute. More capacity without enough diverse, relevant training examples can lead mainly to memorization.

## Understand the memory budget

For Bob's FP32 parameters and AdamW state, an approximate steady-state accounting is:

```text
weights:          4 bytes × parameter count
gradients:        4 bytes × parameter count
two Adam moments: 8 bytes × parameter count
subtotal:        16 bytes × parameter count
```

At 85.8M parameters, this subtotal is about 1.37 GB in decimal units. **It is not total GPU memory.** Add activations, attention/kernel workspaces, temporary optimizer buffers, runtime overhead, and allocator reserve. Save/loading can also use substantial host RAM. BF16 autocast changes selected operation dtypes; Bob still stores FP32 parameters and does not automatically halve every memory component.

Observe `nvidia-smi` during a short run. Low GPU utilization can come from CPU preparation, tiny batches, data transfer, or frequent evaluation/checkpointing. Bigger batches can help throughput, but they also change the optimization process. Compare useful validation progress per unit of compute, not utilization alone.

## Limits you should address before much larger corpora

Bob keeps collected documents in memory while preparing them and hashes each binary via `read_bytes()` at training startup. It is not an out-of-core preparation/verification pipeline for terabytes. Its document cap and byte truncation are deliberate V1 simplifications.

It also lacks gradient accumulation, learning-rate scheduling, activation checkpointing, and distributed training. Renting several GPUs does not make this script use them. DDP would replicate a model across GPUs to train on different batches; model/state sharding solves different memory problems and requires a more substantial training framework.

Prepare a representative dataset and test on genuinely held-out examples. Repeating the 777-document sample longer is not equivalent to obtaining a larger corpus. Token counts here are byte tokens, so do not directly apply subword-token budgets from another model.

## Bound the experiment

Before a rental, decide the question, maximum runtime, checkpoint interval, and success criterion. Estimate expense using the console's current compute rate multiplied by runtime, plus storage and applicable transfer charges. Confirm the checkpoint can be restored, copy it off the Pod, then stop or terminate the resources you no longer need. Retained storage can continue to have charges.

**Checkpoint question:** if the machine disappears midway through the run, which file on which independent storage lets you continue?

## Compare hardware before renting

The [hardware guide](05-choosing-runpod-hardware.md) separates full training from QLoRA and serving. An H100 is not required for the preset named `h100.json`; start with the least costly candidate that meets your measured experiment needs.
