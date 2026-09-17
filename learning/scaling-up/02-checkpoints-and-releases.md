# 02 — Save, package, and recover your weights

[Scaling Up](README.md) · Previous: [RunPod training](01-runpod-training.md) · Next: [Efficient inference](03-efficient-inference.md)

## Three artifacts with different purposes

| Artifact | Contents | Purpose |
| --- | --- | --- |
| Resume checkpoint | Weights, config, optimizer state, step, RNG state, data metadata | Continue training |
| Inference release | Weights, architecture/config, tokenizer contract, version information | Answer requests with a fixed model |
| Plain-text weights export | Numeric weights formatted as readable JSON | Study and inspection |

`export_weights.py` produces the third kind. Its large text output is not a performance optimization or the file to deploy for inference. Bob's binary checkpoint is already usable by `generate.py`.

In PyTorch, a `state_dict` maps parameter names to tensors. To reconstruct the model you also need matching architecture code and configuration. This distinction is described in [PyTorch's saving/loading guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## What Bob already saves

At step 1, each `eval_every` interval, and completion, `train.py` writes:

```text
runs/h100/
  checkpoint.pt
  metrics.jsonl
```

The checkpoint contains `model`, `optimizer`, `config`, `step`, `data`, and CPU `rng`. The save uses a temporary file and replaces the latest checkpoint when writing succeeds. This avoids exposing a partly written latest file during normal replacement; it is not protection against losing the disk or a guarantee of distributed-storage durability.

There is one latest checkpoint per run. No automatic history, best-model selection, or remote backup exists. Save a completed checkpoint under a versioned name before another experiment replaces it. Do not copy a temporary checkpoint while it is being written.

The CPU RNG state is enough for the existing CPU continuation test. It is not a universal exact-reproducibility guarantee across hardware or future additions such as CUDA dropout, different data loaders, or distributed workers.

## Preserve the resume path

After the run has completed on the Pod:

```bash
mkdir -p runs/h100-backup
cp runs/h100/checkpoint.pt runs/h100-backup/checkpoint.pt
cp runs/h100/metrics.jsonl runs/h100-backup/metrics.jsonl
python -m pip freeze > runs/h100-backup/environment.txt
sha256sum runs/h100-backup/checkpoint.pt
```

Choose a new backup directory for each retained version rather than repeatedly overwriting it. Copy the prepared data directory too if you need to resume elsewhere; a seed and dataset URL cannot recreate source files that later disappear. Save the source commit and container/template identifier in your run notes.

Transfer backups to your Mac or a separate durable storage service using your chosen transfer method. On macOS, compute the downloaded file's hash with:

```bash
shasum -a 256 runs/h100-backup/checkpoint.pt
```

Compare the full hash with the Pod's value. A matching hash verifies identical bytes, not whether the model is good. Load it and generate text as a restore test. For a full resume drill, copy both checkpoint and original prepared data and run a very short continuation into a **new** output folder.

A network volume survives Pod deletion, but an independent backup protects against other failures and accidental deletion. See [RunPod storage](https://docs.runpod.io/pods/storage/types) and [transfer methods](https://docs.runpod.io/pods/storage/transfer-files).

## Optional: make a smaller inference-only artifact

This tutorial snippet works with the existing `generate.py`. Run it from the project root after training. Change `source` to your checkpoint if it is elsewhere. It creates a new release directory and refuses to reuse an existing one.

```python
from pathlib import Path
import hashlib
import json
import torch

source = Path('runs/h100/checkpoint.pt')
release = Path('runs/releases/bob-v1')
saved = torch.load(source, map_location='cpu', weights_only=True)
release.mkdir(parents=True, exist_ok=False)
artifact = {
    'model': saved['model'],
    'config': saved['config'],
    'step': saved['step'],
    'tokenizer': 'utf8-bytes-v1',
}
weights_path = release / 'inference.pt'
torch.save(artifact, weights_path)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    'release': 'bob-v1',
    'architecture': 'TinyLM from model.py',
    'tokenizer': 'utf8-bytes-v1',
    'training_step': saved['step'],
    'weights_sha256': sha256_file(weights_path),
    'source_checkpoint_sha256': sha256_file(source),
    'torch_version': str(torch.__version__),
}
(release / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(release)
```

Save the snippet as a temporary Python file or paste it into a Python session. Then:

```bash
python generate.py --checkpoint runs/releases/bob-v1/inference.pt --prompt "The future of AI"
```

The artifact retains FP32 weights, so it produces the same model calculations when loaded under the same conditions. Its reduction in size comes from omitting optimizer and training metadata—not quantization. It **cannot resume training** with `train.py`; retain the full checkpoint for that.

`weights_only=True` is a loading restriction in `torch.load`; it does not automatically discard the optimizer dictionary. The snippet explicitly chooses which fields to save. Only load artifacts from sources you trust and keep the runtime patched.

## What belongs beside a release?

Package the exact `model.py`/inference code, config, tokenizer definition, environment specification, and evaluation results. Add source commit or archive hash, dataset revisions, generation defaults, intended use, and known limitations to the manifest or a release note. Bob's simple snippet does not fill all of these automatically.

A standard pretrained model may instead use a directory with tensor shards, model config, tokenizer files, and a chat template. A LoRA release also needs its exact compatible base-model revision. Renaming Bob's checkpoint or converting tensor storage format does not provide architectural compatibility with another engine.

## Publish a version, not a moving target

A future model registry can be a simple private artifact store plus a database of release IDs and checksums. Upload a complete version, verify it, and only then mark it ready. Start new workers with that immutable version, run evaluation/smoke requests, gradually route traffic, and keep the previous version available for rollback.

Inference workers should not watch a training directory and hot-load whatever file appears. In-flight requests need consistent weights. Training credentials and tenant training data also need not be present on serving workers.

**Restore exercise:** explain which files you need for inference only, and which additional files you need to resume training exactly as closely as this implementation permits.
