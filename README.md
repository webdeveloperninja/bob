# Bob: train your first language model with PyTorch

A small, readable project for training **your own Transformer from scratch** on a slice of NVIDIA's Nemotron sample dataset. All model weights start random. You train them, save them, and use them to continue a prompt.

This is a learning project: the local model has about 875,000 parameters. Expect mostly nonsense at first, then patterns, spaces, and fragments of words. A short run will not produce a capable chatbot. The dataset provides training text, not pretrained model weights.

## Learn the concepts

Start with the [learning library](learning/README.md) for a guided explanation of the math, training loop, attention, and model architecture. It includes an encoder–decoder build plan and an eight-stage study guide with resources for someone who has taken Calculus 1. This main README remains the command reference.

## Ready to try in this folder

The environment and NVIDIA data are already prepared on this Mac, and you have trained `runs/local/checkpoint.pt`. To generate text and then continue that model:

```bash
cd /Users/robertsmith/Desktop/bob
source .venv/bin/activate
python generate.py --prompt "The future of AI"
python train.py --resume runs/local/checkpoint.pt --minutes 5 --out runs/five-minutes
python generate.py --checkpoint runs/five-minutes/checkpoint.pt --prompt "The future of AI"
```

The sections below explain setup from a fresh copy. Skip preparation if `data/nemotron/meta.json` already exists. For existing training runs, use `--resume`; a fresh run refuses a nonempty output folder. The earlier sample is at `runs/first-check/checkpoint.pt`.

Jump to [timed training](#train-for-five-minutes-or-longer), [weight export](#export-the-weights-to-plain-text), [all command options](#complete-command-reference), [config settings](#training-config-reference), [pandas code data](#use-nvidia-code-v3-with-pandas), or [troubleshooting](#common-problems).

## 1. Set up locally

Open a terminal in this folder. Use Python 3.12 (recommended):

```bash
cd /Users/robertsmith/Desktop/bob
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment has already been created, just activate it. If you use `uv`, the equivalent setup is `uv venv --python 3.12` followed by `uv pip install -r requirements.txt`.

## 2. Prepare a small amount of NVIDIA data

```bash
python prepare.py
```

Internet is required for this step. We use the `Nemotron-CC-High-Quality` subset of [NVIDIA's dataset](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample). The sample has multiple subsets; V1 expects a string `text` column and does not handle chat-message schemas or mix subsets.

Preparation streams up to 1,000 unique nonempty documents, keeps at most 20,000 bytes per document, shuffles them with a fixed seed, and reserves 10% of documents for validation. It removes exact duplicates of the retained text before splitting. Related documents may still overlap; this is a basic learning split, not a benchmark.

Outputs:

- `data/nemotron/train.bin`: training bytes.
- `data/nemotron/val.bin`: held-out validation bytes.
- `data/nemotron/meta.json`: source revision, settings, counts, and checksums.

[Streaming](https://huggingface.co/docs/datasets/stream) avoids downloading the entire corpus, but the library may fetch larger chunks than the documents you keep. The selected sample is finite: asking for more documents does not create more data. Preparation prints the actual count. It refuses to overwrite a nonempty output folder.

## 3. Train

Try a short check first:

```bash
python train.py --steps 20 --out runs/first-check
```

Then start your first normal run:

```bash
python train.py
```

It automatically chooses NVIDIA CUDA, Apple Silicon MPS, or CPU, in that order. The default is 500 updates. Speed depends on your hardware; use the first check to estimate it. If MPS causes an error, run `python train.py --device cpu --out runs/cpu`.

At evaluation intervals, the script prints:

```text
{"step": 100, "train_loss": ..., "val_loss": ..., "seconds": ...}
```

Lower loss means better next-byte prediction. A uniform random predictor has loss around 5.55. Training loss is the current training batch; validation loss averages fixed windows from held-out documents. Falling training loss with rising validation loss suggests memorization. Compare runs on the same prepared data.

`runs/local/checkpoint.pt` contains weights, optimizer state, configuration, data metadata, step number, and CPU random state. It is saved after step 1, every 100 steps, and the final step. `metrics.jsonl` records measurements. Ctrl+C stops training; resume from the last saved checkpoint (work since that save is lost).

## 4. Generate text

```bash
python generate.py --prompt "The future of artificial intelligence"
```

For the short check instead:

```bash
python generate.py --checkpoint runs/first-check/checkpoint.pt --prompt "Hello"
```

The generator predicts one byte at a time and feeds it back into the model. `--bytes 500` generates more; `--temperature 0.6` makes sampling less random. `--top-k 40` limits sampling to the 40 highest-scoring bytes. Invalid UTF-8 from an untrained model appears as replacement characters. There is no chat interface or stop token; it generates the requested number of bytes.

## 5. Continue training

```bash
python train.py --resume runs/local/checkpoint.pt --steps 1000
```

`--steps` is the **total target**, so this adds 500 updates to a 500-step checkpoint. Resume uses the saved architecture and training settings, plus the original prepared data; `--config` is ignored. Use `--data` and `--out` explicitly when resuming a nondefault run. Resuming across devices is supported, but results need not be numerically identical across hardware.

## Train for five minutes (or longer)

Continue your existing local model for approximately **five additional minutes**:

```bash
python train.py --resume runs/local/checkpoint.pt --minutes 5 --out runs/five-minutes
python generate.py --checkpoint runs/five-minutes/checkpoint.pt --prompt "The future of AI"
```

This keeps the original checkpoint for comparison and saves the longer run separately. To add another ten minutes to that run:

```bash
python train.py --resume runs/five-minutes/checkpoint.pt --minutes 10 --out runs/five-minutes
```

To start a new model from random weights instead:

```bash
python train.py --minutes 5 --out runs/fresh-five-minutes
```

`--minutes` accepts positive decimals, overrides the config's `steps`, and cannot be combined with `--steps`. The timer starts after setup and checkpoint loading; training, evaluation, and saving count toward it. The script finishes the current update, evaluates, and saves when the time is reached, so the total can run slightly over budget. Regular checkpoints still happen at `eval_every` intervals. Each invocation gets a fresh time budget.

Five minutes means many more updates than your first 500-step run. That can improve spelling and word patterns, but it can also memorize this small dataset. Watch **validation loss**, and compare output using the same prompt, temperature, and seed. Longer training alone does not guarantee coherent sentences or chatbot behavior.

## Export the weights to plain text

```bash
python export_weights.py
```

This reads `runs/local/checkpoint.pt` and writes `runs/local/weights.txt`. It is plain text in JSON format: parameter names, tensor dimensions, and **every numeric weight**, without ellipses or truncation. It also includes the training step and model config. Optimizer state is excluded. Keep the `.pt` file for training and generation; the text export is for inspection.

The initial export already exists in this folder: `runs/local/weights.txt` (about 28 MB). Open it in a text editor and search for `tokens.weight` or `blocks.0.qkv.weight`. An existing output is never overwritten. After more training, export to a new filename:

```bash
python export_weights.py --checkpoint runs/five-minutes/checkpoint.pt --out runs/five-minutes/weights.txt
```

These are learned numbers, not the original training sentences. This exporter is intended for this tiny model; dumping a very large model to text takes substantial memory and disk space.

## How the model learns

1. **Text → bytes.** UTF-8 encodes text as integers from 0 to 255. No tokenizer download or training is needed. Bytes are less efficient than the subword tokens used by larger language models.
2. **Bytes → vectors.** Learned embeddings represent each byte and its position.
3. **Transformer → predictions.** Causal self-attention lets each position look only at itself and earlier positions. Small feed-forward networks process the result.
4. **Compare with the next byte.** For a sequence like `hello`, input `hell` has target `ello`. Cross-entropy measures prediction error.
5. **Update weights.** Backpropagation calculates gradients. AdamW adjusts the weights. Repeat with random windows of training data.

Documents are joined with two newlines. Windows can cross these boundaries. This keeps V1 simple; there is no special end-of-document token. The context length is measured in **bytes**, not words. Generation retains only the latest context window.

Read the code in this order:

| File | What to learn |
| --- | --- |
| `prepare.py` | Loading, splitting, encoding, and recording data |
| `prepare_code.py` | Read code metadata with pandas and fetch source text |
| `model.py` | Embeddings, causal attention, residual connections, output logits |
| `train.py` | Batches, loss, gradients, evaluation, checkpoints |
| `generate.py` | Autoregressive sampling from your weights |
| `export_weights.py` | Export all weights as readable JSON text |
| `configs/local.json` | Small local training settings |
| `configs/h100.json` | Larger single-GPU experiment |

## Experiments to try

Copy the local config and change one setting at a time. Start a new output folder for each run:

```bash
cp configs/local.json configs/experiment.json
# Edit configs/experiment.json, then:
python train.py --config configs/experiment.json --out runs/experiment
```

- `steps`: more optimizer updates; try 1,000 before increasing the model.
- `context_length`: bytes visible at once; longer contexts use more memory.
- `batch_size`: windows per update; lower this first if memory runs out.
- `width` / `layers`: model capacity; bigger models need more compute and data.
- `heads`: attention heads; width must be divisible by this number.
- `learning_rate`: update size; reduce it if loss becomes unstable.

Data preparation is separate from training. To try more data without replacing your first split:

```bash
python prepare.py --documents 5000 --out data/larger
python train.py --data data/larger --out runs/larger-data
```

The script uses the first available unique documents before shuffling; this is not a random sample of the entire source. Training samples random windows repeatedly rather than processing formal epochs.

## Later: use one H100 on RunPod

No cloud resources are created by this project. The H100 config is a larger learning experiment, not a benchmark or an attempt to fully utilize an H100. First prove the small run works before paying for a bigger run.

1. Create a GPU Pod with one H100 and a PyTorch/CUDA template. Check its Python version is supported by the requirements (3.12 recommended) and CUDA PyTorch works.
2. Put the project in `/workspace/bob` using your preferred upload, Git repository, or SSH copy. Copy code/configs/requirements; **do not copy the Mac `.venv`**.
3. Use a network volume mounted at `/workspace` if you want files to survive Pod deletion. A Pod volume survives stopping but is deleted on termination; see [RunPod storage](https://docs.runpod.io/pods/storage/types).
4. Run in the Pod terminal:

```bash
cd /workspace/bob
# Reuse the template's CUDA-enabled PyTorch installation.
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python prepare.py --documents 5000 --out data/h100
python train.py --config configs/h100.json --data data/h100 --out runs/h100 --device cuda
python generate.py --checkpoint runs/h100/checkpoint.pt --device cuda
```

The training script enables BF16 autocast on compatible CUDA hardware and clips gradients. V1 uses one GPU, no distributed training, no scheduler, and no gradient accumulation. If the CUDA check fails, use a compatible CUDA PyTorch template/build before training; consult the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Resume that run with:

```bash
python train.py --resume runs/h100/checkpoint.pt --data data/h100 --out runs/h100 --steps 6000 --device cuda
```

To continue your **local model** on the H100, copy `data/nemotron/` and `runs/local/checkpoint.pt` as well as the code, then use the same local resume command with `--device cuda`. It keeps the small architecture. The bigger H100 configuration starts a new model; it cannot enlarge a saved model.

Download your checkpoint, metrics, and prepared data/metadata before deleting storage. Stop or terminate the Pod when finished; check RunPod's displayed compute and storage charges.

## Checks and offline practice

```bash
python -m unittest discover -s tests -v
```

Tests check that attention cannot see the future, a simple pattern can be learned, and preparation → training → checkpoint resume → generation works. CPU resume is compared with uninterrupted training. Tests also check timed continuation and exact weight export. Tests use synthetic local text and need no network.

You can also prepare your own UTF-8 text file, with one document per line (at least 20 unique nonempty lines):

```bash
python prepare.py --text-file my_text.txt --out data/mine
python train.py --data data/mine --out runs/mine
```

## Use NVIDIA Code v3 with pandas

The linked [Nemotron-Pretraining-Code-v3 dataset](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3) contains **metadata**, not the source code itself. Its table has `repo`, `rel_path`, `language`, and `commit_id`. For training on code, we must retrieve the referenced file contents.

`prepare_code.py` handles that extra step:

1. Read one Parquet shard in batches of up to 1,000 rows using PyArrow.
2. Convert each batch to a **pandas DataFrame** and optionally filter by language.
3. Fetch a limited number of files from GitHub at the recorded commits, without executing them.
4. Apply the same deduplication, byte encoding, and document split as `prepare.py`.
5. Create `train.bin`, `val.bin`, and `meta.json`, including source URLs and retrieval details.

Install the updated dependencies if needed, then preview the metadata:

```bash
python -m pip install -r requirements.txt
python prepare_code.py --inspect
```

Prepare up to 100 unique files, then train a new model for five minutes:

```bash
python prepare_code.py --documents 100 --out data/nemotron-code
python train.py --data data/nemotron-code --minutes 5 --out runs/code
python generate.py --checkpoint runs/code/checkpoint.pt --prompt "body {"
```

The beginning of `part_00000.parquet` contains CSS files, so this first experiment learns CSS patterns. It does not automatically produce a Python coding assistant. The model still starts from random weights.

Resume with the matching data directory:

```bash
python train.py --resume runs/code/checkpoint.pt --data data/nemotron-code --minutes 5 --out runs/code
```

A language filter is available, for example `--language Python`, but the chosen shard and scanned rows must actually contain enough matching files. The script scans at most `--max-rows` metadata rows **before** filtering. If too few matches are found, choose a different `--file` from the dataset's Files tab or increase `--max-rows`. It does not automatically search all shards. Language matching ignores capitalization. `--inspect` previews the first rows without applying the language filter.

Your pandas example is valid for exploring a whole shard:

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/nvidia/Nemotron-Pretraining-Code-v3/"
    "Nemotron-Code-Metadata/part_00000.parquet"
)
print(df.head())
print(df.columns.tolist())
print(df["language"].value_counts())
```

Run that Python snippet in a Python session or save it as a `.py` file. `df` is a table of metadata; there is no code-text column to feed straight into training. [pandas.read_parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) loads the selected table before `head()` runs. The first shard is approximately 129 MB compressed and can occupy substantially more RAM. The preparation script instead uses [PyArrow batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html) converted to pandas. Underlying Parquet/network buffers may still read more than one batch; row limits are not exact download-byte limits.

Fetching needs internet access to both Hugging Face and GitHub. Missing, inaccessible, binary, or empty files are skipped. Limits cap the rows scanned, file requests, and bytes retained per file; these are small experiments, not a full-corpus downloader. You need at least 20 unique usable files. If fewer than requested are available but at least 20 remain, it prepares the smaller set and reports actual counts. A failed preparation with fewer than 20 does not write a training dataset.

One fetched code file is one document, including its internal newlines. The split is by file, not repository; related files may appear on both sides, so validation is not an independent coding benchmark. `meta.json` records successfully fetched candidates (which may include duplicates later removed), their URLs and content hashes, the NVIDIA revision, and retrieval counts. Reuse the prepared binaries for exact future training runs, since GitHub files can become unavailable. The metadata dataset's terms and each fetched repository's code license are separate; the loader does not resolve source-code licenses.

**Already verified here:** `data/code-check` contains 20 real CSS documents fetched using this loader (18 train, 2 validation; one unavailable file was skipped). This is a pipeline check, not a useful coding dataset on its own.

## Complete command reference

Run commands from the `bob` folder with `.venv` activated. Paths are relative to your current terminal folder. Names such as `runs/five-minutes` are your choice; keep the same path when generating or resuming that run.

Every script supports `-h` or `--help` to list its options without doing any work:

```bash
python prepare.py --help
python prepare_code.py --help
python train.py --help
python generate.py --help
python export_weights.py --help
```

### `python prepare.py` — create training data

| Option | Default | Meaning |
| --- | --- | --- |
| `--subset` | `Nemotron-CC-High-Quality` | NVIDIA dataset subset; must contain a string `text` field. |
| `--revision` | `main` | Dataset branch, tag, or commit. The resolved commit is recorded in metadata. |
| `--documents` | `1000` | Maximum unique nonempty documents to keep; must be at least 20. The source may have fewer. |
| `--max-bytes-per-document` | `20000` | Maximum UTF-8 bytes retained per document; must be positive. |
| `--seed` | `42` | Seed for shuffling documents before the train/validation split. |
| `--out` | `data/nemotron` | Destination directory; must be absent or empty. |
| `--text-file` | None | Local UTF-8 file, one document per line, instead of downloading NVIDIA data. Subset/revision do not apply. |

The dataset repository itself is a constant in `prepare.py`, not a command-line option. Reuse the saved revision and settings to reproduce preparation against the same source version.

### `python prepare_code.py` — pandas metadata to code training data

| Option | Default | Meaning |
| --- | --- | --- |
| `--inspect` | Off | Print up to 10 metadata rows; no GitHub code retrieval or dataset output. |
| `--file` | `Nemotron-Code-Metadata/part_00000.parquet` | One shard path inside NVIDIA Code v3. |
| `--revision` | `main` | Dataset branch/tag/commit, resolved to a recorded commit. |
| `--language` | All | Case-insensitive language filter for fetching, such as `CSS` or `Python`. |
| `--documents` | `100` | Maximum unique usable files to retain; at least 20. |
| `--max-rows` | `10000` | Maximum metadata rows examined before filtering. |
| `--max-fetches` | `300` | Maximum candidate fetch attempts, including failures. |
| `--max-bytes-per-document` | `20000` | Positive byte limit per source file. |
| `--seed` | `42` | Document shuffle/split seed. |
| `--out` | `data/nemotron-code` | New or empty output directory. |

Only Code v3's metadata schema is supported by this new script. Use `prepare.py` for the original text dataset or your own text file. Both produce the same training-file format.

### `python train.py` — train or resume

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | `configs/local.json` | Training settings for a fresh model; ignored on resume. |
| `--data` | `data/nemotron` | Directory with `train.bin`, `val.bin`, and `meta.json`. Resume requires the same data. |
| `--out` | `runs/local` | Where checkpoints and metrics are written. Not inferred from `--resume`. |
| `--device` | `auto` | `auto`, `cpu`, `mps` (Mac GPU), or `cuda` (NVIDIA GPU). Explicit devices must be available. |
| `--steps` | Config value | Total target updates, including completed ones. Must exceed the saved step when resuming. |
| `--minutes` | None | Additional minutes for this invocation; positive decimals allowed. Overrides config steps. Cannot combine with `--steps`. |
| `--resume` | None | Existing `.pt` checkpoint to continue; restores weights, optimizer, saved config, and CPU random state. |

With neither duration option, config `steps` applies. With `--resume`, that means the checkpoint's saved config. After a timed run, use `--minutes` again or explicitly choose a `--steps` target greater than the saved step.

A fresh run refuses a nonempty output folder. A resumed run replaces `checkpoint.pt` and appends to `metrics.jsonl` in the chosen output folder. Use the same run folder or a new folder; avoid pointing it at an unrelated experiment. There is one latest checkpoint per run, not an archive of every save or a separately selected best model.

### `python generate.py` — continue a prompt

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `runs/local/checkpoint.pt` | Trained model to load. |
| `--prompt` | `The future of artificial intelligence` | Nonempty starting text; quote prompts containing spaces. |
| `--bytes` | `300` | Number of additional bytes to generate; zero or greater. |
| `--temperature` | `0.8` | Positive sampling temperature; lower values favor likely predictions, higher values add variety. |
| `--top-k` | `40` | Sample among this many top byte predictions; integer from 1 to 256. `1` always chooses the highest-scoring byte. |
| `--seed` | `42` | Sampling seed. Keep it fixed when comparing runs on the same hardware. |
| `--device` | `auto` | `auto`, `cpu`, `mps`, or `cuda`, as in training. |

Generation only reads the checkpoint; it does not train or change weights.

### `python export_weights.py` — read weights as text

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `runs/local/checkpoint.pt` | Checkpoint whose model weights will be exported. |
| `--out` | `runs/local/weights.txt` | Plain-text JSON destination; must not already exist. Parent folders are created. |

`model.py` supplies the architecture to the other scripts; it has no command-line interface. To run the offline test suite, use `python -m unittest discover -s tests -v`.

## Training config reference

These are JSON settings, **not command-line flags**. Copy a config, edit it, and pass it with `--config` for a fresh run. Resume restores saved settings instead.

| Setting | Local default | Meaning |
| --- | --- | --- |
| `context_length` | `128` | Bytes in each input window. Both data splits must be longer than this. |
| `width` | `128` | Embedding/hidden vector size; must divide evenly by `heads`. |
| `layers` | `4` | Number of Transformer blocks. |
| `heads` | `4` | Attention heads in each block. |
| `batch_size` | `8` | Windows processed per optimizer update. |
| `steps` | `500` | Total update target unless overridden with `--steps` or `--minutes`. |
| `learning_rate` | `0.0003` | AdamW update scale. |
| `eval_every` | `100` | Evaluate and save every this many updates; also at step 1 and completion. |
| `eval_batches` | `10` | Held-out batches averaged for validation loss. Larger values take more time. |
| `seed` | `42` | Initial weights and training-window random seed for a fresh run. |

Counts and learning rate must be positive. Seeds control different stages: preparation shuffling, training, and generation each have their own seed. The vocabulary is fixed at 256 byte values in the code.

## Common problems

| What you see | What to do |
| --- | --- |
| Missing `torch` or `datasets` | Activate `.venv`, then install `requirements.txt` in that environment. |
| Checkpoint not found | Train first, or pass `--checkpoint` pointing at an existing run. The sample is `runs/first-check/checkpoint.pt`. |
| Output directory is not empty | For training, resume that run or choose a new `--out`; for preparation, choose a new data folder. |
| Export output already exists | Choose a different text filename with `--out`. |
| Step target is already reached | Resume with `--minutes 5` or a larger total `--steps` target. |
| Resume data mismatch | Pass the original prepared dataset with `--data`. A new dataset needs a fresh run in V1. |
| MPS/CUDA unavailable | Use an available device or `--device cpu`. |
| Out of memory | For a fresh run, lower `batch_size` or `context_length` in a copied config. Config changes do not apply to resume. |
| Generated text is gibberish | Compare validation loss after more training; this tiny byte model has limited capacity. More time can also lead to memorization. |
| Interrupted training | Resume the most recently saved checkpoint; updates since that save are lost. |

## Dataset terms

The data remains governed by NVIDIA's [dataset card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) and linked [NVIDIA Open Data License](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md). The card also identifies potential upstream model terms for distributing trained models. Read those terms before sharing a model or dataset. This repository does not relicense NVIDIA's data.

## Verified on this machine

Checked with Python 3.12.11, PyTorch 2.14.0, Datasets 4.8.5, and NumPy 2.5.3 on CPU. All three offline tests passed; installed dependencies passed `pip check`. Real NVIDIA preparation yielded 777 unique documents (700 train / 77 validation) at revision `3ad096e6394e487bb4f778733300da85275bb449`. A 20-step real-data run reduced validation loss from 5.57 to 3.90. This only verifies that the learning pipeline works; it does not demonstrate useful language ability. You also reported a successful 500-step MPS run on your MacBook Air, reaching validation loss about 2.61. H100 execution has not been tested here.
