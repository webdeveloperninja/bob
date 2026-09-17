# 05 — Learn by experimenting, then build the next thing

[Learning home](README.md) · Previous: [Encoder–decoder](04-encoder-decoder.md) · Next: [Study guide](06-study-guide.md)

A useful experiment starts with a prediction you could discover is wrong. “A bigger model should be better” is too broad. “At the same number of updates on this dataset, doubling width reduces validation loss” is something you can measure.

Use the same prepared data for comparisons. Choose new output directories for fresh runs. Run the commands below from `bob` with `.venv` activated. Suggested future projects are not existing command-line features.

## Experiment 1: separate training from sampling

Use one checkpoint and change only temperature:

```bash
python generate.py --checkpoint runs/local/checkpoint.pt --prompt "The future of AI" --seed 42 --temperature 0.5
python generate.py --checkpoint runs/local/checkpoint.pt --prompt "The future of AI" --seed 42 --temperature 1.0
```

Write down what changed: repetition, variety, spelling, or apparent coherence. The weights and their validation loss have not changed. The same seed improves comparability but does not force the same sampled sequence when probabilities differ.

Try `--top-k 1` to choose the highest-scoring byte each time. Does deterministic output become repetitive? A plausible-looking sample is not sufficient evaluation, and a dull sample does not by itself prove training failed.

## Experiment 2: compare additional training

Continue your original checkpoint into a separate folder:

```bash
python train.py --resume runs/local/checkpoint.pt --minutes 5 --out runs/study-longer
python generate.py --checkpoint runs/study-longer/checkpoint.pt --prompt "The future of AI" --seed 42 --temperature 0.8
```

Compare with the original checkpoint using the same prompt and generation settings. Record validation loss and completed steps, not just minutes. On faster hardware, a five-minute run may perform many more updates. Resumed metrics show elapsed seconds for that invocation, not lifetime training time.

If training loss keeps decreasing but validation gets worse, more time may mainly improve memorization. Bob saves the latest checkpoint, not a separately selected best-validation checkpoint. Preserve runs you want to compare.

A rough exposure calculation is:

```text
prediction targets processed ≈ updates × batch_size × context_length
```

For 500 local updates: `500 × 8 × 128 = 512,000` targets. That includes overlaps and repeats; it is not 512,000 unique bytes.

## Experiment 3: one architecture change

Start with a copied config:

```bash
cp configs/local.json configs/study-width.json
```

Edit only `width` from 128 to 256. Keep `heads` at 4 so width remains divisible by head count. Then compare fresh runs:

```bash
python train.py --config configs/local.json --steps 1000 --out runs/study-baseline
python train.py --config configs/study-width.json --steps 1000 --out runs/study-width
```

Equal updates control the number of sampled batches, but the wider model uses more compute. An equal-time comparison asks a different question: which configuration does more for a fixed budget? Report which comparison you chose.

More width increases capacity and cost. It does not ensure improvement on a small dataset. Repeat promising comparisons with another training seed by changing `seed` in a fresh config. One lucky run is weak evidence.

## Experiment 4: what data teaches

Prepare your own text with at least 20 distinct nonempty lines, ideally far more. Start a separate model using `--data data/mine` and a fresh output folder, following the main README's instructions.

Choose a narrow topic or writing style and predict what patterns will become common. Compare examples qualitatively, but do not directly rank losses from different validation datasets as if difficulty were identical.

For code, add tests of generated programs in an appropriately isolated environment before claiming useful coding ability. A program that resembles code can still be wrong. The current generation script only emits text; it does not test programs.

## A small experiment log

Copy this into a personal Markdown note for each experiment:

```text
Question:
Prediction:
Data directory and source revision:
Checkpoint or fresh config:
What changed:
What stayed fixed:
Command:
Completed steps and elapsed time:
Training loss / validation loss:
Generation prompt, seed, temperature, top-k:
Observed behavior:
Conclusion and uncertainty:
Next experiment:
```

A good conclusion can be “this did not help under the conditions I tried.” Change one thing next rather than layering several changes and losing track of causes.

## Build progression after V1

These are proposed projects, not completed features. Choose one at a time.

| Project | What it teaches | A useful completion check |
| --- | --- | --- |
| Add a unigram/bigram baseline | What token frequency and short-range statistics can explain | Compare against the Transformer on the same held-out bytes |
| Plot metrics and preserve the best validation checkpoint | Evaluation and training diagnostics | Resume and generation can use either latest or best explicitly |
| Visualize one attention head on a short prompt | Queries, keys, masks, intermediate activations | Future positions have zero weight; valid rows sum to one |
| Add a subword tokenizer | Vocabulary/context tradeoffs | Save tokenizer identity with checkpoints and reject mismatches |
| Add “initialize from weights on new data” | Continued pretraining and forgetting | Evaluate on both old and new held-out data |
| Add a learning-rate schedule and gradient accumulation | Optimization and memory budgets | Check how many optimizer updates occur, not just batches |
| Build the paired reversal model | Encoders, cross-attention, padding, EOS | Exact-match accuracy on unseen strings |
| Fine-tune a small pretrained model in a separate project | Adaptation versus learning everything from scratch | Compare original and adapted model on defined tasks |

A byte bigram model predicts the next byte from only the current byte. If Bob cannot beat that baseline convincingly, investigate data, training, or implementation before adding complexity.

## Other concepts worth knowing

**Retrieval-augmented generation (RAG):** retrieve relevant documents and put their content into the prompt. Retrieval does not itself update model weights. This is useful when the information should remain inspectable or change frequently, but a generator must still be capable of using it.

**Instruction tuning:** teach response behavior with input/output examples. It is not accomplished merely by adding an encoder or increasing temperature.

**LoRA / parameter-efficient adaptation:** update a small set of added trainable parameters while keeping most pretrained weights fixed. Study this after you understand ordinary training and fine-tuning. It is not implemented here.

**Quantization:** represent some model values using fewer bits. This changes storage/computation tradeoffs; it is different from making the model smaller by removing layers. Bob does not currently quantize checkpoints.

**Scaling:** model size, data, optimization, and compute need to work together. An H100 increases available computation; it does not automatically give an 875k-parameter model more capacity or more relevant data.

## Check yourself

1. Can changing temperature fix an overfitted model's weights?
2. What differs between a same-time and same-step comparison?
3. Why evaluate both old and new data after continued pretraining?

**Answer hints:** sampling changes output choices, not weights. Time comparisons vary update count and hardware efficiency; step comparisons can vary total compute. New learning may improve the new domain while degrading earlier capabilities.
