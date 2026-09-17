# 01 — From text to a trained model

[Learning home](README.md) · Next: [The math](02-math-with-intuition.md)

## A model is a learned function

Your model takes a sequence of byte IDs and produces scores for the next byte at each position. Its 875,264 parameters are numbers that determine how it computes those scores. The architecture specifies the operations; training chooses their numerical settings.

A useful distinction:

| Object | What it is in Bob | Does it change while generating? |
| --- | --- | --- |
| Parameters / weights | Embeddings, linear matrices, biases, normalization scales | No |
| Activations | Temporary vectors calculated from this prompt | Yes |
| Hyperparameters | Width, layers, batch size, learning rate | No |
| Optimizer state | AdamW's running statistics used for updates | Not used in generation |
| Training data | The prepared bytes used as examples | Not accessed by generation |

Weights can encode patterns and can also memorize some examples. There is no simple lookup from one weight to a sentence. The exported text file is a readable list of model numbers, not a recovered copy of the training corpus.

## Why bytes?

`prepare.py` encodes text with UTF-8. Each token is a number from 0 to 255. ASCII letters usually occupy one byte; other characters may use multiple bytes. The model can therefore represent arbitrary UTF-8 text with a fixed vocabulary of 256 IDs.

That simplicity has a cost: a 128-byte context holds much less language than 128 word-piece tokens. A subword tokenizer learns reusable pieces such as word fragments, making many texts shorter in token units. Changing the tokenizer later also changes token meanings and usually the embedding/output sizes. It is not a switch you can flip on an existing byte checkpoint.

`prepare_code.py` retrieves code files through metadata before the same byte preparation. Pandas helps organize data; it does not perform the neural-network training. Code files keep their internal newlines. A `.bin` training file is a packed stream of bytes, while a `.pt` checkpoint is a serialized collection of tensors and training state. They serve different purposes.

## Where do the answers come from?

Language-model training creates targets from the text itself. If the document contains `hello`, we can construct:

| Position | Input byte shown here as a letter | Target |
| --- | --- | --- |
| 0 | h | e |
| 1 | e | l |
| 2 | l | l |
| 3 | l | o |

At position 2 the model may use `h`, `e`, and `l`; it must not use position 3. This is **next-token prediction**. No human needs to label each next letter separately, which is why the objective is called self-supervised.

In `train.py`, `batch()` takes a window with one extra byte, then uses `tokens[:, :-1]` as input and `tokens[:, 1:]` as target. The causal attention mask makes it possible to score all positions in parallel without letting earlier positions see later answers.

## What one training step means

A local batch contains 8 windows of 128 input bytes. That produces 1,024 next-byte prediction targets for one optimizer update.

1. Sample random windows from training bytes.
2. Run the model forward to produce scores.
3. Compare scores with the true next bytes using cross-entropy.
4. Backpropagate: calculate how each parameter affects the loss locally.
5. Clip unusually large gradients and let AdamW update the parameters.

The next update samples another batch. Since windows are sampled repeatedly, 500 updates do not mean 500 passes through the dataset. Bob does not organize its sampling into epochs. Repeated or overlapping windows also mean “tokens processed” is not “new information seen.”

## Why separate validation data?

A model can get better at examples it has seen without getting better at new ones. Validation checks a held-out set that does not contribute gradients. Bob splits documents before joining them and uses fixed validation windows for comparable measurements.

Exact deduplication helps, but related documents can still leak information across the split. Code files from the same repository can be on both sides. A serious coding evaluation should separate repositories or another meaningful unit, not just files.

Training loss here measures the current batch. Validation loss averages several batches. They need not move together at every printed step. Repeatedly selecting changes using the same validation set can also overfit your decisions to that set; later, keep a final test set untouched until evaluation.

## Why your first output was gibberish

Your reported validation loss fell from about 5.57 to 2.61. The model learned useful local statistics, but correctly favoring spaces, common letters, and familiar fragments is easier than producing coherent paragraphs.

During training, previous tokens come from real text. During generation, previous tokens increasingly come from the model itself. An awkward prediction changes the context for the next one. Small errors can accumulate. More steps may help, but data coverage, model capacity, context length, and the objective also matter.

`generate.py` repeatedly runs the model on the latest context, samples a byte, and appends it. It does not update any weights. Temperature changes sampling probabilities, not the learned knowledge. Your implementation recomputes the context each time; a future key/value cache could speed this up.

## Pretraining, fine-tuning, and prompting

**Pretraining from scratch** starts from random weights, as Bob does. **Continued pretraining** takes existing weights and applies a similar prediction objective to additional text. **Supervised fine-tuning** often uses prompt/response examples and trains desired response behavior, commonly restricting loss to answer tokens. **Prompting** changes the input while keeping weights fixed.

Bob can resume the same prepared dataset. Its strict metadata check currently prevents changing datasets during resume. That is an implementation safeguard, not a mathematical restriction on continued pretraining. Adding a separate “initialize from weights on new data” mode is a future project. It would need explicit choices about optimizer state, evaluation, and retention of previous capabilities.

## Check yourself

1. Does generating 1,000 bytes teach the model anything?
2. Is the byte at the same input position as the target allowed to be visible?
3. Does lower training loss alone prove better generalization?

**Answer hints:** (1) No optimizer updates occur. (2) The target is shifted one position forward; the current input is visible, the future target is not. (3) No—look at held-out performance and meaningful tasks.
