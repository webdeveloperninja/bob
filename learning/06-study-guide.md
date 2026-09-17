# 06 — A study guide for understanding and building

[Learning home](README.md) · Previous: [Experiments](05-experiments-and-building.md)

Aim to explain the system and run sound experiments. You do not need to master every theorem or reproduce a frontier model. Your Calculus 1 background is enough to begin; fill gaps when they become relevant.

## A flexible eight-stage path

Think of each stage as roughly a week with two or three focused sessions, but take longer whenever needed. A session can be 30–60 minutes: learn one idea, connect it to code, then explain it without notes. These are suggested study blocks, not a promise about mastery time.

| Stage | Focus | Do or build | Ready to move on when… |
| --- | --- | --- | --- |
| 1 | Data, tokens, train versus generate | Read chapter 01; generate from an existing checkpoint | You can explain why generation does not change weights |
| 2 | Vectors, shapes, linear layers | Read math sections 1–2; trace the shape table in chapter 03 | You can distinguish sequence length from feature width |
| 3 | Probability, softmax, cross-entropy | Read math sections 3–4; compare probabilities 0.5 and 0.01 for the true target | You can explain why assigning the truth tiny probability is costly |
| 4 | Gradients and backpropagation | Run the scalar gradient example; follow a small autodiff lesson | You can explain the chain rule's role and why gradients are cleared |
| 5 | Attention and causal masking | Read chapter 03; run its checkpoint shape example; inspect the causality test | You can explain Q/K/V and draw the allowed-position triangle |
| 6 | Generalization and experiments | Run one controlled experiment from chapter 05 | Your notes separate observation from assumptions and use held-out data |
| 7 | Encoder–decoder models | Read chapter 04; design paired reversal examples and masks | You can identify source, target input, labels, and cross-attention inputs |
| 8 | Build a small extension | Choose one project from chapter 05; implement and evaluate it | You can explain the change, test it, and describe its limitations |

Start stages 1–3 with the local chapters, not a long playlist. Use a resource below when a concept remains unclear. Building one small thing you can explain is more useful than collecting ten courses you never finish.

## The math bridge after Calculus 1

### Linear algebra: learn this next

Focus on vectors, matrix shapes, linear combinations, dot products, norms, and matrix multiplication as a transformation. Later learn bases, projections, eigenvectors, and singular values when you need them. You do not need to begin by calculating large determinants.

For intuition, use [3Blue1Brown's linear algebra collection](https://www.3blue1brown.com/?topic=linear-algebra). For structured lectures and exercises, use [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). Choose early material on vectors and matrix operations first rather than treating completion of the full course as a prerequisite.

**Enough for now:** given an input `[B, T, D]` and a linear layer from `D` to `4D`, you can predict the output dimensions and explain what the layer learns.

### Multivariable calculus: focus on sensitivity

Carry your Calc 1 chain rule into partial derivatives, gradients, and functions with several inputs. A Jacobian is a table of output sensitivities to inputs; recognize what it represents before trying to compute large ones manually.

[MIT 18.02SC Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) includes a Partial Derivatives unit, especially the part on the chain rule, gradient, and directional derivatives. Those are the useful next pieces for this project. Surface integrals and vector-calculus theorems can wait for other applications.

**Enough for now:** you can interpret `∂loss/∂weight`, explain a small step in the negative-gradient direction, and follow the scalar autograd example.

### Probability and statistics: learn uncertainty and evidence

Prioritize conditional probability, random variables, expectation, variance, likelihood, and sampling. Then add entropy/cross-entropy and uncertainty in evaluation. These ideas explain both the training objective and why a single generated sample is weak evidence.

[Harvard Stat 110](https://stat110.hsites.harvard.edu/) is a deeper probability resource. Start with conditional probability and expectation. It is a full course; you can select topics as needed rather than pausing model building until you finish it.

**Enough for now:** you can distinguish “probability assigned to the true byte” from “fraction of predictions whose top choice was correct.”

### What can wait?

For this learning goal, advanced integration techniques, measure theory, rigorous convergence proofs, and differential equations are not prerequisites. They become useful for particular research directions. Logs, exponentials, vector operations, the chain rule, and basic probability have a more immediate payoff here.

## A short resource shelf

These links were checked when these notes were created. Use the sections named below; there is no need to read everything end to end.

| Resource | Why use it | Suggested selection |
| --- | --- | --- |
| [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Understand the library operations behind Bob | Tensors, building a model, automatic differentiation, optimization, saving/loading |
| [3Blue1Brown: Attention](https://www.3blue1brown.com/lessons/attention/) | Visual reinforcement of the attention mechanism | Watch after reading chapter 03; pause to identify Q, K, V |
| [Andrej Karpathy: Neural Networks, Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Build small neural networks and work toward a language model | Start with micrograd; continue through relevant makemore lessons, then the GPT build |
| [Dive into Deep Learning](https://d2l.ai/) | A searchable textbook connecting math and implementations | Preliminaries, MLPs, optimization, and attention chapters; choose PyTorch examples |
| [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | Read the original architecture description | Figure 1 and Section 3 first; skip benchmark details on the first pass |

The mathematical prerequisites above are additional references, not extra courses you must complete in parallel. Work on one main resource at a time. Keep larger tutorial projects in separate environments so their dependency requirements do not disrupt Bob.

## How to read code without getting lost

Start with `TinyLM.forward` in `model.py`. Write the shape after each operation. Then read `Block.forward`, connecting its QKV projection, attention, residual paths, and MLP to chapter 03. Only then inspect `train.py`'s optimizer and checkpoint details.

When a line is confusing, ask three questions: What is the input shape? What is the output shape? Does this operation contain learned parameters or just rearrange values? For example, `transpose` changes axes; a learned `Linear` changes representations.

The [official PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) can help with library mechanics. These local notes explain this project's design choices.

## How to read a paper without needing a PhD

First identify the task and the problem the authors were trying to solve. Draw the input/output diagram. Find the training objective and ask what information is available at each position. Then read one core equation, naming every symbol and dimension.

Read experiments afterward: what was compared, what was held fixed, and which metric was reported? Separate “the paper reports improvement in this setup” from “this architecture must be better for every task.” Details that do not answer your current question can be left for a second pass.

For the original Transformer paper, ask: where does the encoder output go, which attention sublayer is causal, and why can training process target positions in parallel? Being able to answer those is a valuable first reading.

## Capstone choices

Choose one after the eight stages:

- **Explain Bob:** annotate the shapes in every forward-pass line, present a five-minute explanation of the loss and causal mask, and show one controlled training comparison.
- **Add continued pretraining:** design a new-data initialization mode, preserve the original checkpoint, and evaluate retention on the old dataset as well as improvement on the new one.
- **Build encoder–decoder reversal:** follow chapter 04, implement padding/BOS/EOS, and report exact-match results on unseen strings and different lengths.

You are ready for the next project when you can explain why it works and how you would notice if it does not. Renting a larger GPU can come after that; these learning milestones fit small local experiments.
