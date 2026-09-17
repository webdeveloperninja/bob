# 02 — The math you need, with intuition

[Learning home](README.md) · Previous: [Text to model](01-text-to-model.md) · Next: [Attention](03-attention-and-your-model.md)

Calculus 1 gives you a useful starting point: functions, derivatives, and the chain rule. The next essentials are linear algebra, probability, and gradients for functions of many variables. You can learn these alongside the code.

The equations below are descriptions of operations, not exercises in arithmetic speed. Read each as “what information goes in, what comes out, and what can training change?”

## 1. Vectors and matrices: representations and transformations

A vector is an ordered list of numbers. Bob represents each byte by a 128-number embedding vector. This is a learned representation, not 128 hand-designed properties or a unique label for every possible meaning.

A matrix transforms vectors into other vectors. A linear layer is:

```text
output = input @ Wᵀ + b
```

`@` means matrix multiplication; `Wᵀ` transposes the weight matrix. PyTorch stores a linear layer's weights as `[output_features, input_features]`. The bias `b` shifts the output. Training adjusts `W` and `b` so the transformation helps prediction.

The dot product `q · k = Σᵢ qᵢ kᵢ` combines two vectors into one score. Attention learns representations where this score helps decide which positions should contribute information. It is not automatically a semantic similarity metric; lengths and the learned transformations both influence it.

A tensor generalizes these arrays to more axes. `[8, 128, 128]` means 8 examples, 128 positions per example, 128 features per position. The last two dimensions happen to match in the local config; their meanings are different.

## 2. Why neural networks need nonlinear functions

Two affine transformations composed without a nonlinear operation still give one affine transformation. Stacking many such layers would not create the desired extra expressive power.

Bob's feed-forward network uses:

```text
MLP(x) = Linear₂(GELU(Linear₁(x)))
```

It expands 128 features to 512, applies a smooth nonlinear activation, then returns to 128. The same MLP is applied separately at each position. Attention exchanges information across positions; the MLP transforms the information held at each position.

You do not need to derive GELU's formula to use the model. Learn why a nonlinearity matters before comparing activation functions.

## 3. Scores become probabilities

The model outputs 256 **logits**: unrestricted scores, one per possible next byte. Softmax turns them into a probability distribution:

```text
pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

`zᵢ` is a logit. Every probability is nonnegative and they sum to 1. Increasing one score relative to others increases its probability. Adding the same constant to all scores does not change the probabilities; this fact helps numerical stability.

For an invented three-token vocabulary, logits `[2, 1, 0]` become approximately `[0.665, 0.245, 0.090]`. These are illustration values, not probabilities from your checkpoint.

Generation applies `softmax(logits / temperature)` after keeping top-k candidates. A smaller positive temperature sharpens preferences. This can reduce randomness but cannot supply missing knowledge. Top-k truncation changes and renormalizes the distribution too.

## 4. Loss measures surprise at the true answer

For one position with true next byte `y`:

```text
loss = -ln p(y | context)
```

If the model assigns the correct byte probability 0.5, the loss is about 0.693. If it assigns 0.01, the loss is about 4.605. Confidently assigning very little probability to the truth is costly.

For N prediction positions, Bob averages the losses:

```text
L(θ) = -(1/N) Σₜ ln pθ(actual next byte at t | visible context at t)
```

`θ` means all model parameters together. The sum sign means add the per-position terms. Minimizing this is equivalent to maximizing the likelihood of the observed targets under the chosen contexts. `F.cross_entropy` takes raw logits and handles the log-softmax calculation internally; do not apply softmax first in this training loop.

A uniform 256-byte predictor has loss `ln(256) ≈ 5.545`. Your validation loss near 2.61 corresponds to **byte perplexity** `exp(2.61) ≈ 13.6`. One intuition is effective prediction uncertainty; it does not mean exactly 14 candidates occur at every position, or that accuracy is 1/14. Cross-entropy in bits per byte is `L / ln(2)`—about 3.77 at loss 2.61.

Only compare these values on compatible data and tokenization. Byte perplexity cannot be directly compared to a subword model's perplexity as if they were the same measurement.

## 5. From a derivative to a gradient

In Calc 1, a derivative tells you how a scalar output changes when one input changes. Here loss depends on hundreds of thousands of parameters.

A partial derivative asks: “If I slightly change this parameter while holding the others fixed, how does loss change?” The gradient collects those partial derivatives:

```text
∇θ L = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]
```

A basic gradient-descent update would be:

```text
θnew = θold - learning_rate × ∇θ L
```

The negative gradient points toward locally decreasing loss in parameter space. “Locally” matters: a step that is too large may overshoot, and a minibatch is a noisy estimate of the larger-data objective.

A one-parameter example: `L(w) = (w - 3)²`. At `w = 0`, the derivative is `-6`. With learning rate 0.1, the next value is `0 - 0.1(-6) = 0.6`, closer to the minimum at 3. A neural network uses the same sensitivity idea in many dimensions, with a much more complicated loss surface.

## 6. Backpropagation is the chain rule organized efficiently

If `w → a → prediction → loss`, then:

```text
∂loss/∂w = (∂loss/∂prediction) × (∂prediction/∂a) × (∂a/∂w)
```

Multiple paths contribute sums as well. PyTorch records differentiable operations in a computational graph during the forward pass. `loss.backward()` traverses it backward and accumulates gradients. It does not perturb every weight and rerun the whole model to estimate each derivative.

A tiny runnable example—run from the `bob` folder with `.venv` activated:

```bash
python - <<'PY'
import torch
w = torch.tensor(0.0, requires_grad=True)
loss = (w - 3) ** 2
loss.backward()
print("loss:", loss.item())  # 9
print("gradient:", w.grad.item())  # -6
with torch.no_grad():
    w -= 0.1 * w.grad
print("updated w:", w.item())  # approximately 0.6
PY
```

`optimizer.zero_grad()` clears old accumulated gradients before the next batch. `torch.no_grad()` disables graph recording for calculations that will not be differentiated. It is different from `model.eval()`, which changes behaviors of certain layers, such as dropout, when present.

## 7. What Bob adds to basic gradient descent

**AdamW** uses running gradient statistics to adapt updates and applies decoupled weight decay. It is not exactly the simple update above. Optimizer state belongs in a resume checkpoint because it affects future updates.

**Gradient clipping** limits the total gradient norm when it is too large. Bob uses a maximum norm of 1.0. This does not constrain every weight to the interval [-1, 1].

**Layer normalization** rescales each position's feature vector using its mean and variance, then applies learned scale and offset. It helps keep computations manageable but is not itself a probability normalization like softmax.

## Check yourself

1. Why are logits allowed to be negative?
2. What is the difference between a weight and its gradient?
3. Why not increase the learning rate whenever progress is slow?

**Answer hints:** logits are scores, converted to probabilities later. A weight sets the function; its gradient measures local sensitivity of loss. Large updates can destabilize training rather than improve it.

For targeted study rather than a full math degree, use the [math bridge in the study guide](06-study-guide.md#the-math-bridge-after-calculus-1).
