# Mathematical Background: Ideal Denoiser

This document explains the mathematical theory behind the ideal denoiser implementation.

## 1. Problem Setup

### Noise Model

Given a clean image $x' \sim p_{\text{data}}$, we observe a noisy version:

$$
x = x' + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)
$$

where:
- $x'$: clean image from the data distribution
- $x$: noisy observation
- $n$: Gaussian noise with standard deviation $\sigma$

### Denoising Goal

The ideal denoiser aims to recover the clean image by computing:

$$
D(x; \sigma) = \mathbb{E}[x' \mid x] = \int x' \cdot p(x' \mid x) \, dx'
$$

This is the **posterior mean** - the expected value of the clean image given the noisy observation.

## 2. Two Methods to Derive Equation 57

There are two equivalent approaches to derive the closed-form solution for the ideal denoiser. Both methods arrive at the same formula (Equation 57 from the EDM paper), but provide different insights.

---

### Method 1: Bayesian Posterior Mean Approach

This approach directly computes the posterior mean using Bayes' rule.

#### Step 1: Bayes' Rule

Using Bayes' rule:

$$
p(x' \mid x) = \frac{p(x \mid x') \cdot p(x')}{p(x)}
$$

#### Step 2: Likelihood

The likelihood of observing $x$ given $x'$ follows from the Gaussian noise model:

$$
p(x \mid x') = \mathcal{N}(x; x', \sigma^2 I) = \frac{1}{Z} \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
$$

where $Z$ is the normalization constant.

#### Step 3: Empirical Distribution

For a finite dataset $\{x_1, x_2, \ldots, x_N\}$, we approximate:

$$
p_{\text{data}}(x') \approx \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i)
$$

#### Step 4: Posterior Mean Calculation

Combining the above:

$$
D(x; \sigma) = \int x' \cdot p(x' \mid x) \, dx'
$$

$$
= \frac{\int x' \cdot p(x \mid x') \cdot p(x') \, dx'}{\int p(x \mid x') \cdot p(x') \, dx'}
$$

Substituting the empirical distribution:

$$
= \frac{\int x' \cdot \mathcal{N}(x; x', \sigma^2 I) \cdot \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i) \, dx'}{\int \mathcal{N}(x; x', \sigma^2 I) \cdot \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i) \, dx'}
$$

$$
= \frac{\sum_{i=1}^{N} x_i \cdot \mathcal{N}(x; x_i, \sigma^2 I)}{\sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)}
$$

Expanding the Gaussian density (up to a constant):

$$
\boxed{D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}}
$$

This is **Equation 57** from the EDM paper.

---

### Method 2: Denoising Score Matching Optimization

This approach derives the same formula by minimizing the denoising score matching loss. This is the derivation presented in **Appendix B.3** of the EDM paper.

#### Step 1: Noisy Data Distribution

For a finite training set $\{x_1, \dots, x_N\}$, the empirical data distribution is:

$$
p_{\text{data}}(x') = \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i)
$$

The noisy distribution at noise level $\sigma$ is obtained by convolving with Gaussian noise:

$$
p(x; \sigma) = p_{\text{data}} \ast \mathcal{N}(0, \sigma^2 I) = \int_{\mathbb{R}^d} p_{\text{data}}(x_0) \cdot \mathcal{N}(x; x_0, \sigma^2 I) \, dx_0
$$

Substituting the empirical distribution:

$$
p(x; \sigma) = \int_{\mathbb{R}^d} \left[\frac{1}{N} \sum_{i=1}^{N} \delta(x_0 - x_i)\right] \mathcal{N}(x; x_0, \sigma^2 I) \, dx_0
$$

$$
= \frac{1}{N} \sum_{i=1}^{N} \int_{\mathbb{R}^d} \mathcal{N}(x; x_0, \sigma^2 I) \cdot \delta(x_0 - x_i) \, dx_0
$$

$$
= \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)
$$

#### Step 2: Denoising Score Matching Loss

The denoising score matching loss is defined as:

$$
\mathcal{L}(D; \sigma) = \mathbb{E}_{x' \sim p_{\text{data}}} \, \mathbb{E}_{n \sim \mathcal{N}(0, \sigma^2 I)} \, \|D(x' + n; \sigma) - x'\|^2
$$

By expanding the expectations, we rewrite this as an integral over noisy samples $x$:

$$
\mathcal{L}(D; \sigma) = \mathbb{E}_{x' \sim p_{\text{data}}} \, \mathbb{E}_{x \sim \mathcal{N}(x', \sigma^2 I)} \, \|D(x; \sigma) - x'\|^2
$$

$$
= \mathbb{E}_{x' \sim p_{\text{data}}} \int_{\mathbb{R}^d} \mathcal{N}(x; x', \sigma^2 I) \, \|D(x; \sigma) - x'\|^2 \, dx
$$

Substituting the empirical distribution:

$$
= \frac{1}{N} \sum_{i=1}^{N} \int_{\mathbb{R}^d} \mathcal{N}(x; x_i, \sigma^2 I) \, \|D(x; \sigma) - x_i\|^2 \, dx
$$

$$
= \int_{\mathbb{R}^d} \underbrace{\frac{1}{N} \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \, \|D(x; \sigma) - x_i\|^2}_{=: \, \mathcal{L}(D; x, \sigma)} \, dx
$$

#### Step 3: Pointwise Optimization

The above equation shows we can minimize $\mathcal{L}(D; \sigma)$ by minimizing $\mathcal{L}(D; x, \sigma)$ independently for each $x$:

$$
D(x; \sigma) = \arg\min_{D(x; \sigma)} \mathcal{L}(D; x, \sigma)
$$

This is a **convex optimization problem**. The optimal solution is found by setting the gradient with respect to $D(x; \sigma)$ to zero:

$$
\mathbf{0} = \nabla_{D(x; \sigma)} \left[\mathcal{L}(D; x, \sigma)\right]
$$

$$
\mathbf{0} = \nabla_{D(x; \sigma)} \left[\frac{1}{N} \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \, \|D(x; \sigma) - x_i\|^2\right]
$$

$$
\mathbf{0} = \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \, \nabla_{D(x; \sigma)} \left[\|D(x; \sigma) - x_i\|^2\right]
$$

Using the fact that $\nabla_D \|D - x_i\|^2 = 2(D - x_i)$:

$$
\mathbf{0} = \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \cdot [2D(x; \sigma) - 2x_i]
$$

$$
\mathbf{0} = \left[\sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)\right] D(x; \sigma) - \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \cdot x_i
$$

Solving for $D(x; \sigma)$:

$$
\boxed{D(x; \sigma) = \frac{\sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I) \cdot x_i}{\sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)}}
$$

This is the **closed-form solution** for the ideal denoiser (Equation 57).

#### Step 4: Equivalence to Method 1

Note that $\mathcal{N}(x; x_i, \sigma^2 I) \propto \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)$, so:

$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}
$$

This is **identical** to the formula derived in Method 1.

---

### Key Insights from Both Methods

**Method 1 (Bayesian)**: 
- Shows that the ideal denoiser is the **posterior mean** $\mathbb{E}[x' \mid x]$
- Provides a probabilistic interpretation
- Natural from a Bayesian inference perspective

**Method 2 (Optimization)**:
- Shows that the ideal denoiser **minimizes the mean squared error**
- Provides an optimization-based interpretation
- Natural from a loss minimization perspective
- Demonstrates the connection to **score matching**

Both derivations are rigorous and arrive at the same closed-form solution, confirming that the formula is the unique optimal denoiser under the $L^2$ loss.

## 3. Intuitive Understanding

### Weighted Average

The ideal denoiser computes a **weighted average** of all training images:

$$
D(x; \sigma) = \sum_{i=1}^{N} w_i \cdot x_i
$$

where the weights are:

$$
w_i = \frac{\exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right)}
$$

### Interpretation

- **Similar images get higher weight**: If $x_i$ is close to the noisy input $x$, then $\|x - x_i\|^2$ is small, and $w_i$ is large.
- **Dissimilar images get lower weight**: If $x_i$ is far from $x$, then $w_i$ is small.
- **Sigma controls similarity**: Larger $\sigma$ makes the weights more uniform; smaller $\sigma$ makes the weights more peaked.

## 4. Special Cases

### Case 1: Zero Noise ($\sigma \to 0$)

When $\sigma \to 0$:

$$
D(x; \sigma) \to x_{\text{nearest}}
$$

The denoiser returns the **nearest neighbor** from the training set.

**Proof**: As $\sigma \to 0$, the weight $w_i$ for the closest training image dominates all others.

### Case 2: Infinite Noise ($\sigma \to \infty$)

When $\sigma \to \infty$:

$$
D(x; \sigma) \to \frac{1}{N} \sum_{i=1}^{N} x_i = \bar{x}
$$

The denoiser returns the **mean** of all training images.

**Proof**: As $\sigma \to \infty$, all weights become equal: $w_i \to 1/N$.

### Case 3: Exact Match ($x = x_k$ for some $k$)

If the noisy input exactly matches a training image:

$$
D(x; \sigma) \approx x_k
$$

for small $\sigma$.

## 5. Connection to Score Matching

### Score Function

The **score function** is the gradient of the log-density:

$$
\nabla_x \log p(x; \sigma) = -\frac{1}{\sigma^2}(x - D(x; \sigma))
$$

This relates the ideal denoiser to the **score** of the noisy distribution.

### Tweedie's Formula

The ideal denoiser can be written as:

$$
D(x; \sigma) = x + \sigma^2 \nabla_x \log p(x; \sigma)
$$

This is known as **Tweedie's formula** in statistics.

### Derivation of Score-Denoiser Relationship

Taking the gradient of $\log p(x; \sigma)$:

$$
\nabla_x \log p(x; \sigma) = \nabla_x \log \left[\frac{1}{N} \sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)\right]
$$

$$
= \frac{1}{p(x; \sigma)} \nabla_x p(x; \sigma)
$$

$$
= \frac{1}{p(x; \sigma)} \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_x \mathcal{N}(x; x_i, \sigma^2 I)
$$

Using $\nabla_x \mathcal{N}(x; x_i, \sigma^2 I) = -\frac{1}{\sigma^2}(x - x_i) \mathcal{N}(x; x_i, \sigma^2 I)$:

$$
= -\frac{1}{\sigma^2 p(x; \sigma)} \cdot \frac{1}{N} \sum_{i=1}^{N} (x - x_i) \mathcal{N}(x; x_i, \sigma^2 I)
$$

$$
= -\frac{1}{\sigma^2} \left[x - \frac{\sum_{i=1}^{N} x_i \mathcal{N}(x; x_i, \sigma^2 I)}{\sum_{i=1}^{N} \mathcal{N}(x; x_i, \sigma^2 I)}\right]
$$

$$
= -\frac{1}{\sigma^2}(x - D(x; \sigma))
$$

This proves the fundamental relationship between the score function and the ideal denoiser.

## 6. Computational Considerations

### Time Complexity

Computing $D(x; \sigma)$ requires:

1. **Distance computation**: $O(N \times d)$ where $d = C \times H \times W$
2. **Weight computation**: $O(N)$
3. **Weighted sum**: $O(N \times d)$

**Total**: $O(N \times d)$ per query

For CIFAR-10:
- $N = 50,000$ training images
- $d = 3 \times 32 \times 32 = 3,072$ dimensions
- **Total**: ~150 million operations per denoising

### Memory Complexity

Storing the training set requires:

$$
\text{Memory} = N \times C \times H \times W \times \text{bytes per pixel}
$$

For CIFAR-10 with float32:
- $50,000 \times 3 \times 32 \times 32 \times 4 = 600 \text{ MB}$

### Numerical Stability

The direct computation of weights:

$$
w_i = \frac{\exp(-\|x - x_i\|^2 / (2\sigma^2))}{\sum_j \exp(-\|x - x_j\|^2 / (2\sigma^2))}
$$

can cause **numerical overflow/underflow** when $\sigma$ is small or distances are large.

#### Log-Sum-Exp Trick

To improve stability, we use the **log-sum-exp trick** which prevents numerical overflow by subtracting the maximum value before computing exponentials.

**Original Formula (Numerically Unstable):**

$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}
$$

**Problem**: When $\sigma$ is small or $\|x - x_i\|^2$ is large, the exponential terms can underflow to zero or the computation becomes numerically unstable.

**Solution**: Define the log-probability for each training sample:

$$
\ell_i = -\frac{\|x - x_i\|^2}{2\sigma^2}
$$

Compute the maximum log-probability:

$$
\delta = \max_{j=1,\ldots,N} \ell_j = \max_{j=1,\ldots,N} \left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right)
$$

**Numerically Stable Formula:**

$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp(\ell_i - \delta)}{\sum_{i=1}^{N} \exp(\ell_i - \delta)}
$$

Expanding:

$$
\boxed{D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2} - \delta\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2} - \delta\right)}}
$$

**Why This Works:**

1. **Subtract maximum**: By subtracting $\delta$ from all log-probabilities, the largest value becomes 0
2. **Bounded exponentials**: All exponentials $\exp(\ell_i - \delta)$ are now in the range $(0, 1]$
3. **Numerically stable**: Prevents both overflow and underflow
4. **Equivalent result**: Since $\delta$ appears in both numerator and denominator, it cancels out:

$$
D(x; \sigma) = \frac{\sum_i x_i \cdot \exp(\ell_i - \delta)}{\sum_i \exp(\ell_i - \delta)} = \frac{\exp(-\delta) \sum_i x_i \cdot \exp(\ell_i)}{\exp(-\delta) \sum_i \exp(\ell_i)} = \frac{\sum_i x_i \cdot \exp(\ell_i)}{\sum_i \exp(\ell_i)}
$$

**Implementation Note**: This is the technique used in the code at lines 86-87 of `ideal_denoiser/core.py`, where `delta` is computed as the maximum of `sigma_norm2` values.

## 7. Comparison with Neural Denoisers

| Aspect | Ideal Denoiser | Neural Denoiser |
|--------|----------------|-----------------|
| **Knowledge** | Full training set | Learned parameters |
| **Computation** | $O(N \times d)$ | $O(d)$ |
| **Memory** | Stores entire dataset | Stores network weights |
| **Performance** | Optimal (theoretical upper bound) | Approximates optimal |
| **Scalability** | Poor (increases with dataset size) | Good (fixed size) |

### Why Use Neural Networks?

Neural networks **approximate** the ideal denoiser but with:
- **Constant computation**: $O(d)$ regardless of dataset size
- **Compact representation**: Store only network weights
- **Generalization**: Can denoise images outside the training set

## 8. Practical Implementation

### PyTorch Implementation

```python
def ideal_denoiser(x_noisy, sigma, x_train):
 # Compute distances
 norm2 = ((x_train[:, None] - x_noisy[None, :]) ** 2).sum(dim=(2,3,4))
 
 # Compute log weights (with numerical stability)
 log_weights = -norm2 / (2 * sigma ** 2)
 delta = log_weights.max(dim=0, keepdim=True)[0]
 weights = (log_weights - delta).exp()
 
 # Weighted average
 numerator = (weights[:, :, None, None, None] * x_train[:, None]).sum(dim=0)
 denominator = weights.sum(dim=0)
 
 return numerator / denominator[:, None, None, None]
```

### Batch Processing

For efficiency, process multiple noisy images at once:

```python
# x_noisy: (B, C, H, W) - batch of B noisy images
# x_train: (N, C, H, W) - N training images
denoised = ideal_denoiser(x_noisy, sigma, x_train) # (B, C, H, W)
```

## 9. Extensions and Variations

### 1. Kernel Denoising

Replace Gaussian kernel with other kernels:

$$
D(x; \sigma) = \frac{\sum_i K(x, x_i) \cdot x_i}{\sum_i K(x, x_i)}
$$

where $K(x, x_i)$ is any positive kernel function.

### 2. Local Denoising

Use only $k$ nearest neighbors:

$$
D(x; \sigma) = \frac{\sum_{i \in \mathcal{N}_k(x)} \exp(-\|x - x_i\|^2 / (2\sigma^2)) \cdot x_i}{\sum_{i \in \mathcal{N}_k(x)} \exp(-\|x - x_i\|^2 / (2\sigma^2))}
$$

where $\mathcal{N}_k(x)$ are the $k$ nearest neighbors.

### 3. Anisotropic Denoising

Use different noise levels for different dimensions or channels.

## 10. References

1. **EDM Paper**: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
 - Equation 57 in Appendix B.3
 
2. **Score Matching**: Hyvärinen, "Estimation of Non-Normalized Statistical Models by Score Matching", JMLR 2005

3. **Tweedie's Formula**: Efron, "Tweedie's Formula and Selection Bias", JASA 2011

4. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

5. **Non-parametric Regression**: Nadaraya-Watson kernel regression

---

**Note**: This mathematical background provides the theoretical foundation for understanding the ideal denoiser implementation in this repository.

