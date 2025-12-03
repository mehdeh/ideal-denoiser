# Ideal Denoiser - Implementation of EDM Equation 57

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2206.00364)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository provides a clean implementation of the **Ideal Denoiser** (Equation 57) from the paper:

> **Elucidating the Design Space of Diffusion-Based Generative Models**  
> Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
> NeurIPS 2022

## 📖 Overview

The **ideal denoiser** represents the theoretical optimal solution to the image denoising problem under additive Gaussian noise. It computes the exact **posterior mean** $\mathbb{E}[x' \mid x]$, where $x'$ is the clean image and $x$ is the noisy observation. This closed-form solution, given as Equation 57 in the EDM paper, serves as an upper bound for evaluating practical denoising algorithms.

### Key Contributions

This repository provides:

1. **Theoretical Foundation**: Implementation of Equation 57 (closed-form optimal denoiser) from the EDM paper
2. **Numerical Methods**: Our own implementation of stable computation using log-sum-exp techniques for extreme noise levels
3. **Empirical Analysis**: Visualization tools for comparing denoising performance across noise levels
4. **Mathematical Documentation**: Rigorous derivations connecting Bayesian inference, score matching, and denoising

### Equation 57: The Closed-Form Solution

Given a dataset $\{x_1, \ldots, x_N\}$ representing the empirical data distribution, the ideal denoiser is expressed as:

$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}
$$

This formula computes a **weighted kernel average** over the training distribution, where each weight $w_i \propto \mathcal{N}(x; x_i, \sigma^2 I)$ represents the likelihood that training image $x_i$ generated the noisy observation $x$ under Gaussian noise model $\mathcal{N}(0, \sigma^2 I)$.


## 📁 Project Structure

```
ideal-denoising/
├── ideal_denoiser.py           # Core implementation (Equation 57)
├── run_ideal_denoiser.py       # CLI tool to run denoiser with custom parameters
├── generate_edm_figure1.py     # Generate EDM Figure 1 visualization
├── MATHEMATICAL_BACKGROUND.md  # Mathematical theory and derivations
├── README.md                   # This file
├── requirements.txt            # Dependencies
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── noise_utils.py          # Noise generation
│   ├── image_utils.py          # Data loading and processing
│   └── visualization.py        # Plotting and visualization
│
├── docs/                       # Documentation images
│   ├── figure1_combined_train.png
│   └── figure1_combined_test.png
│
├── data/                       # Dataset storage (auto-downloaded)
└── results/                    # Output directory
    ├── edm_figure1/            # EDM Figure 1 results
    └── denoiser_runs/          # CLI denoiser run results
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the repository
cd ideal-denoising

# Install dependencies
pip install -r requirements.txt
```

### Reproduce EDM Paper Visualization

Generate empirical demonstrations of the ideal denoiser across the noise spectrum:

```bash
python generate_edm_figure1.py
```

This experiment:
1. Loads CIFAR-10 as the empirical approximation of $p_{\text{data}}$
2. Synthesizes noisy observations at noise levels $\sigma \in [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]$
3. Applies Equation 57 to compute optimal posterior means
4. Generates comparative visualizations

**Output Files:**
- `figure1_combined_train.png`: In-distribution denoising results
- `figure1_combined_test.png`: Out-of-sample denoising results

**Visualization Structure:**
- **Top row:** Noisy observations $x = x' + n$ at various $\sigma$
- **Bottom row:** Posterior means $D(x; \sigma) = \mathbb{E}[x' \mid x]$
- **Columns:** Progression from low to high noise regimes

This demonstrates the denoiser's behavior across different signal-to-noise ratios.

**Example Results:**

<div align="center">

**Training Set (In-distribution denoising):**

<img src="docs/figure1_combined_train.png" width="50%">

**Test Set (Out-of-sample denoising):**

<img src="docs/figure1_combined_test.png" width="50%">

</div>

**Expected runtime:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

### Run Ideal Denoiser with Custom Parameters

Use the CLI tool to run the ideal denoiser with configurable parameters:

```bash
# Basic usage with default parameters
python run_ideal_denoiser.py

# Custom number of images and noise levels
python run_ideal_denoiser.py --num-images 5 --sigma-list 0 1 2 5 10

# Use GPU and larger training set
python run_ideal_denoiser.py --device cuda --train-size 5000
```

**CLI Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-root` | `./data` | Root directory for CIFAR-10 data |
| `--save-dir` | `./results/denoiser_runs` | Directory to save output images |
| `--num-images` | `3` | Number of images to denoise from each dataset |
| `--train-size` | `1000` | Number of training images for denoiser reference |
| `--sigma-list` | `0 0.2 0.5 1 2 3 5 7 10 20 50` | List of noise levels to test |
| `--device` | auto-detect | Device to use (`cpu` or `cuda`) |
| `--seed` | `None` | Random seed for reproducibility (if not set, selection is non-deterministic) |

**Output Files:**

Output files are automatically named with timestamp and configuration:
- `{timestamp}_n{num_images}_s{sigma_min}-{sigma_max}_train{train_size}_train.png`: Training set results
- `{timestamp}_n{num_images}_s{sigma_min}-{sigma_max}_train{train_size}_test.png`: Test set results

Example: `20231215_143022_n3_s0-50_train1000_train.png`

**Visualization Structure:**
- **Row 1:** Noisy images at different noise levels
- **Row 2:** Denoised results using ideal denoiser

**Expected runtime:** ~1-15 minutes depending on `--train-size` and `--sigma-list` length


## 📐 Mathematical Foundation

### Problem Formulation

Given a clean image $x' \sim p_{\text{data}}$, we observe a noisy version:

$$
x = x' + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)
$$

where:
- $x'$: clean image from the data distribution
- $x$: noisy observation  
- $n$: Gaussian noise with standard deviation $\sigma$

### The Ideal Denoiser

The ideal denoiser computes the **posterior mean** - the expected value of the clean image given the noisy observation:

$$
D(x; \sigma) = \mathbb{E}[x' \mid x] = \int x' \cdot p(x' \mid x) \, dx'
$$

This represents the **theoretical optimal denoiser** under the $L^2$ loss, providing an upper bound on denoising performance.

### Closed-Form Solution (Equation 57)

For a finite dataset $\{x_1, x_2, \ldots, x_N\}$ with empirical distribution $p_{\text{data}}(x') = \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i)$, the ideal denoiser has the closed-form solution:

$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}
$$

This is a **weighted average** of all training images, where the weights are:

$$
w_i = \frac{\exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right)}
$$

### Interpretation

The formula can be understood as a **kernel density estimator** where:
- **Similar images receive higher weights**: If $x_i$ is close to the noisy input $x$, then $\|x - x_i\|^2$ is small and $w_i$ is large
- **Dissimilar images receive lower weights**: If $x_i$ is far from $x$, then $w_i$ approaches zero
- **Noise level controls smoothness**: Larger $\sigma$ produces more uniform weights (smoother averaging); smaller $\sigma$ produces peaked weights (nearest neighbor-like behavior)

### Numerical Stability: Log-Sum-Exp Trick

**Note**: *This numerical stabilization technique is our own implementation and is not described in the original EDM paper.*

Direct computation can cause numerical overflow/underflow when $\sigma$ is small. We employ the **log-sum-exp trick** for stability:

Define the log-probabilities:
$$
\ell_i = -\frac{\|x - x_i\|^2}{2\sigma^2}
$$

Compute the maximum:
$$
\delta = \max_{j=1,\ldots,N} \ell_j
$$

Then the numerically stable formula becomes:
$$
D(x; \sigma) = \frac{\sum_{i=1}^{N} x_i \cdot \exp(\ell_i - \delta)}{\sum_{i=1}^{N} \exp(\ell_i - \delta)}
$$

By subtracting $\delta$, all exponentials are bounded in $(0, 1]$, preventing overflow while maintaining mathematical equivalence.

### Connection to Score Matching

The ideal denoiser is fundamentally connected to the **score function** of the noisy distribution:

$$
\nabla_x \log p(x; \sigma) = -\frac{1}{\sigma^2}(x - D(x; \sigma))
$$

This relationship, known as **Tweedie's formula**, can be rewritten as:

$$
D(x; \sigma) = x + \sigma^2 \nabla_x \log p(x; \sigma)
$$

This connects denoising to score-based generative modeling, where the denoiser directly computes the score-corrected estimate.

**For detailed mathematical derivations**, including:
- Bayesian posterior mean derivation
- Denoising score matching optimization approach  
- Special cases and limiting behavior
- Extensions and variations

Please refer to the comprehensive mathematical documentation:  
**[MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md)**


## 🔧 Experimental Parameters

The experiments can be configured to explore different regimes of the denoising problem. Key parameters include:

- **Noise levels ($\sigma$)**: Range of standard deviations $[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]$
  - Low regime: $\sigma \in [0.2, 1]$ → nearest-neighbor behavior
  - Intermediate: $\sigma \in [2, 10]$ → kernel averaging
  - High regime: $\sigma \geq 20$ → approaches dataset mean

- **Dataset size ($N$)**: Number of training samples for empirical distribution (controlled by `--train-size`)
  - Affects approximation quality of $p_{\text{data}}$
  - Computational complexity scales linearly with $N$
  - Recommended: 1000-5000 for CIFAR-10

- **Number of test images**: Number of images to denoise (controlled by `--num-images`)
  - More images provide better visual assessment
  - Trade-off between visualization clarity and processing time

## 🎓 Theoretical Background

### The Ideal Denoiser as an Optimal Estimator

The ideal denoiser represents the **Bayes-optimal estimator** under the $L^2$ loss for the image denoising problem. It provides a **theoretical performance upper bound** against which practical denoising algorithms can be evaluated. The optimality is derived from two equivalent perspectives:

1. **Bayesian Inference**: The posterior mean $\mathbb{E}[x' \mid x]$ minimizes the expected squared error
2. **Denoising Score Matching**: The solution that minimizes $\mathbb{E}_{x' \sim p_{\text{data}}} \mathbb{E}_{n \sim \mathcal{N}(0,\sigma^2)} \|D(x'+n) - x'\|^2$

### Assumptions and Limitations

The ideal denoiser requires:
- **Complete knowledge of $p_{\text{data}}$**: Access to the entire training distribution (empirically: all training samples)
- **Known noise model**: Precise knowledge of noise level $\sigma$
- **Computational resources**: $O(N \times d)$ complexity per query, where $N$ is dataset size

### Neural Denoisers as Function Approximators

Practical neural network-based denoisers **approximate** the ideal denoiser with critical advantages:
- **Constant complexity**: $O(d)$ inference time, independent of $N$
- **Compact parametrization**: Store only network weights $\theta$, not entire dataset
- **Generalization capability**: Learned denoisers can handle out-of-distribution images

The ideal denoiser thus serves as a theoretical benchmark, while neural networks provide scalable approximations.

### Asymptotic Behavior

The denoiser exhibits well-defined limiting behavior:

1. **Low noise regime ($\sigma \to 0$)**: 
   $$D(x; \sigma) \to x_{\text{NN}}, \quad \text{where } x_{\text{NN}} = \arg\min_{x_i} \|x - x_i\|$$
   Reduces to nearest neighbor selection

2. **High noise regime ($\sigma \to \infty$)**: 
   $$D(x; \sigma) \to \bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$
   Converges to the dataset mean (all weights become uniform)

3. **Interpolation regime**: For intermediate $\sigma$, performs smooth kernel-weighted averaging

## 📚 Mathematical Derivations and Proofs

For comprehensive mathematical derivations, proofs, and theoretical analysis, please refer to:

**[MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md)**

This document provides:

- **Two rigorous derivations of Equation 57**:
  - Method 1: Bayesian posterior mean approach using Bayes' rule
  - Method 2: Denoising score matching optimization via convex analysis
  
- **Theoretical connections**:
  - Relationship to score-based generative models
  - Tweedie's formula and its implications
  - Connection to Nadaraya-Watson kernel regression
  
- **Analytical properties**:
  - Special cases and asymptotic behavior
  - Proof of optimality under $L^2$ loss
  - Numerical stability analysis (log-sum-exp trick)
  
- **Computational considerations**:
  - Time and space complexity analysis
  - Comparison with neural network denoisers
  - Practical implementation guidelines

## 🤝 Acknowledgments

### Concept and Formula

The ideal denoiser concept and formula (Equation 57) are from:
- **EDM Paper**: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
- **Paper**: https://arxiv.org/abs/2206.00364
- **Equation 57**: Appendix B.3 of the paper

### Implementation

This implementation is our own work. **The original EDM repository does not include code for the ideal denoiser (Equation 57)**. This repository fills that gap by providing a clean, well-documented implementation.

## 📖 Citation

If you use this code, please cite the original EDM paper:

```bibtex
@inproceedings{Karras2022edm,
  author    = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Proc. NeurIPS},
  year      = {2022}
}
```

## 📄 License

This project is provided freely for educational and research purposes.

**Attribution:**
- The ideal denoiser concept and formula (Equation 57) are from the EDM paper by Karras et al. (NeurIPS 2022)
- This implementation is our own work and is provided without license restrictions

## 🐛 Issues & Contributions

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

---

**Note:** This repository focuses exclusively on the ideal denoiser (Equation 57). For the full EDM implementation including neural network-based denoisers, please refer to the [official EDM repository](https://github.com/NVlabs/edm).
