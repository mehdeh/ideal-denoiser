# Ideal Denoiser - Implementation of EDM Equation 57

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2206.00364)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository provides a clean implementation of the **Ideal Denoiser** (Equation 57) from the paper:

> **Elucidating the Design Space of Diffusion-Based Generative Models**  
> Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
> NeurIPS 2022

## 📖 Overview

The ideal denoiser is the **theoretical optimal denoiser** that computes the exact expected value of clean images given noisy observations. This repository provides:

1. **Clean Implementation**: Direct implementation of Equation 57 from the EDM paper
2. **Numerical Stability**: Uses log-sum-exp trick for stable computation
3. **Visualization Tools**: Generate comparison figures across different noise levels
4. **Mathematical Background**: Comprehensive documentation of the theory

### Ideal Denoiser Formula

The ideal denoiser computes:

```
D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
```

where:
- `x`: noisy observation
- `σ`: noise level (standard deviation)
- `xᵢ`: training images from the reference distribution

This is a **weighted average** of all training images, where weights are proportional to the likelihood of each training image generating the observed noisy image under Gaussian noise.


## 📁 Project Structure

```
ideal-denoising/
├── ideal_denoiser.py           # Core implementation (Equation 57)
├── compare_denoisers.py        # Comparison across noise levels
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
├── data/                       # Dataset storage (auto-downloaded)
└── results/                    # Output directory
    ├── edm_figure1/            # EDM Figure 1 results
    └── denoiser_comparison/    # Comparison results
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the repository
cd ideal-denoising

# Install dependencies
pip install -r requirements.txt
```

### Generate EDM Figure 1

Reproduce the ideal denoiser visualization from the EDM paper:

```bash
python generate_edm_figure1.py
```

This will:
1. Download CIFAR-10 dataset (if needed)
2. Generate noisy images with σ = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
3. Denoise using the ideal denoiser
4. Save results to `./results/edm_figure1/` directory

**Output:**
- `figure1_combined_train.png`: Combined visualization for training images
- `figure1_combined_test.png`: Combined visualization for test images

Each figure shows:
- **Top row:** Noisy images with different σ values
- **Bottom row:** Ideal denoiser results
- **Columns:** Different noise levels

**Expected runtime:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

### Compare Across Noise Levels

Compare ideal denoiser performance across multiple noise levels with different delta computation methods:

```bash
python compare_denoisers.py
```

This will:
1. Load selected images from CIFAR-10
2. Add noise at various sigma levels
3. Apply multiple denoiser variants (Max, Median, Percentile-95, Adaptive)
4. Create comparison visualizations for each method
5. Save results to `./results/denoiser_comparison/` directory

**Output:**
- `comparison_train_{method}.png`: Training images with each method
- `comparison_test_{method}.png`: Test images with each method

**Available Methods:**
- **Max (Original)**: Uses maximum value for delta (most stable, default)
- **Median**: Uses median-based blending for smoother weighting
- **Percentile-95**: Uses 95th percentile blending
- **Adaptive**: Adaptive interpolation between max and high percentile
- **Mean+Std**: Uses mean + alpha * std approach with adaptive blending

Each figure shows:
- **Row 1:** Noisy images at different noise levels
- **Row 2:** Denoised results using the specific method

**Expected runtime:** ~15-20 minutes on CPU, ~5-7 minutes on GPU


## 💻 API Usage

### Basic Usage

```python
import torch
from ideal_denoiser import ideal_denoiser
from utils import add_gaussian_noise, load_cifar10_subset

# Load data
train_images = load_cifar10_subset(root="./data", train=True, max_samples=1000)
test_image = load_cifar10_subset(root="./data", train=False, max_samples=1)[0:1]

# Add noise
sigma = 2.0
noisy_image = add_gaussian_noise(test_image, sigma)

# Denoise
denoised_image = ideal_denoiser(noisy_image, sigma, train_images)
```

### Enhanced Denoiser with Multiple Delta Methods

```python
from ideal_denoiser import ideal_denoiser_enhanced, get_available_denoiser_methods

# Get available methods
methods = get_available_denoiser_methods()
for name, config in methods.items():
    print(f"{name}: {config['display_name']}")

# Use specific method
denoised_max = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 
                                        delta_method='max')
denoised_median = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 
                                           delta_method='mean')
denoised_percentile = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 
                                                delta_method='percentile', delta_param=95)
denoised_adaptive = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 
                                             delta_method='adaptive', delta_param=0.5)
denoised_std = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 
                                        delta_method='mean_std', delta_param=1.0)
```

**Delta Methods Explained:**
- **max**: Most numerically stable, uses maximum log-probability
- **mean**: Blends max with median (sigma-adaptive ratio for stability at low sigma)
- **percentile**: Blends max with high percentile (sigma-adaptive blending)
- **adaptive**: Interpolates between max and blended high-percentile based on coefficient
- **mean_std**: Uses mean + alpha * std with sigma-adaptive blending for stability

All methods except 'max' use sigma-adaptive blending to ensure numerical stability at low sigma values (0.2, 0.5, 1.0).


## 📝 Implementation Details

### Mathematical Formula

The ideal denoiser implements Equation 57 from the EDM paper:

```
D(x; σ) = E[x' | x]  where  x = x' + n,  n ~ N(0, σ²I)
```

This computes the **posterior mean** - the expected value of the clean image given the noisy observation.

### Numerical Stability

The implementation uses the **log-sum-exp trick** to prevent numerical overflow:

```python
# Compute log probabilities
log_probs = -||x - xᵢ||² / (2σ²)

# Subtract max for stability (log-sum-exp trick)
delta = max(log_probs)
weights = exp(log_probs - delta)

# Weighted average
D(x; σ) = Σᵢ [weights_i · xᵢ] / Σᵢ [weights_i]
```

This ensures stable computation even for small sigma values or large distances.


## 🔧 Configuration

You can customize the comparison by modifying the configuration in the scripts:

```python
# In compare_denoisers.py or generate_edm_figure1.py:
config = {
    'sigma_values': [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],  # Noise levels
    'ideal_denoiser_subset_size': 1000,  # Number of training images to use
    'train_selection_indices': [2, 3, 4],  # Which images to visualize
    'test_selection_indices': [2, 3, 4]
}
```

## 🎓 Background

### What is the Ideal Denoiser?

The ideal denoiser is a **theoretical upper bound** on denoising performance. It assumes:

1. **Full knowledge of data distribution**: Access to entire training set
2. **Known noise level**: Perfect knowledge of σ
3. **Exact computation**: Ability to compute exact posterior mean

In practice, neural networks trained as denoisers approximate this ideal denoiser but with:
- **Constant computation**: O(d) regardless of dataset size
- **Compact representation**: Store only network weights
- **Better generalization**: Can denoise images outside training set

### Key Insights

- **Weighted Average**: The ideal denoiser is a weighted average of all training images
- **Similarity-based**: Similar images get higher weights
- **Sigma-dependent**: Larger σ makes weights more uniform; smaller σ makes them more peaked

### Special Cases

1. **Zero noise (σ → 0)**: Returns nearest neighbor from training set
2. **Infinite noise (σ → ∞)**: Returns mean of all training images
3. **Exact match**: If noisy input matches a training image, returns that image

## 📚 Mathematical Background

For detailed mathematical derivations and theory, see [`MATHEMATICAL_BACKGROUND.md`](MATHEMATICAL_BACKGROUND.md), which includes:

- Two methods to derive Equation 57 (Bayesian and Optimization approaches)
- Connection to score matching and Tweedie's formula
- Intuitive understanding of the weighted average
- Special cases and limiting behavior
- Numerical stability considerations
- Computational complexity analysis

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
