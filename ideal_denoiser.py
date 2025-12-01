"""
Ideal Denoiser Implementation (Equation 57 from EDM Paper).

This module implements the theoretical ideal denoiser from the paper:
"Elucidating the Design Space of Diffusion-Based Generative Models"
by Karras et al., NeurIPS 2022.

The ideal denoiser computes the exact expected value of clean images given
noisy observations, using the closed-form solution from Appendix B.3, Eq. 57.

Reference:
    Paper: https://arxiv.org/abs/2206.00364
    Formula: D(x; σ) = E[x' | x], where x = x' + n with n ~ N(0, σ²I)
    
Note:
    While the ideal denoiser concept and formula (Equation 57) are from the EDM paper,
    this implementation is our own work. The original EDM repository does not include
    code for the ideal denoiser.
"""

import torch


def ideal_denoiser(x_noisy, sigma, x_all):
    """
    Ideal denoiser using closed-form solution from EDM paper (Eq. 57).
    
    This computes D(x; sigma) = E[x' | x], where x' ~ p_data and x = x' + n 
    with n ~ N(0, sigma^2 I).
    
    The formula from the paper:
    D(x; σ) = Σᵢ [N(x; xᵢ, σ²I) · xᵢ] / Σᵢ [N(x; xᵢ, σ²I)]
    
    Which expands to:
    D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
    
    This is a weighted average of all training images, where the weights are
    proportional to the likelihood of each training image generating the observed
    noisy image under Gaussian noise.
    
    Parameters:
    -----------
    x_noisy : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    x_all : torch.Tensor
        All training images used as reference distribution of shape (num_samples, C, H, W)
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
        
    Notes:
    ------
    - Uses log-sum-exp trick for numerical stability (subtracts max value before exp)
    - More numerically stable, especially for large distances or small sigma values
    - Computational complexity: O(N × B × C × H × W)
      where N is the number of training images and B is the batch size
    - Memory complexity: O(N × C × H × W)
    - Only feasible for small datasets like CIFAR-10
    
    Examples:
    ---------
    >>> import torch
    >>> from ideal_denoiser import ideal_denoiser
    >>> from utils.noise_utils import add_gaussian_noise
    >>> 
    >>> # Create sample data
    >>> train_images = torch.randn(1000, 3, 32, 32)  # 1000 training images
    >>> test_image = torch.randn(1, 3, 32, 32)  # 1 test image
    >>> 
    >>> # Add noise and denoise
    >>> sigma = 2.0
    >>> noisy_image = add_gaussian_noise(test_image, sigma)
    >>> denoised_image = ideal_denoiser(noisy_image, sigma, train_images)
    >>> 
    >>> print(f"Noisy shape: {noisy_image.shape}")
    >>> print(f"Denoised shape: {denoised_image.shape}")
    """
    # Compute squared L2 distance between noisy images and all training images
    # x_all: (N, C, H, W), x_noisy: (B, C, H, W)
    # Result: (N, B)
    norm2 = ((x_all[:, None, :, :, :] - x_noisy[None, :, :, :, :]) ** 2).sum(dim=(2, 3, 4))
    
    # Compute log probabilities: log p(x | x_i) = -||x - x_i||^2 / (2*sigma^2)
    sigma_norm2 = -norm2 / (2 * sigma ** 2)
    
    # Numerical stability: subtract max value before exp (log-sum-exp trick)
    delta = torch.max(sigma_norm2, dim=0, keepdim=True)[0]
    
    # Compute exp of log probabilities
    exp_norm2 = (sigma_norm2 - delta).exp()
    
    # Compute weighted sum: numerator and denominator
    # exp_norm2: (N, B) -> (N, B, 1, 1, 1)
    # x_all: (N, C, H, W) -> (N, 1, C, H, W)
    numerator = exp_norm2[:, :, None, None, None] * x_all[:, None, :, :, :]  # (N, B, C, H, W)
    denominator = exp_norm2.sum(dim=0)  # (B,)
    
    # Compute denoised images
    denoised = numerator.sum(dim=0) / denominator[:, None, None, None]  # (B, C, H, W)
    
    return denoised


__all__ = ['ideal_denoiser']

