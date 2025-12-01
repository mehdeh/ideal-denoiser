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


def ideal_denoiser_enhanced(x_noisy, sigma, x_all, delta_method='max', delta_param=None):
    """
    Enhanced ideal denoiser with configurable delta computation methods.
    
    This is an improved version of the ideal denoiser that offers multiple strategies
    for computing the numerical stability parameter (delta) in the log-sum-exp trick.
    
    The formula from the paper:
    D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
    
    Different delta methods can affect numerical stability and potentially the
    denoising quality.
    
    Parameters:
    -----------
    x_noisy : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    x_all : torch.Tensor
        All training images used as reference distribution of shape (num_samples, C, H, W)
    delta_method : str, optional (default='max')
        Method for computing delta parameter:
        - 'max': Maximum value (most stable, default in original implementation)
        - 'mean': Median value (balanced approach, uses 50th percentile)
        - 'percentile': Percentile value (requires delta_param for percentile value)
        - 'adaptive': Adaptive interpolation between median and high percentile
        - 'mean_std': Mean + alpha * std approach (requires delta_param for alpha)
    delta_param : float, optional
        Parameter for specific delta methods:
        - For 'percentile': percentile value (0-100), default=95
        - For 'adaptive': interpolation coefficient (0-1), default=0.5
        - For 'mean_std': alpha coefficient for std, default=1.0
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
        
    Notes:
    ------
    - Uses log-sum-exp trick for numerical stability with configurable delta
    - Different delta methods may provide different trade-offs between stability
      and denoising quality
    - All methods use sigma-adaptive blending with max for stability at low sigma
    - Recommended to experiment with different methods for your specific use case
    
    Examples:
    ---------
    >>> import torch
    >>> from ideal_denoiser import ideal_denoiser_enhanced
    >>> 
    >>> # Create sample data
    >>> train_images = torch.randn(1000, 3, 32, 32)
    >>> noisy_image = torch.randn(1, 3, 32, 32)
    >>> sigma = 2.0
    >>> 
    >>> # Try different delta methods
    >>> denoised_max = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 'max')
    >>> denoised_mean = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 'mean')
    >>> denoised_perc = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 'percentile', 95)
    >>> denoised_adapt = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 'adaptive', 0.5)
    >>> denoised_std = ideal_denoiser_enhanced(noisy_image, sigma, train_images, 'mean_std', 1.0)
    """
    # Compute squared L2 distance between noisy images and all training images
    # x_all: (N, C, H, W), x_noisy: (B, C, H, W)
    # Result: (N, B)
    norm2 = ((x_all[:, None, :, :, :] - x_noisy[None, :, :, :, :]) ** 2).sum(dim=(2, 3, 4))
    
    # Compute log probabilities: log p(x | x_i) = -||x - x_i||^2 / (2*sigma^2)
    sigma_norm2 = -norm2 / (2 * sigma ** 2)
    
    # Compute delta based on selected method
    # Note: delta must be close to max(sigma_norm2) for numerical stability to avoid overflow
    # For small sigma values, we need to be even more conservative
    max_val = torch.max(sigma_norm2, dim=0, keepdim=True)[0]
    
    # Compute sigma-dependent blending factor for extra stability at low sigma
    # At low sigma, exp values grow faster, so we need to be more conservative
    sigma_scale = torch.clamp(torch.tensor(sigma / 2.0), 0.1, 1.0)  # Scale factor: higher for large sigma
    
    if delta_method == 'max':
        # Maximum value (default, most numerically stable)
        delta = max_val
        
    elif delta_method == 'mean':
        # Weighted mean: blend between max and median for smoother weighting
        median_val = torch.median(sigma_norm2, dim=0, keepdim=True)[0]
        # Use sigma-adaptive blending: more conservative at low sigma
        blend_ratio = 0.1 * sigma_scale  # 0.01-0.1 depending on sigma
        delta = (1 - blend_ratio) * max_val + blend_ratio * median_val
        
    elif delta_method == 'percentile':
        # High percentile value (requires delta_param)
        percentile = delta_param if delta_param is not None else 95.0
        percentile = max(90.0, percentile)
        k = int(sigma_norm2.size(0) * percentile / 100.0)
        k = max(1, min(k, sigma_norm2.size(0) - 1))
        percentile_val = torch.topk(sigma_norm2, k, dim=0, largest=True, sorted=False)[0].mean(dim=0, keepdim=True)
        # Sigma-adaptive blending (more conservative than mean method)
        blend_ratio = 0.08 * sigma_scale  # More conservative: 0.008-0.08
        delta = (1 - blend_ratio) * max_val + blend_ratio * percentile_val
        
    elif delta_method == 'adaptive':
        # Adaptive method: blend between max and high percentile
        coefficient = delta_param if delta_param is not None else 0.5
        coefficient = max(0.0, min(1.0, coefficient))
        k = int(sigma_norm2.size(0) * 0.95)
        k = max(1, min(k, sigma_norm2.size(0) - 1))
        high_percentile = torch.topk(sigma_norm2, k, dim=0, largest=True, sorted=False)[0].mean(dim=0, keepdim=True)
        # Sigma-adaptive interpolation
        blend_ratio = coefficient * 0.2 * sigma_scale
        blended = (1 - 0.5) * max_val + 0.5 * high_percentile
        delta = (1 - blend_ratio) * max_val + blend_ratio * blended
        
    elif delta_method == 'mean_std':
        # Mean + alpha * std method
        alpha = delta_param if delta_param is not None else 1.0
        mean_val = torch.mean(sigma_norm2, dim=0, keepdim=True)
        std_val = torch.std(sigma_norm2, dim=0, keepdim=True)
        computed_delta = mean_val + alpha * std_val
        # For safety, blend with max (more conservative at low sigma)
        blend_ratio = 0.2 * sigma_scale
        delta = (1 - blend_ratio) * max_val + blend_ratio * computed_delta
        
    else:
        raise ValueError(f"Unknown delta_method: {delta_method}. "
                        f"Valid options: 'max', 'mean', 'percentile', 'adaptive', 'mean_std'")
    
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


def get_available_denoiser_methods():
    """
    Get a dictionary of available denoiser methods with their configurations.
    
    Returns:
    --------
    dict : Dictionary mapping method names to their configuration
        Each configuration contains:
        - 'function': The denoiser function to use
        - 'params': Dictionary of parameters for the function
        - 'display_name': Human-readable name for visualization
    
    Examples:
    ---------
    >>> methods = get_available_denoiser_methods()
    >>> for name, config in methods.items():
    >>>     print(f"{name}: {config['display_name']}")
    """
    return {
        'max': {
            'function': ideal_denoiser_enhanced,
            'params': {'delta_method': 'max'},
            'display_name': 'Max (Original)'
        },
        'mean': {
            'function': ideal_denoiser_enhanced,
            'params': {'delta_method': 'mean'},
            'display_name': 'Median'
        },
        'percentile_95': {
            'function': ideal_denoiser_enhanced,
            'params': {'delta_method': 'percentile', 'delta_param': 95.0},
            'display_name': 'Percentile-95'
        },
        'adaptive': {
            'function': ideal_denoiser_enhanced,
            'params': {'delta_method': 'adaptive', 'delta_param': 0.5},
            'display_name': 'Adaptive'
        },
        'mean_std': {
            'function': ideal_denoiser_enhanced,
            'params': {'delta_method': 'mean_std', 'delta_param': 1.0},
            'display_name': 'Mean+Std'
        }
    }


__all__ = ['ideal_denoiser', 'ideal_denoiser_enhanced', 'get_available_denoiser_methods']

