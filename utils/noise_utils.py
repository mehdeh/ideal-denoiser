"""
Noise utilities for image denoising experiments.

This module provides functions for adding various types of noise to images,
commonly used across different denoising methods.
"""

import torch


def add_gaussian_noise(images, sigma):
    """
    Add Gaussian noise to images.
    
    Parameters:
    -----------
    images : torch.Tensor
        Clean images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Standard deviation of Gaussian noise
        
    Returns:
    --------
    noisy_images : torch.Tensor
        Noisy images of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from utils.noise_utils import add_gaussian_noise
    >>> 
    >>> # Single image
    >>> clean_img = torch.randn(1, 3, 32, 32)
    >>> noisy_img = add_gaussian_noise(clean_img, sigma=2.0)
    >>> 
    >>> # Batch of images
    >>> clean_batch = torch.randn(10, 3, 32, 32)
    >>> noisy_batch = add_gaussian_noise(clean_batch, sigma=5.0)
    """
    if sigma == 0:
        return images.clone()
    
    noise = torch.randn_like(images) * sigma
    return images + noise

