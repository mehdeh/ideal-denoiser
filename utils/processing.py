"""
Image processing utilities for ideal denoising experiments.

This module provides common functions for processing images with noise and denoising,
including batch processing at various noise levels and grid generation.
"""

import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import os

from ideal_denoiser import ideal_denoiser
from .core import add_gaussian_noise, normalize_for_display
from .visualization import create_labeled_figure, create_comparison_figure


def process_images_at_sigma(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma: float,
    device: str,
    denoise_sigma: float = None
) -> tuple:
    """
    Process images at a specific noise level with ideal denoiser.
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Clean images to process
    train_images : torch.Tensor
        Training images for ideal denoiser reference
    sigma : float
        Noise level for adding noise to images
    device : str
        Device to run computations on
    denoise_sigma : float, optional
        Noise level for denoising. If None, uses the same sigma as noising.
        This allows denoising at a different noise level than what was used for noising.
        
    Returns:
    --------
    tuple : (noisy, ideal_denoised)
        Two tensors containing the noisy images and denoised versions
        
    Examples:
    ---------
    >>> import torch
    >>> from utils.processing import process_images_at_sigma
    >>> 
    >>> # Process images at sigma=2.0
    >>> clean_images = torch.randn(3, 3, 32, 32)
    >>> train_images = torch.randn(1000, 3, 32, 32)
    >>> noisy, denoised = process_images_at_sigma(clean_images, train_images, 2.0, 'cpu')
    """
    # Determine which sigma to use for denoising
    sigma_for_denoising = denoise_sigma if denoise_sigma is not None else sigma
    
    # Handle sigma = 0 case for noising
    if sigma == 0:
        noisy_batch = selected_images.clone()
    else:
        # Add noise (in float32, matching how data is loaded)
        noisy_batch = add_gaussian_noise(selected_images, sigma)
    
    # Handle sigma = 0 case for denoising
    if sigma_for_denoising == 0:
        ideal_denoised_batch = noisy_batch.clone()
    else:
        # Denoise with ideal denoiser using the specified denoising sigma
        with torch.no_grad():
            ideal_denoised_batch = ideal_denoiser(
                noisy_batch,
                sigma_for_denoising,
                train_images
            )
    
    return noisy_batch, ideal_denoised_batch


def generate_denoiser_output(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma_values: list,
    dataset_name: str,
    save_path: str,
    device: str = 'cpu',
    denoise_sigma: float = None,
    use_edm_style: bool = False
) -> tuple:
    """
    Generate output of ideal denoiser across noise levels.
    
    This function processes selected images by:
    1. Adding Gaussian noise at various sigma levels
    2. Denoising with ideal denoiser (closed-form solution)
    3. Creating comparative visualizations
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Selected images to process (from train or test set)
        Shape: (num_images, C, H, W)
    train_images : torch.Tensor
        CIFAR-10 training images (used for ideal denoiser)
        Shape: (num_train, C, H, W)
    sigma_values : list
        List of noise levels to test for adding noise
    dataset_name : str
        Name of the dataset ('train' or 'test') for naming output file
    save_path : str
        Full path to save output image
    device : str
        Device to run computations on ('cpu' or 'cuda')
    denoise_sigma : float, optional
        Noise level for denoising. If None, each image is denoised with the same
        sigma used for noising. If specified, all images are denoised with this sigma.
    use_edm_style : bool
        If True, uses EDM paper style visualization (create_labeled_figure).
        If False, uses standard comparison visualization (create_comparison_figure).
        
    Returns:
    --------
    tuple : (noisy_grid, ideal_grid)
        Two grids containing noisy and ideal denoised images
        
    Examples:
    ---------
    >>> import torch
    >>> from utils.processing import generate_denoiser_output
    >>> 
    >>> selected = torch.randn(3, 3, 32, 32)
    >>> train = torch.randn(1000, 3, 32, 32)
    >>> sigmas = [0, 0.5, 1, 2, 5]
    >>> 
    >>> noisy_grid, denoised_grid = generate_denoiser_output(
    ...     selected, train, sigmas, "test", "output.png", "cpu"
    ... )
    """
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nProcessing {num_images} images with {num_sigmas} sigma values...")
    print(f"Noising sigma values: {sigma_values}")
    if denoise_sigma is not None:
        print(f"Denoising sigma (fixed): {denoise_sigma}")
    else:
        print(f"Denoising sigma: same as noising sigma")
    print(f"Dataset: {dataset_name}")
    
    # Storage for results
    noisy_images_all = []
    ideal_denoised_all = []
    
    # Process each sigma value with batch of all images
    for sigma in tqdm(sigma_values, desc=f"Processing {dataset_name} set"):
        noisy, ideal_denoised = process_images_at_sigma(
            selected_images,
            train_images,
            sigma,
            device,
            denoise_sigma=denoise_sigma
        )
        
        noisy_images_all.append(noisy)
        ideal_denoised_all.append(ideal_denoised)
    
    # Stack and organize images: transpose from (num_sigmas, num_images, C, H, W)
    # to (num_images, num_sigmas, C, H, W) then flatten to grid format
    noisy_stack = torch.stack(noisy_images_all, dim=0).transpose(0, 1)
    ideal_stack = torch.stack(ideal_denoised_all, dim=0).transpose(0, 1)
    
    # Flatten to grid format
    noisy_grid = noisy_stack.reshape(-1, *noisy_stack.shape[2:])
    ideal_grid = ideal_stack.reshape(-1, *ideal_stack.shape[2:])
    
    # Normalize for display
    noisy_display = normalize_for_display(noisy_grid)
    ideal_display = normalize_for_display(ideal_grid)
    
    # Create grids
    print(f"Creating image grids for {dataset_name}...")
    noisy_grid_img = make_grid(noisy_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    ideal_grid_img = make_grid(ideal_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Create combined visualization with labels
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if use_edm_style:
        # Use EDM paper style visualization
        create_labeled_figure(
            noisy_grid_img,
            ideal_grid_img,
            sigma_values,
            save_path,
            num_sigmas
        )
    else:
        # Use standard comparison visualization
        create_comparison_figure(
            noisy_grid_img,
            ideal_grid_img,
            sigma_values,
            save_path,
            num_sigmas
        )
    
    print(f"Saved: {save_path}")
    
    return noisy_grid_img, ideal_grid_img


__all__ = ['process_images_at_sigma', 'generate_denoiser_output']

