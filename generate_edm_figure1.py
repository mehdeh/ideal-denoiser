"""
Generate Figure 1 from EDM Paper (Elucidating the Design Space of Diffusion-Based Generative Models)

This script reproduces the ideal denoiser visualization from the paper by:
1. Loading three sample images from both CIFAR-10 training and test sets
2. Adding Gaussian noise with various sigma values
3. Denoising using the ideal denoiser (closed-form solution from Eq. 57)
4. Visualizing both noisy and denoised images in combined grid format with titles and sigma labels

The ideal denoiser is computed using the entire CIFAR-10 training set as the reference distribution.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364
    Ideal Denoiser Formula: Appendix B.3, Equation 57
"""

import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import ideal denoiser and utilities
from ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset, load_cifar10_subset, normalize_for_display
from utils.visualization import create_labeled_figure


def generate_figure1(selected_images, train_images, 
                     sigma_values=[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],
                     dataset_name="train",
                     save_dir="./results",
                     device='cpu'):
    """
    Generate Figure 1 from EDM paper showing ideal denoiser performance.
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Selected images to process (from train or test set)
    train_images : torch.Tensor
        CIFAR-10 training images (used for ideal denoiser)
    sigma_values : list
        List of noise levels to test
    dataset_name : str
        Name of the dataset ('train' or 'test') for naming output file
    save_dir : str
        Directory to save output images
    device : str
        Device to run computations on ('cpu' or 'cuda')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nGenerating Figure 1 with {num_images} images and {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
    
    # Storage for results
    noisy_images_all = []
    denoised_images_all = []
    
    # Process each sigma value with batch of all images (more efficient)
    for sigma_idx, sigma in enumerate(tqdm(sigma_values, desc="Processing sigma values")):
        # Add noise to all images at once
        noisy_batch = add_gaussian_noise(selected_images, sigma)  # (num_images, C, H, W)
        
        # Denoise using ideal denoiser
        if sigma == 0:
            denoised_batch = selected_images.clone()
        else:
            with torch.no_grad():
                denoised_batch = ideal_denoiser(noisy_batch, sigma, train_images)
        
        noisy_images_all.append(noisy_batch)
        denoised_images_all.append(denoised_batch)
    
    # Stack all images: transpose from (num_sigmas, num_images, C, H, W) to (num_images, num_sigmas, C, H, W)
    # then flatten to (num_images * num_sigmas, C, H, W)
    noisy_stack = torch.stack(noisy_images_all, dim=0)  # (num_sigmas, num_images, C, H, W)
    denoised_stack = torch.stack(denoised_images_all, dim=0)  # (num_sigmas, num_images, C, H, W)
    
    # Transpose to organize by image rows, sigma columns
    noisy_stack = noisy_stack.transpose(0, 1)  # (num_images, num_sigmas, C, H, W)
    denoised_stack = denoised_stack.transpose(0, 1)  # (num_images, num_sigmas, C, H, W)
    
    # Flatten to grid format
    noisy_images_grid = noisy_stack.reshape(-1, *noisy_stack.shape[2:])  # (num_images * num_sigmas, C, H, W)
    denoised_images_grid = denoised_stack.reshape(-1, *denoised_stack.shape[2:])  # (num_images * num_sigmas, C, H, W)
    
    # Normalize for display
    noisy_images_display = normalize_for_display(noisy_images_grid)
    denoised_images_display = normalize_for_display(denoised_images_grid)
    
    # Create grids
    print("\nCreating image grids...")
    noisy_grid = make_grid(noisy_images_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    denoised_grid = make_grid(denoised_images_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Create combined visualization with labels
    combined_path = os.path.join(save_dir, f"figure1_combined_{dataset_name}.png")
    create_labeled_figure(noisy_grid, denoised_grid, sigma_values, combined_path, num_sigmas)
    
    return noisy_grid, denoised_grid


def main():
    """
    Main function to generate Figure 1 from EDM paper.
    Generates combined figures for both training and test datasets.
    
    Optimized to load only small subsets instead of entire dataset for faster execution.
    """
    # Configuration
    data_root = "./data"
    save_dir = "./results/edm_figure1"
    sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
    
    # Optimization: Only load a small subset (e.g., 10 images) and select 3 from them
    # This avoids loading the entire dataset (50K train + 10K test images)
    max_samples_for_selection = 10  # Load only first 10 images for selection
    train_selection_indices = [2, 3, 4]  # Select 3 images from the loaded subset (indices relative to subset)
    test_selection_indices = [2, 3, 4]   # Select 3 images from the loaded subset
    
    # For ideal denoiser, use a reasonable subset (1000 images) instead of full 50K
    # This is much faster while still providing good denoising quality
    ideal_denoiser_subset_size = 1000  # Use 1000 training images for ideal denoiser reference
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load small subset for selecting images to visualize (much faster!)
    print("\n" + "="*80)
    print("Loading small subsets for image selection (optimized)...")
    print("="*80)
    
    train_subset = load_cifar10_subset(
        root=data_root, 
        normalize=True, 
        train=True, 
        max_samples=max_samples_for_selection
    )
    test_subset = load_cifar10_subset(
        root=data_root, 
        normalize=True, 
        train=False, 
        max_samples=max_samples_for_selection
    )
    
    # Select images from the subset
    train_selected = train_subset[train_selection_indices]
    test_selected = test_subset[test_selection_indices]
    
    # Load subset for ideal denoiser reference (much smaller than full 50K)
    print("\n" + "="*80)
    print(f"Loading {ideal_denoiser_subset_size} training images for ideal denoiser reference...")
    print("="*80)
    train_images_for_denoiser = load_cifar10_subset(
        root=data_root, 
        normalize=True, 
        train=True, 
        max_samples=ideal_denoiser_subset_size
    )
    
    # Generate Figure 1 for training set
    print("\n" + "="*80)
    print("Generating EDM Figure 1: Training Set")
    print("="*80)
    
    generate_figure1(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        sigma_values=sigma_values,
        dataset_name="train",
        save_dir=save_dir,
        device=device
    )
    
    # Generate Figure 1 for test set
    print("\n" + "="*80)
    print("Generating EDM Figure 1: Test Set")
    print("="*80)
    
    generate_figure1(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        sigma_values=sigma_values,
        dataset_name="test",
        save_dir=save_dir,
        device=device
    )
    
    print("\n" + "="*80)
    print("Figure generation completed successfully!")
    print("="*80)
    print(f"\nOutput files saved in: {save_dir}/")
    print("- figure1_combined_train.png: Combined visualization for training set")
    print("- figure1_combined_test.png: Combined visualization for test set")
    print(f"\nOptimization: Loaded only {max_samples_for_selection} images for selection")
    print(f"and {ideal_denoiser_subset_size} images for ideal denoiser (instead of 50K+10K)")


if __name__ == "__main__":
    main()

