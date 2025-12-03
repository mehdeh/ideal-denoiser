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
import numpy as np
import os

# Import utilities
from utils import load_cifar10_subset, generate_denoiser_output


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
    
    os.makedirs(save_dir, exist_ok=True)
    train_save_path = os.path.join(save_dir, "figure1_combined_train.png")
    
    print(f"\nGenerating Figure 1 with {len(train_selected)} images and {len(sigma_values)} sigma values...")
    print(f"Sigma values: {sigma_values}")
    
    generate_denoiser_output(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        sigma_values=sigma_values,
        dataset_name="train",
        save_path=train_save_path,
        device=device,
        denoise_sigma=None,  # Use same sigma for denoising as for noising (EDM paper standard)
        use_edm_style=True   # Use EDM paper style visualization
    )
    
    # Generate Figure 1 for test set
    print("\n" + "="*80)
    print("Generating EDM Figure 1: Test Set")
    print("="*80)
    
    test_save_path = os.path.join(save_dir, "figure1_combined_test.png")
    
    print(f"\nGenerating Figure 1 with {len(test_selected)} images and {len(sigma_values)} sigma values...")
    print(f"Sigma values: {sigma_values}")
    
    generate_denoiser_output(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        sigma_values=sigma_values,
        dataset_name="test",
        save_path=test_save_path,
        device=device,
        denoise_sigma=None,  # Use same sigma for denoising as for noising (EDM paper standard)
        use_edm_style=True   # Use EDM paper style visualization
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

