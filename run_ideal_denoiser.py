"""
Run Ideal Denoiser with CLI Arguments

This script demonstrates the ideal denoiser (Equation 57 from EDM paper)
on CIFAR-10 images at various noise levels with configurable parameters via CLI.

For selected train and test images with various noise levels, it generates visualizations
showing:
- Row 1: Noisy images at different sigma levels
- Row 2: Results from ideal denoiser

This allows visual assessment of the theoretical optimal denoising performance.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364

Note:
    While the ideal denoiser concept and formula (Equation 57) are from the EDM paper,
    this implementation is our own work. The original EDM repository does not include
    code for the ideal denoiser.

Usage:
    python run_ideal_denoiser.py --num-images 3 --train-size 1000
    python run_ideal_denoiser.py --sigma-list 0 0.5 1 2 5 10 --device cuda
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

# Import ideal denoiser
from ideal_denoiser import ideal_denoiser
from utils import add_gaussian_noise, load_cifar10_subset, normalize_for_display
from utils.visualization import create_comparison_figure


def process_images_at_sigma(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma: float,
    device: str
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
        Noise level
    device : str
        Device to run computations on
        
    Returns:
    --------
    tuple : (noisy, ideal_denoised)
        Two tensors containing the noisy images and denoised versions
    """
    # Handle sigma = 0 case
    if sigma == 0:
        return (
            selected_images.clone(),
            selected_images.clone()
        )
    
    # Add noise (in float32, matching how data is loaded)
    noisy_batch = add_gaussian_noise(selected_images, sigma)
    
    # Denoise with ideal denoiser
    with torch.no_grad():
        ideal_denoised_batch = ideal_denoiser(
            noisy_batch,
            sigma,
            train_images
        )
    
    return noisy_batch, ideal_denoised_batch


def generate_denoiser_output(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma_values: list,
    dataset_name: str,
    save_path: str,
    device: str = 'cpu'
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
        List of noise levels to test
    dataset_name : str
        Name of the dataset ('train' or 'test') for naming output file
    save_path : str
        Full path to save output image
    device : str
        Device to run computations on ('cpu' or 'cuda')
        
    Returns:
    --------
    tuple : (noisy_grid, ideal_grid)
        Two grids containing noisy and ideal denoised images
    """
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nProcessing {num_images} images with {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
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
            device
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
    create_comparison_figure(
        noisy_grid_img,
        ideal_grid_img,
        sigma_values,
        save_path,
        num_sigmas
    )
    
    print(f"Saved: {save_path}")
    
    return noisy_grid_img, ideal_grid_img


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Ideal Denoiser on CIFAR-10 images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for CIFAR-10 data'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results/denoiser_runs',
        help='Directory to save output images'
    )
    
    # Image selection parameters
    parser.add_argument(
        '--num-images',
        type=int,
        default=3,
        help='Number of images to denoise from each dataset (train/test)'
    )
    
    # Denoiser parameters
    parser.add_argument(
        '--train-size',
        type=int,
        default=1000,
        help='Number of training images to use for ideal denoiser reference'
    )
    parser.add_argument(
        '--sigma-list',
        type=float,
        nargs='+',
        default=[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],
        help='List of sigma (noise level) values to test'
    )
    
    # Device parameters
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu or cuda). If not specified, auto-detect.'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (if not set, selection is non-deterministic)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run ideal denoiser with CLI arguments.
    
    Generates output figures for both training and test datasets,
    showing noisy images and results from ideal denoiser side-by-side
    for visual quality assessment.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Device selection
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*80)
    print("Ideal Denoiser - Equation 57 from EDM Paper")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Data root: {args.data_root}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Number of images: {args.num_images}")
    print(f"  Training images for denoiser: {args.train_size}")
    print(f"  Sigma values: {args.sigma_list}")
    print(f"  Random seed: {args.seed}")
    
    # Set random seed for reproducibility (only if provided)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load data subsets
    print("\n" + "="*80)
    print("Loading Data Subsets")
    print("="*80)
    
    print("\nLoading CIFAR-10 subsets for random image selection...")
    train_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=True,
        max_samples=None
    )
    test_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=False,
        max_samples=None
    )
    
    num_available = len(train_subset)
    if args.num_images > num_available:
        raise ValueError(
            f"Requested num-images={args.num_images} but only {num_available} samples are available."
        )
    
    indices = np.random.choice(num_available, size=args.num_images, replace=False)
    
    print(f"\n  Randomly selected indices: {indices.tolist()}")
    
    train_selected = train_subset[indices]
    test_selected = test_subset[indices]
    
    print(f"Selected {len(train_selected)} training images")
    print(f"Selected {len(test_selected)} test images")
    
    # Load training images for ideal denoiser reference
    print(f"\nLoading {args.train_size} training images for ideal denoiser...")
    train_images_for_denoiser = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=True,
        max_samples=args.train_size
    )
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive filename with key config parameters
    sigma_min = min(args.sigma_list)
    sigma_max = max(args.sigma_list)
    filename_base = f"{timestamp}_n{args.num_images}_s{sigma_min}-{sigma_max}_train{args.train_size}"
    
    # Generate output for training set
    print("\n" + "="*80)
    print("Processing Training Set")
    print("="*80)
    
    train_output_path = os.path.join(args.save_dir, f"{filename_base}_train.png")
    generate_denoiser_output(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        sigma_values=args.sigma_list,
        dataset_name="train",
        save_path=train_output_path,
        device=device
    )
    
    # Generate output for test set
    print("\n" + "="*80)
    print("Processing Test Set")
    print("="*80)
    
    test_output_path = os.path.join(args.save_dir, f"{filename_base}_test.png")
    generate_denoiser_output(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        sigma_values=args.sigma_list,
        dataset_name="test",
        save_path=test_output_path,
        device=device
    )
    
    print("\n" + "="*80)
    print("Completed Successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Training set: {train_output_path}")
    print(f"  Test set: {test_output_path}")
    print(f"\nEach figure shows:")
    print(f"  Row 1: Noisy images at different noise levels")
    print(f"  Row 2: Denoised results using ideal denoiser")
    print()


if __name__ == "__main__":
    main()

