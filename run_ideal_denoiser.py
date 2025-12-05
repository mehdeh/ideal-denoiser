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
    python run_ideal_denoiser.py --sigma-list 0 1 2 5 --denoise-sigma 0.5
"""

import torch
import numpy as np
import os
import argparse
from datetime import datetime

# Import utilities
from utils import load_cifar10_subset, generate_denoiser_output


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
        default='./results',
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
        default=50000,
        help='Number of training images to use for ideal denoiser reference'
    )
    parser.add_argument(
        '--sigma-list',
        type=float,
        nargs='+',
        default=[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],
        help='List of sigma (noise level) values to test for noising images'
    )
    parser.add_argument(
        '--denoise-sigma',
        type=float,
        default=None,
        help='Fixed sigma value for denoising all images. If not specified (None), '
             'each noised image will be denoised with the same sigma used for noising. '
             'If specified, all noised images will be denoised with this fixed sigma value.'
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
    print(f"  Noising sigma values: {args.sigma_list}")
    if args.denoise_sigma is not None:
        print(f"  Denoising sigma (fixed): {args.denoise_sigma}")
    else:
        print(f"  Denoising sigma: same as noising sigma")
    print(f"  Random seed: {args.seed}")
    
    # Set random seed for reproducibility (only if provided)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load data subsets
    print("\n" + "="*80)
    print("Loading Data Subsets")
    print("="*80)
    
    print("\nLoading CIFAR-10 training subset for random selection and ideal denoiser...")
    train_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=True,
        max_samples=args.train_size
    )
    print("\nLoading full CIFAR-10 test set for random selection...")
    test_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=False,
        max_samples=None
    )
    
    # Generate separate random indices for train and test sets
    num_train_available = len(train_subset)
    num_test_available = len(test_subset)
    
    if args.num_images > num_train_available:
        raise ValueError(
            f"Requested num-images={args.num_images} but only {num_train_available} training samples are available."
        )
    if args.num_images > num_test_available:
        raise ValueError(
            f"Requested num-images={args.num_images} but only {num_test_available} test samples are available."
        )
    
    train_indices = np.random.choice(num_train_available, size=args.num_images, replace=False)
    test_indices = np.random.choice(num_test_available, size=args.num_images, replace=False)
    
    print(f"\n  Randomly selected train indices: {train_indices.tolist()}")
    print(f"  Randomly selected test indices: {test_indices.tolist()}")
    
    train_selected = train_subset[train_indices]
    test_selected = test_subset[test_indices]
    
    print(f"Selected {len(train_selected)} training images")
    print(f"Selected {len(test_selected)} test images")
    
    # Training images for ideal denoiser reference: use the same train subset
    train_images_for_denoiser = train_subset
    
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
        device=device,
        denoise_sigma=args.denoise_sigma,
        use_edm_style=False  # Use standard comparison visualization
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
        device=device,
        denoise_sigma=args.denoise_sigma,
        use_edm_style=False  # Use standard comparison visualization
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

