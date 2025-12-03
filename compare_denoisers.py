"""
Compare Ideal Denoiser Performance Across Different Noise Levels

This script demonstrates the ideal denoiser (Equation 57 from EDM paper) 
on CIFAR-10 images at various noise levels.

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
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import ideal denoiser
from ideal_denoiser import ideal_denoiser, get_available_denoiser_methods
from utils import add_gaussian_noise, load_cifar10_subset, normalize_for_display
from utils.visualization import create_comparison_figure


def process_images_at_sigma(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma: float,
    device: str,
    denoiser_func=None,
    denoiser_params=None
) -> tuple:
    """
    Process images at a specific noise level with configurable denoiser.
    
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
    denoiser_func : callable, optional
        Denoiser function to use (default: ideal_denoiser)
    denoiser_params : dict, optional
        Additional parameters for the denoiser function
        
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
    
    # Use default denoiser if not specified
    if denoiser_func is None:
        denoiser_func = ideal_denoiser
        denoiser_params = {}
    elif denoiser_params is None:
        denoiser_params = {}
    
    # Denoise with specified denoiser
    with torch.no_grad():
        ideal_denoised_batch = denoiser_func(
            noisy_batch,
            sigma,
            train_images,
            **denoiser_params
        )
    
    return noisy_batch, ideal_denoised_batch


def generate_denoiser_comparison(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma_values: list,
    dataset_name: str,
    save_dir: str,
    device: str = 'cpu',
    denoiser_func=None,
    denoiser_params=None,
    method_name: str = 'original'
) -> tuple:
    """
    Generate comparison of ideal denoiser across noise levels.
    
    This function processes selected images by:
    1. Adding Gaussian noise at various sigma levels
    2. Denoising with specified denoiser (closed-form solution)
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
    save_dir : str
        Directory to save output images
    device : str
        Device to run computations on ('cpu' or 'cuda')
    denoiser_func : callable, optional
        Denoiser function to use (default: ideal_denoiser)
    denoiser_params : dict, optional
        Additional parameters for the denoiser function
    method_name : str
        Name of the denoising method for file naming
        
    Returns:
    --------
    tuple : (noisy_grid, ideal_grid)
        Two grids containing noisy and ideal denoised images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nGenerating comparison with {num_images} images and {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
    print(f"Denoising method: {method_name}")
    
    # Storage for results
    noisy_images_all = []
    ideal_denoised_all = []
    
    # Process each sigma value with batch of all images
    for sigma in tqdm(sigma_values, desc="Processing sigma values"):
        noisy, ideal_denoised = process_images_at_sigma(
            selected_images,
            train_images,
            sigma,
            device,
            denoiser_func,
            denoiser_params
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
    print("\nCreating image grids...")
    noisy_grid_img = make_grid(noisy_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    ideal_grid_img = make_grid(ideal_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Create combined visualization with labels
    combined_path = os.path.join(save_dir, f"comparison_{dataset_name}_{method_name}.png")
    create_comparison_figure(
        noisy_grid_img,
        ideal_grid_img,
        sigma_values,
        combined_path,
        num_sigmas
    )
    
    return noisy_grid_img, ideal_grid_img


def setup_data_subsets(data_root: str, config: dict) -> tuple:
    """
    Load data subsets for comparison.
    
    Parameters:
    -----------
    data_root : str
        Root directory for CIFAR-10 data
    config : dict
        Configuration dictionary with selection parameters
        
    Returns:
    --------
    tuple : (train_selected, test_selected, train_images_for_denoiser)
    """
    print("\n" + "="*80)
    print("Loading image subsets...")
    print("="*80)
    
    # Load small subsets for selection
    train_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=config['max_samples_for_selection']
    )
    test_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=False,
        max_samples=config['max_samples_for_selection']
    )
    
    # Select specific images
    train_selected = train_subset[config['train_selection_indices']]
    test_selected = test_subset[config['test_selection_indices']]
    
    # Load training images for ideal denoiser reference
    print(f"\nLoading {config['ideal_denoiser_subset_size']} training images for ideal denoiser...")
    train_images_for_denoiser = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=config['ideal_denoiser_subset_size']
    )
    
    return train_selected, test_selected, train_images_for_denoiser


def generate_multi_method_comparison(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    sigma_values: list,
    dataset_name: str,
    save_dir: str,
    device: str,
    methods_to_compare: list
) -> None:
    """
    Generate comparisons for one or more denoising methods.
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Images to process
    train_images : torch.Tensor
        Training images for denoiser reference
    sigma_values : list
        List of noise levels
    dataset_name : str
        'train' or 'test'
    save_dir : str
        Directory to save results
    device : str
        Device to use
    methods_to_compare : list
        List of method names from available methods (currently only 'max' is supported)
    """
    available_methods = get_available_denoiser_methods()
    
    for method_name in methods_to_compare:
        if method_name not in available_methods:
            print(f"\nWarning: Method '{method_name}' not found, skipping...")
            continue
        
        method_config = available_methods[method_name]
        
        print("\n" + "="*80)
        print(f"Processing Method: {method_config['display_name']}")
        print("="*80)
        
        generate_denoiser_comparison(
            selected_images=selected_images,
            train_images=train_images,
            sigma_values=sigma_values,
            dataset_name=dataset_name,
            save_dir=save_dir,
            device=device,
            denoiser_func=method_config['function'],
            denoiser_params=method_config['params'],
            method_name=method_name
        )


def main():
    """
    Main function to demonstrate the ideal denoiser across different noise levels.
    
    Generates comparison figures for both training and test datasets,
    showing noisy images and results from the denoising method side-by-side
    for visual quality assessment.
    """
    # Configuration
    config = {
        'data_root': "./data",
        'save_dir': "./results/denoiser_comparison",
        'sigma_values': [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],
        'max_samples_for_selection': 10,
        'train_selection_indices': [2, 3, 4],
        'test_selection_indices': [2, 3, 4],
        'ideal_denoiser_subset_size': 1000,
        'methods_to_compare': ['max']
    }
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data subsets
    train_selected, test_selected, train_images_for_denoiser = setup_data_subsets(
        config['data_root'],
        config
    )
    
    # Display available methods
    available_methods = get_available_denoiser_methods()
    print("\n" + "="*80)
    print("Available Denoising Methods:")
    print("="*80)
    for method_name, method_config in available_methods.items():
        print(f"  - {method_name}: {method_config['display_name']}")
    
    # Generate comparisons for training set
    print("\n" + "="*80)
    print("Generating Comparisons: Training Set")
    print("="*80)
    
    generate_multi_method_comparison(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        sigma_values=config['sigma_values'],
        dataset_name="train",
        save_dir=config['save_dir'],
        device=device,
        methods_to_compare=config['methods_to_compare']
    )
    
    # Generate comparisons for test set
    print("\n" + "="*80)
    print("Generating Comparisons: Test Set")
    print("="*80)
    
    generate_multi_method_comparison(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        sigma_values=config['sigma_values'],
        dataset_name="test",
        save_dir=config['save_dir'],
        device=device,
        methods_to_compare=config['methods_to_compare']
    )
    
    print("\n" + "="*80)
    print("Comparison generation completed successfully!")
    print("="*80)
    print(f"\nOutput files saved in: {config['save_dir']}/")
    print("\nGenerated files for each method:")
    for method_name in config['methods_to_compare']:
        if method_name in available_methods:
            display_name = available_methods[method_name]['display_name']
            print(f"  - comparison_train_{method_name}.png: Training set with {display_name}")
            print(f"  - comparison_test_{method_name}.png: Test set with {display_name}")
    print("\nEach figure shows:")
    print("  Row 1: Noisy images at different noise levels")
    print("  Row 2: Denoised results using the specific method")


if __name__ == "__main__":
    main()
