"""
Image utilities for loading, processing, and normalizing images.

This module provides common image processing functions used across different
denoising methods.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def load_cifar10_dataset(root="./data", normalize=True):
    """
    Load CIFAR-10 training and test datasets.
    
    Parameters:
    -----------
    root : str
        Root directory where CIFAR-10 data will be downloaded/stored
    normalize : bool
        Whether to apply normalization to [-1, 1] range
        
    Returns:
    --------
    train_images : torch.Tensor
        Training images of shape (50000, 3, 32, 32)
    test_images : torch.Tensor
        Test images of shape (10000, 3, 32, 32)
        
    Examples:
    ---------
    >>> from utils.image_utils import load_cifar10_dataset
    >>> 
    >>> # Load normalized dataset
    >>> train_imgs, test_imgs = load_cifar10_dataset(root="./data", normalize=True)
    >>> print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")
    >>> 
    >>> # Load unnormalized dataset
    >>> train_raw, test_raw = load_cifar10_dataset(root="./data", normalize=False)
    """
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.ToTensor()
    
    # Load training set
    print("Loading CIFAR-10 training set...")
    trainset = torchvision.datasets.CIFAR10(
        root=root, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Load test set
    print("Loading CIFAR-10 test set...")
    testset = torchvision.datasets.CIFAR10(
        root=root, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Convert to tensors
    print("Converting datasets to tensors...")
    train_images = torch.stack([trainset[i][0] for i in tqdm(range(len(trainset)), desc="Train")])
    test_images = torch.stack([testset[i][0] for i in tqdm(range(len(testset)), desc="Test")])
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    return train_images, test_images


def load_cifar10_subset(root="./data", normalize=True, train=True, max_samples=None, selected_indices=None):
    """
    Load a subset of CIFAR-10 dataset (much faster than loading entire dataset).
    
    Parameters:
    -----------
    root : str
        Root directory where CIFAR-10 data will be downloaded/stored
    normalize : bool
        Whether to apply normalization to [-1, 1] range
    train : bool
        Whether to load training set (True) or test set (False)
    max_samples : int, optional
        Maximum number of samples to load. If None, loads all samples.
        If specified, loads first max_samples images.
    selected_indices : list of int, optional
        Specific indices to load. If provided, max_samples is ignored.
        
    Returns:
    --------
    images : torch.Tensor
        Selected images of shape (num_selected, 3, 32, 32)
        
    Examples:
    ---------
    >>> from utils.image_utils import load_cifar10_subset
    >>> 
    >>> # Load first 10 training images
    >>> train_subset = load_cifar10_subset(root="./data", train=True, max_samples=10)
    >>> 
    >>> # Load specific indices from test set
    >>> test_subset = load_cifar10_subset(root="./data", train=False, selected_indices=[20, 21, 22])
    """
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.ToTensor()
    
    # Load dataset (this doesn't load all images into memory, just creates the dataset object)
    dataset_name = "training" if train else "test"
    print(f"Loading CIFAR-10 {dataset_name} set...")
    dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    # Determine which indices to load
    if selected_indices is not None:
        indices = selected_indices
        print(f"Loading {len(indices)} specific images from {dataset_name} set (indices: {indices})...")
    elif max_samples is not None:
        indices = list(range(min(max_samples, len(dataset))))
        print(f"Loading first {len(indices)} images from {dataset_name} set...")
    else:
        indices = list(range(len(dataset)))
        print(f"Loading all {len(indices)} images from {dataset_name} set...")
    
    # Convert selected images to tensors
    images = torch.stack([dataset[i][0] for i in tqdm(indices, desc=f"Loading {dataset_name}")])
    
    print(f"Loaded {dataset_name} images shape: {images.shape}")
    
    return images


def normalize_for_display(images):
    """
    Min-max normalize images to [0, 1] range for display.
    Each image in the batch is normalized independently.
    
    Parameters:
    -----------
    images : torch.Tensor
        Images of shape (batch_size, C, H, W)
        
    Returns:
    --------
    normalized : torch.Tensor
        Normalized images in [0, 1] range
        
    Examples:
    ---------
    >>> import torch
    >>> from utils.image_utils import normalize_for_display
    >>> 
    >>> # Normalize batch of images for visualization
    >>> images = torch.randn(10, 3, 32, 32)  # Random images
    >>> normalized = normalize_for_display(images)
    >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    """
    batch_size = images.shape[0]
    images_flat = images.view(batch_size, -1)
    
    min_vals = images_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = images_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    
    normalized = (images - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized

