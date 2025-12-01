"""
Utility modules for image processing and visualization.

This package contains common utilities used across different denoising methods:
- core: Core utilities (noise generation, image loading, normalization)
- visualization: Visualization and plotting utilities
"""

from .core import (
    add_gaussian_noise,
    load_cifar10_dataset,
    load_cifar10_subset,
    normalize_for_display
)
from .visualization import create_labeled_figure, create_comparison_figure

__all__ = [
    'add_gaussian_noise',
    'load_cifar10_dataset',
    'load_cifar10_subset',
    'normalize_for_display',
    'create_labeled_figure',
    'create_comparison_figure'
]

