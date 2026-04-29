"""
data.py - MNIST Data Loader for Optical Neural Network
=======================================================

Prepares MNIST digits for the optical system:
    28×28 → resize to DIGIT_SIZE → center-pad to LCD_SIZE (200×200)

All parameters come from config.py - no hardcoded values.

Usage:
    from data import get_mnist_loaders
    train_loader, test_loader = get_mnist_loaders()
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from config import Config


# =============================================================================
# CONFIGURATION (all from config.py)
# =============================================================================

config = Config()
LCD_SIZE = config.lcd.resolution[0]       # LCD pixels (200)
DIGIT_SIZE = config.input.digit_size      # Digit size before padding (28 = no resize)
PADDING_VALUE = config.input.pad_value    # Background value (0.0 = black)
NORMALIZE_ENERGY = config.input.normalize_energy  # Constant total energy per image
DATA_DIR = './data'                       # MNIST download location


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist_resized(lcd_size=None, digit_size=None):
    """
    Load MNIST, resize, and center-pad to LCD size.
    
    Pipeline: 28×28 → resize to digit_size → center-pad to lcd_size
    
    Returns:
        x_train: [60000, 1, lcd_size, lcd_size] float32, values in [0, 1]
        y_train: [60000] int64, labels 0-9
        x_test:  [10000, 1, lcd_size, lcd_size] float32, values in [0, 1]  
        y_test:  [10000] int64, labels 0-9
    """
    lcd = lcd_size if lcd_size is not None else LCD_SIZE
    digit = digit_size if digit_size is not None else DIGIT_SIZE
    
    print(f"Loading MNIST...")
    print(f"  Resize: 28×28 → {digit}×{digit}")
    print(f"  Pad to: {lcd}×{lcd} (background={PADDING_VALUE})")
    
    # Download/load MNIST
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    
    # Extract to tensors
    print("Extracting data...")
    x_train = torch.stack([img for img, _ in train_data])
    y_train = torch.tensor([label for _, label in train_data])
    x_test = torch.stack([img for img, _ in test_data])
    y_test = torch.tensor([label for _, label in test_data])
    
    # Resize if needed (skip if digit_size == 28)
    if digit != 28:
        print(f"Resizing to {digit}×{digit}...")
        x_train = F.interpolate(x_train, size=(digit, digit), mode='bilinear', align_corners=False)
        x_test = F.interpolate(x_test, size=(digit, digit), mode='bilinear', align_corners=False)
    
    # Center-pad to LCD size
    if lcd > digit:
        pad = (lcd - digit) // 2
        pad_r = lcd - digit - pad  # Handle odd differences
        print(f"Padding to {lcd}×{lcd} (border={pad}px)...")
        x_train = F.pad(x_train, (pad, pad_r, pad, pad_r), value=PADDING_VALUE)
        x_test = F.pad(x_test, (pad, pad_r, pad, pad_r), value=PADDING_VALUE)
    
    # Normalize so each image has the same total energy (constant laser power)
    if NORMALIZE_ENERGY:
        train_energy = x_train.sum(dim=(1, 2, 3), keepdim=True)  # [60000, 1, 1, 1]
        mean_energy = train_energy.mean()                         # scalar
        x_train = x_train * (mean_energy / (train_energy + 1e-8))
        
        test_energy = x_test.sum(dim=(1, 2, 3), keepdim=True)
        x_test = x_test * (mean_energy / (test_energy + 1e-8))
        print(f"Energy normalized: target={mean_energy:.1f} per image")
    
    print(f"Done! Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


def get_mnist_loaders(batch_size=None):
    """
    Get PyTorch DataLoaders for training.
    
    Args:
        batch_size: Images per batch (default: from config.py)
    
    Returns:
        train_loader: Shuffled training data
        test_loader:  Test data (not shuffled)
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    if batch_size is None:
        batch_size = config.training.batch_size
    
    x_train, y_train, x_test, y_test = load_mnist_resized()
    
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


# =============================================================================
# VISUALIZATION (for debugging)
# =============================================================================

def show_samples(x, y, num_per_class=5):
    """Show sample digits from each class (0-9)."""
    fig, axes = plt.subplots(10, num_per_class, figsize=(num_per_class * 2, 20))
    
    for digit in range(10):
        indices = (y == digit).nonzero(as_tuple=True)[0]
        selected = indices[torch.randperm(len(indices))[:num_per_class]]
        
        for j, idx in enumerate(selected):
            axes[digit, j].imshow(x[idx].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[digit, j].axis('off')
            if j == 0:
                axes[digit, j].set_ylabel(f'{digit}', fontsize=12)
    
    plt.suptitle(f'MNIST: {DIGIT_SIZE}×{DIGIT_SIZE} in {LCD_SIZE}×{LCD_SIZE}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=100)
    plt.show()


# =============================================================================
# MAIN (test data loading)
# =============================================================================

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist_resized()
    
    print(f"\nSummary:")
    print(f"  Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"  Shape: {x_train.shape[-2]}×{x_train.shape[-1]}")
    print(f"  Range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    show_samples(x_train, y_train)
