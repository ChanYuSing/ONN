"""
visualize_propagation.py - ONN Visualization Tool
=================================================

Visualize light propagation through trained ONN layers.
Generates intensity/phase plots and exports masks as PNG for LCD display.

Usage: python visualize_propagation.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from physics import (
    propagate, 
    create_transfer_function,
    apply_fill_factor,
    apply_contrast_limits,
    apply_binary_quantization,
    apply_lcd_effects
)
from data import load_mnist_resized
from model import OpticalNeuralNetwork


# =============================================================================
# CONFIG
# =============================================================================

config = Config()
N = config.lcd.resolution[0]
WAVELENGTH = config.light.wavelength
PIXEL_PITCH = config.lcd.pixel_pitch
FILL_FACTOR = config.lcd.fill_factor
MIN_TRANSMISSION = config.lcd.min_transmission
MAX_TRANSMISSION = config.lcd.max_transmission
IS_BINARY = config.lcd.is_binary


# =============================================================================
# MASK FUNCTIONS
# =============================================================================

def create_random_masks(num_layers, seed=42):
    """Create random high-frequency masks (heavy diffraction, for demo)."""
    torch.manual_seed(seed)
    return [torch.sigmoid(torch.randn(N, N) * 0.5) for _ in range(num_layers)]


def create_smooth_masks(num_layers, seed=42):
    """Create smooth low-frequency masks (like trained networks)."""
    torch.manual_seed(seed)
    masks = []
    for _ in range(num_layers):
        low_res = N // 10
        mask_low = torch.randn(low_res, low_res) * 2
        mask = torch.nn.functional.interpolate(
            mask_low.unsqueeze(0).unsqueeze(0),
            size=(N, N), mode='bilinear', align_corners=False
        )[0, 0]
        masks.append(torch.sigmoid(mask))
    return masks


def load_masks(filepath):
    """Load masks from .pt file (supports list, dict, or checkpoint format)."""
    data = torch.load(filepath, weights_only=False)
    
    if isinstance(data, list):
        masks = data
    elif isinstance(data, dict) and 'masks' in data:
        masks = data['masks']
    elif isinstance(data, dict) and 'model_state_dict' in data:
        state_dict = data['model_state_dict']
        masks = []
        layer_idx = 0
        while True:
            key_raw = f'layers.{layer_idx}.raw'
            key_mask = f'layers.{layer_idx}.mask'
            if key_raw in state_dict:
                masks.append(state_dict[key_raw])
                layer_idx += 1
            elif key_mask in state_dict:
                masks.append(state_dict[key_mask])
                layer_idx += 1
            else:
                break
        if not masks:
            raise ValueError(f"No mask weights found in checkpoint {filepath}")
    else:
        raise ValueError(f"Unknown format in {filepath}")
    
    # Apply sigmoid if needed (raw values need sigmoid) and move to CPU
    processed_masks = []
    for m in masks:
        if m.min() < 0 or m.max() > 1:
            m = torch.sigmoid(m)
        m = m.detach().cpu() if m.is_cuda else m.detach()
        processed_masks.append(m)
    masks = processed_masks
    print(f"Loaded {len(masks)} masks from {filepath}")
    return masks


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint (auto-detects detection method)."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Detect detection method
    if 'detector.zone_logits' in state_dict:
        detection_method = 'maxzone'
    elif 'detector.pattern_to_logits' in state_dict:
        detection_method = 'binary'
    elif 'detector.classifier.weight' in state_dict:
        detection_method = 'center'
    else:
        detection_method = 'zone'
    
    # Infer grid size
    if 'detector.extractor.zones' in state_dict:
        num_zones = state_dict['detector.extractor.zones'].shape[0]
        import math
        grid_size = int(math.sqrt(num_zones))
        if grid_size * grid_size != num_zones:
            grid_size = int(math.ceil(math.sqrt(num_zones)))
    else:
        grid_size = 3
    
    model = OpticalNeuralNetwork(
        Config(), detection_method=detection_method, rows=grid_size, cols=grid_size
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded: {checkpoint_path}")
    print(f"  Method: {detection_method}, Grid: {grid_size}×{grid_size}, Acc: {checkpoint.get('best_acc', 0):.2f}%")
    return model


def predict(model, image, device='cpu'):
    """Run prediction on single image. Returns (class, probabilities)."""
    model.eval()
    with torch.no_grad():
        x = image.unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1).item(), probs[0]


def save_masks_as_images(masks, folder='mask_images'):
    """Save masks as PNG images for LCD display. [0,1] → [0,255] uint8."""
    from PIL import Image
    import os
    
    os.makedirs(folder, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_cpu = mask.detach().cpu() if mask.is_cuda else mask
        mask_uint8 = (mask_cpu * 255).clamp(0, 255).byte().numpy()
        Image.fromarray(mask_uint8).save(f"{folder}/mask_{i+1}.png")
        print(f"Saved: {folder}/mask_{i+1}.png")


def load_mask_from_image(filepath):
    """Load mask from PNG image. [0,255] → [0,1]."""
    from PIL import Image
    img = Image.open(filepath).convert('L')
    return torch.from_numpy(np.array(img)).float() / 255.0


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_confusion_matrix(model, x_test, y_test, device='cpu', batch_size=100):
    """Run inference on full test set, return 10×10 confusion matrix."""
    model.eval()
    num_classes = 10
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    num_samples = x_test.shape[0]
    print(f"  Running inference on {num_samples} test images...")
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = x_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu()
            
            for true, pred in zip(batch_y, preds):
                cm[true, pred] += 1
            
            if (i // batch_size) % 20 == 0:
                print(f"    Processed {min(i+batch_size, num_samples)}/{num_samples}")
    
    return cm


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix as annotated heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_numpy = cm.numpy()
    im = ax.imshow(cm_numpy, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')
    
    # Labels
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Annotate cells
    thresh = cm_numpy.max() / 2
    for i in range(10):
        for j in range(10):
            val = cm_numpy[i, j]
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center', color=color, fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def compute_per_class_accuracy(cm):
    """From confusion matrix, compute per-class accuracy."""
    per_class = {}
    for i in range(10):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        per_class[i] = 100.0 * correct / total if total > 0 else 0.0
    return per_class


def get_binary_patterns(model, x_test, y_test, device='cpu', batch_size=100):
    """For binary mode: collect binary patterns produced for each digit."""
    model.eval()
    detector = model.detector
    extractor = detector.extractor
    
    # Storage: patterns_per_digit[digit] = list of binary pattern tuples
    patterns_per_digit = {i: [] for i in range(10)}
    
    num_samples = x_test.shape[0]
    print(f"  Collecting binary patterns from {num_samples} images...")
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = x_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            # Forward through optical layers (replicate model.forward logic)
            if batch_x.dim() == 4:
                batch_x = batch_x.squeeze(1)
            field = batch_x.to(torch.complex64)
            
            for layer_idx, layer in enumerate(model.layers):
                amplitude = layer.get_amplitude(
                    binary=model.is_binary,
                    binary_method='hard',  # Use hard threshold for inference
                    binary_sharpness=10.0,
                    min_transmission=model.min_transmission if model.min_transmission > 0 else None,
                    max_transmission=model.max_transmission if model.max_transmission < 1 else None
                )
                field = field * amplitude
                field = propagate(field, model.transfer_functions[layer_idx])
            
            intensity = torch.abs(field) ** 2
            
            # Extract zone intensities (use .extract() method)
            zone_intensities = extractor.extract(intensity)  # [batch, num_zones]
            
            # Normalize per-sample (same as model's _soft_binary)
            z_min = zone_intensities.min(dim=1, keepdim=True)[0]
            z_max = zone_intensities.max(dim=1, keepdim=True)[0]
            z_norm = (zone_intensities - z_min) / (z_max - z_min + 1e-8)
            
            # Apply threshold to get binary patterns
            threshold = detector.threshold
            binary = (z_norm > threshold).int().cpu()  # [batch, num_zones]
            
            for label, pattern in zip(batch_y, binary):
                patterns_per_digit[label.item()].append(tuple(pattern.tolist()))
    
    return patterns_per_digit


def analyze_binary_patterns(patterns_per_digit):
    """Find most common pattern and distribution for each digit."""
    from collections import Counter
    
    analysis = {}
    for digit in range(10):
        patterns = patterns_per_digit[digit]
        counter = Counter(patterns)
        most_common = counter.most_common(10)  # Top 10 patterns
        
        analysis[digit] = {
            'total': len(patterns),
            'unique': len(counter),
            'most_common': most_common,
            'top_pattern': most_common[0][0] if most_common else None,
            'top_count': most_common[0][1] if most_common else 0,
            'counter': counter,  # Store full counter for overlap analysis
        }
    return analysis


def compute_pattern_overlap(analysis):
    """Compute which digits share the same patterns."""
    # Find all patterns that appear for multiple digits
    pattern_to_digits = {}
    for digit in range(10):
        for pattern, count in analysis[digit]['counter'].items():  # use full counter, not just top-10
            if pattern not in pattern_to_digits:
                pattern_to_digits[pattern] = {}
            pattern_to_digits[pattern][digit] = count
    
    # Find patterns shared by multiple digits
    shared_patterns = {p: d for p, d in pattern_to_digits.items() if len(d) > 1}
    return shared_patterns


def print_pattern_overlap(analysis):
    """Print detailed pattern overlap analysis."""
    shared = compute_pattern_overlap(analysis)
    
    print(f"\n{'='*50}")
    print("PATTERN OVERLAP ANALYSIS")
    print(f"{'='*50}")
    
    # Sort by number of digits sharing the pattern
    sorted_shared = sorted(shared.items(), key=lambda x: -len(x[1]))
    
    for pattern, digit_counts in sorted_shared[:10]:  # Top 10 shared
        pattern_str = ''.join(str(b) for b in pattern)
        digits_str = ', '.join(f"{d}({c})" for d, c in sorted(digit_counts.items()))
        print(f"  [{pattern_str}] shared by: {digits_str}")
    
    # Print confusion pairs (digits that share many patterns)
    print(f"\nMost confused digit pairs (by shared patterns):")
    pair_overlap = {}
    for pattern, digit_counts in shared.items():
        digits = list(digit_counts.keys())
        for i, d1 in enumerate(digits):
            for d2 in digits[i+1:]:
                pair = (min(d1, d2), max(d1, d2))
                if pair not in pair_overlap:
                    pair_overlap[pair] = 0
                pair_overlap[pair] += min(digit_counts[d1], digit_counts[d2])
    
    sorted_pairs = sorted(pair_overlap.items(), key=lambda x: -x[1])
    for (d1, d2), overlap in sorted_pairs[:5]:
        print(f"  Digits {d1} & {d2}: {overlap} overlapping samples")


def get_zone_intensities(model, x_test, y_test, device='cpu', batch_size=100):
    """Collect zone intensities for all test samples (works for all detection modes)."""
    model.eval()
    
    # Get extractor from detector (all modes have this)
    extractor = model.detector.extractor
    num_zones = extractor.zones.shape[0]
    
    # Storage: zone_intensities_per_digit[digit] = list of zone intensity vectors
    zone_intensities_per_digit = {i: [] for i in range(10)}
    
    num_samples = x_test.shape[0]
    print(f"  Collecting zone intensities from {num_samples} images...")
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = x_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            # Forward through optical layers
            if batch_x.dim() == 4:
                batch_x = batch_x.squeeze(1)
            field = batch_x.to(torch.complex64)
            
            for layer_idx, layer in enumerate(model.layers):
                amplitude = layer.get_amplitude(
                    binary=model.is_binary,
                    binary_method='hard',
                    binary_sharpness=10.0,
                    min_transmission=model.min_transmission if model.min_transmission > 0 else None,
                    max_transmission=model.max_transmission if model.max_transmission < 1 else None
                )
                field = field * amplitude
                field = propagate(field, model.transfer_functions[layer_idx])
            
            intensity = torch.abs(field) ** 2
            
            # Extract zone intensities (use .extract() method)
            zone_intensities = extractor.extract(intensity).cpu()  # [batch, num_zones]
            
            for label, zones in zip(batch_y, zone_intensities):
                zone_intensities_per_digit[label.item()].append(zones.numpy())
    
    return zone_intensities_per_digit, num_zones


def compute_avg_zone_intensities(zone_intensities_per_digit):
    """Compute average and std zone intensities per digit."""
    avg_intensities = {}
    std_intensities = {}
    
    for digit in range(10):
        intensities = np.array(zone_intensities_per_digit[digit])  # [num_samples, num_zones]
        avg_intensities[digit] = intensities.mean(axis=0)
        std_intensities[digit] = intensities.std(axis=0)
    
    return avg_intensities, std_intensities


def plot_zone_intensity_heatmap(avg_intensities, grid_size, save_path=None):
    """Plot average zone intensity per digit as heatmap grids."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    
    # Find global min/max for consistent colormap
    all_vals = np.concatenate([avg_intensities[d] for d in range(10)])
    vmin, vmax = all_vals.min(), all_vals.max()
    
    for digit in range(10):
        ax = axes[digit]
        intensities = avg_intensities[digit]
        
        # Reshape to grid
        grid = intensities.reshape(grid_size, grid_size)
        
        im = ax.imshow(grid, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
        
        # Add grid lines
        for x in range(grid_size + 1):
            ax.axhline(x - 0.5, color='white', linewidth=0.5, alpha=0.5)
            ax.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.5)
        
        # Add intensity values (auto-adjust decimal places based on magnitude)
        for r in range(grid_size):
            for c in range(grid_size):
                val = grid[r, c]
                color = 'white' if val < (vmax + vmin) / 2 else 'black'
                # Use scientific notation for very small values, otherwise auto decimal places
                if vmax < 0.01:
                    label = f'{val:.1e}'
                elif vmax < 0.1:
                    label = f'{val:.3f}'
                else:
                    label = f'{val:.2f}'
                ax.text(c, r, label, ha='center', va='center', 
                       fontsize=7 if vmax < 0.01 else 8, color=color)
        
        ax.set_title(f'Digit {digit}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Avg Intensity')
    
    plt.suptitle('Average Zone Intensity per Digit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_zone_intensity_comparison(avg_intensities, std_intensities, grid_size, save_path=None):
    """Plot zone intensities as bar chart comparison across digits."""
    num_zones = grid_size * grid_size
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(num_zones)
    width = 0.08
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        offset = (digit - 4.5) * width
        ax.bar(x + offset, avg_intensities[digit], width, 
               label=f'{digit}', color=colors[digit], alpha=0.8)
    
    ax.set_xlabel('Zone Index', fontsize=12)
    ax.set_ylabel('Average Intensity', fontsize=12)
    ax.set_title('Zone Intensity Distribution by Digit', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Z{i}' for i in range(num_zones)])
    ax.legend(title='Digit', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_binary_pattern_summary(analysis, grid_size=3, save_path=None):
    """Show most common pattern per digit as grid visualization."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    
    for digit in range(10):
        ax = axes[digit]
        info = analysis[digit]
        pattern = info['top_pattern']
        
        if pattern is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Digit {digit}')
            ax.axis('off')
            continue
        
        # Reshape pattern to grid
        pattern_grid = np.array(pattern).reshape(grid_size, grid_size)
        
        # Plot as colored squares
        ax.imshow(pattern_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        
        # Add grid lines
        for x in range(grid_size + 1):
            ax.axhline(x - 0.5, color='black', linewidth=2)
            ax.axvline(x - 0.5, color='black', linewidth=2)
        
        # Add 0/1 text in each cell
        for r in range(grid_size):
            for c in range(grid_size):
                val = int(pattern_grid[r, c])
                color = 'white' if val == 1 else 'black'
                ax.text(c, r, str(val), ha='center', va='center', 
                       fontsize=14, fontweight='bold', color=color)
        
        coverage = 100.0 * info['top_count'] / info['total'] if info['total'] > 0 else 0
        ax.set_title(f"Digit {digit}\n{info['top_count']}/{info['total']} ({coverage:.0f}%)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Most Common Binary Pattern per Digit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_pattern_to_logits(model, save_path=None):
    """Visualize the learned pattern→class mapping weights."""
    if not hasattr(model.detector, 'pattern_to_logits'):
        print("  (Not binary mode, skipping pattern_to_logits visualization)")
        return
    
    weights = model.detector.pattern_to_logits.detach().cpu().numpy()  # [num_patterns, 10]
    num_patterns, num_classes = weights.shape
    
    # Only show patterns with strong activation (top patterns per class)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full heatmap (downsampled if too large)
    ax1 = axes[0]
    if num_patterns > 64:
        # Show every Nth pattern
        step = num_patterns // 64
        weights_down = weights[::step, :]
        im1 = ax1.imshow(weights_down.T, cmap='RdBu_r', aspect='auto')
        ax1.set_xlabel(f'Pattern Index (every {step}th)')
    else:
        im1 = ax1.imshow(weights.T, cmap='RdBu_r', aspect='auto')
        ax1.set_xlabel('Pattern Index')
    ax1.set_ylabel('Class')
    ax1.set_yticks(range(10))
    ax1.set_title('Pattern → Logits Weights')
    plt.colorbar(im1, ax=ax1)
    
    # Right: Top-3 patterns per class
    ax2 = axes[1]
    top_n = 3
    top_patterns_text = []
    for cls in range(10):
        top_idx = np.argsort(weights[:, cls])[-top_n:][::-1]
        top_patterns_text.append(f"Class {cls}: patterns {list(top_idx)}")
    
    ax2.text(0.1, 0.5, '\n'.join(top_patterns_text), transform=ax2.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    ax2.set_title('Top-3 Patterns per Class')
    ax2.axis('off')
    
    plt.suptitle(f'Binary Encoding: {num_patterns} Patterns → 10 Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_pattern_digit_mapping(model, grid_size=3, save_path=None):
    """Show which binary patterns the network learned to associate with each digit.
    
    This shows the LEARNED WEIGHTS - which patterns have highest weight for each class.
    These are the patterns the network "wants" to see for each digit.
    """
    if not hasattr(model.detector, 'pattern_to_logits'):
        print("  (Not binary mode, skipping pattern mapping visualization)")
        return
    
    weights = model.detector.pattern_to_logits.detach().cpu().numpy()  # [num_patterns, 10]
    num_zones = grid_size * grid_size
    
    # For each digit, find top-3 patterns (highest weight)
    top_n = 3
    
    # Horizontal layout: 10 columns (digits) × 4 rows (label + 3 patterns)
    fig, axes = plt.subplots(top_n + 1, 10, figsize=(24, 10))
    
    for digit in range(10):
        # Get top patterns for this digit
        pattern_weights = weights[:, digit]
        top_indices = np.argsort(pattern_weights)[-top_n:][::-1]
        
        # First row: digit label
        ax_label = axes[0, digit]
        ax_label.text(0.5, 0.5, f'{digit}', ha='center', va='center', 
                     fontsize=24, fontweight='bold')
        ax_label.axis('off')
        ax_label.set_facecolor('#e0e0e0')
        
        # Next rows: top patterns
        for rank, pattern_idx in enumerate(top_indices):
            ax = axes[rank + 1, digit]
            
            # Convert pattern index to binary pattern (LSB = zone 0)
            pattern = []
            idx = pattern_idx
            for _ in range(num_zones):
                pattern.append(idx % 2)
                idx //= 2
            # No reversal: zone 0 (LSB) first, matching model's (i >> j) & 1
            
            # Reshape to grid
            pattern_grid = np.array(pattern).reshape(grid_size, grid_size)
            
            # Plot
            ax.imshow(pattern_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
            
            # Add grid lines
            for x in range(grid_size + 1):
                ax.axhline(x - 0.5, color='black', linewidth=1)
                ax.axvline(x - 0.5, color='black', linewidth=1)
            
            # Add 0/1 text
            for r in range(grid_size):
                for c in range(grid_size):
                    val = int(pattern_grid[r, c])
                    color = 'white' if val == 1 else 'black'
                    ax.text(c, r, str(val), ha='center', va='center',
                           fontsize=10, fontweight='bold', color=color)
            
            # Title with weight
            weight = pattern_weights[pattern_idx]
            ax.set_title(f'W={weight:.1f}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Row labels on the left
    axes[0, 0].set_ylabel('Digit', fontsize=12, rotation=0, ha='right', va='center')
    for i in range(top_n):
        axes[i + 1, 0].set_ylabel(f'#{i+1}', fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle('LEARNED Pattern→Digit Mapping\n(Patterns with highest weights for each digit - what the network "prefers")', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_observed_pattern_mapping(binary_analysis, grid_size=3, save_path=None):
    """Show actual observed patterns per digit from test data.
    
    This shows the ACTUAL OBSERVATIONS - which patterns each digit actually produces.
    These are the patterns that appear when real digit images go through the ONN.
    """
    top_n = 3
    
    # Horizontal layout: 10 columns (digits) × 4 rows (label + 3 patterns)
    fig, axes = plt.subplots(top_n + 1, 10, figsize=(24, 10))
    
    for digit in range(10):
        info = binary_analysis[digit]
        
        # First row: digit label with stats
        ax_label = axes[0, digit]
        ax_label.text(0.5, 0.6, f'{digit}', ha='center', va='center',
                     fontsize=24, fontweight='bold')
        ax_label.text(0.5, 0.2, f'{info["unique"]} uniq', 
                     ha='center', va='center', fontsize=8, color='gray')
        ax_label.axis('off')
        ax_label.set_facecolor('#e0e0e0')
        
        # Next rows: top observed patterns
        for rank, (pattern, count) in enumerate(info['most_common'][:top_n]):
            ax = axes[rank + 1, digit]
            
            # Reshape to grid
            pattern_grid = np.array(pattern).reshape(grid_size, grid_size)
            
            # Plot
            ax.imshow(pattern_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
            
            # Add grid lines
            for x in range(grid_size + 1):
                ax.axhline(x - 0.5, color='black', linewidth=1)
                ax.axvline(x - 0.5, color='black', linewidth=1)
            
            # Add 0/1 text
            for r in range(grid_size):
                for c in range(grid_size):
                    val = int(pattern_grid[r, c])
                    color = 'white' if val == 1 else 'black'
                    ax.text(c, r, str(val), ha='center', va='center',
                           fontsize=10, fontweight='bold', color=color)
            
            # Title with count
            coverage = 100.0 * count / info['total'] if info['total'] > 0 else 0
            ax.set_title(f'{coverage:.0f}%', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Row labels on the left
    axes[0, 0].set_ylabel('Digit', fontsize=12, rotation=0, ha='right', va='center')
    for i in range(top_n):
        axes[i + 1, 0].set_ylabel(f'#{i+1}', fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle('OBSERVED Pattern→Digit Mapping\n(Patterns that actually appear for each digit in test data)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def get_pattern_assignment(model, grid_size=3):
    """Assign each of the 512 patterns to its best digit based on learned weights.
    
    Returns dict: {digit: [(pattern_tuple, weight), ...]} sorted by weight descending.
    """
    if not hasattr(model.detector, 'pattern_to_logits'):
        return None
    
    weights = model.detector.pattern_to_logits.detach().cpu().numpy()  # [512, 10]
    num_patterns, num_classes = weights.shape
    num_zones = grid_size * grid_size
    
    # For each pattern, find which digit it maps to (argmax)
    assigned_digit = np.argmax(weights, axis=1)  # [512]
    
    # Group patterns by assigned digit
    assignment = {d: [] for d in range(10)}
    for pattern_idx in range(num_patterns):
        digit = assigned_digit[pattern_idx]
        weight = weights[pattern_idx, digit]
        
        # Convert index to binary pattern tuple (LSB = zone 0)
        bits = []
        idx = pattern_idx
        for _ in range(num_zones):
            bits.append(idx % 2)
            idx //= 2
        bits = tuple(bits)  # No reversal: zone 0 (LSB) first, matching model
        
        assignment[digit].append((bits, weight))
    
    # Sort each digit's patterns by weight (highest first)
    for d in range(10):
        assignment[d].sort(key=lambda x: -x[1])
    
    return assignment


def print_pattern_assignment(assignment, grid_size=3):
    """Print full list of which patterns belong to which digit."""
    print(f"\n{'='*60}")
    print("PATTERN → DIGIT ASSIGNMENT (all 512 patterns)")
    print(f"{'='*60}")
    print("Each pattern is assigned to the digit with its highest weight.")
    
    for digit in range(10):
        patterns = assignment[digit]
        print(f"\n  Digit {digit}: {len(patterns)} patterns assigned")
        # Show top 5 with full detail
        for i, (bits, weight) in enumerate(patterns[:5]):
            bit_str = ''.join(str(b) for b in bits)
            print(f"    [{bit_str}]  W={weight:+.2f}")
        if len(patterns) > 5:
            print(f"    ... and {len(patterns) - 5} more")


def save_pattern_assignment_txt(assignment, grid_size=3, save_path=None):
    """Save full pattern assignment list to text file."""
    if save_path is None:
        return
    
    with open(save_path, 'w') as f:
        f.write("PATTERN → DIGIT ASSIGNMENT\n")
        f.write("="*60 + "\n")
        f.write("Each of the 512 binary patterns is assigned to the digit\n")
        f.write("with the highest learned weight.\n\n")
        
        total_assigned = 0
        for digit in range(10):
            patterns = assignment[digit]
            total_assigned += len(patterns)
            f.write(f"Digit {digit}: {len(patterns)} patterns\n")
            f.write("-" * 40 + "\n")
            for bits, weight in patterns:
                bit_str = ''.join(str(b) for b in bits)
                f.write(f"  [{bit_str}]  W={weight:+.3f}\n")
            f.write("\n")
        
        f.write(f"Total: {total_assigned} patterns assigned across 10 digits\n")
    print(f"  Saved: {save_path}")


def plot_pattern_assignment(assignment, grid_size=3, save_path=None):
    """Visualize all patterns assigned to each digit (top 6 per digit as grids)."""
    top_n = 6  # Show top 6 patterns per digit
    
    fig, axes = plt.subplots(top_n + 1, 10, figsize=(22, 15))
    
    for digit in range(10):
        patterns = assignment[digit]
        
        # Header row: digit label + count
        ax_label = axes[0, digit]
        ax_label.text(0.5, 0.55, f'{digit}', ha='center', va='center',
                     fontsize=22, fontweight='bold')
        ax_label.text(0.5, 0.15, f'{len(patterns)} patterns', 
                     ha='center', va='center', fontsize=8, color='gray')
        ax_label.axis('off')
        ax_label.set_facecolor('#e0e0e0')
        
        # Pattern rows
        for rank in range(top_n):
            ax = axes[rank + 1, digit]
            
            if rank < len(patterns):
                bits, weight = patterns[rank]
                pattern_grid = np.array(bits).reshape(grid_size, grid_size)
                
                ax.imshow(pattern_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
                
                # Grid lines
                for x in range(grid_size + 1):
                    ax.axhline(x - 0.5, color='black', linewidth=1)
                    ax.axvline(x - 0.5, color='black', linewidth=1)
                
                # 0/1 text
                for r in range(grid_size):
                    for c in range(grid_size):
                        val = int(pattern_grid[r, c])
                        color = 'white' if val == 1 else 'black'
                        ax.text(c, r, str(val), ha='center', va='center',
                               fontsize=9, fontweight='bold', color=color)
                
                ax.set_title(f'W={weight:+.1f}', fontsize=7)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Row labels
    axes[0, 0].set_ylabel('Digit', fontsize=11, rotation=0, ha='right', va='center')
    for i in range(top_n):
        axes[i + 1, 0].set_ylabel(f'#{i+1}', fontsize=9, rotation=0, ha='right', va='center')
    
    plt.suptitle('Pattern → Digit Assignment (Top 6 per digit, sorted by weight)\n'
                 'Each pattern belongs to the digit where it has the highest learned weight',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def print_analysis_summary(cm, per_class, binary_analysis=None):
    """Print text summary of analysis results."""
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    overall_acc = 100.0 * correct / total
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {correct}/{total} = {overall_acc:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for digit, acc in per_class.items():
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  Digit {digit}: {bar} {acc:5.1f}%")
    
    if binary_analysis:
        print(f"\nBinary Patterns (top 5 per digit):")
        for digit in range(10):
            info = binary_analysis[digit]
            print(f"\n  Digit {digit}: ({info['unique']} unique patterns)")
            for i, (pattern, count) in enumerate(info['most_common'][:5]):
                coverage = 100.0 * count / info['total'] if info['total'] > 0 else 0
                pattern_str = ''.join(str(b) for b in pattern)
                marker = "  ★" if i == 0 else "   "
                print(f"   {marker} [{pattern_str}] {coverage:5.1f}% ({count} samples)")


# =============================================================================
# MAXZONE ANALYSIS FUNCTIONS
# =============================================================================

def print_zone_assignment(zone_assignment, grid_size):
    """Print which digit each zone is assigned to."""
    print(f"\n{'='*50}")
    print(f"ZONE → DIGIT ASSIGNMENT ({grid_size}×{grid_size} = {grid_size**2} zones)")
    print(f"{'='*50}")
    
    # Count zones per digit
    digit_zones = {}
    for zone_idx, (digit, weight) in zone_assignment.items():
        if digit not in digit_zones:
            digit_zones[digit] = []
        digit_zones[digit].append((zone_idx, weight))
    
    # Print grid layout
    print(f"\n  Grid layout (zone → digit):")
    for r in range(grid_size):
        row_str = "  "
        for c in range(grid_size):
            z = r * grid_size + c
            digit, weight = zone_assignment[z]
            row_str += f"  [{z:2d}→{digit}]"
        print(row_str)
    
    # Print per-digit summary
    print(f"\n  Per-digit zone count:")
    for digit in range(10):
        zones = digit_zones.get(digit, [])
        zone_str = ', '.join(f'Z{z}(W={w:.1f})' for z, w in zones)
        print(f"    Digit {digit}: {len(zones)} zones — {zone_str if zones else '(none)'}")


def plot_zone_assignment_map(zone_assignment, grid_size, save_path=None):
    """Visualize 5×5 zone grid colored by assigned digit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    num_zones = grid_size * grid_size
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Left: Zone assignment grid (colored by digit)
    ax = axes[0]
    grid = np.zeros((grid_size, grid_size))
    for z, (digit, weight) in zone_assignment.items():
        r, c = z // grid_size, z % grid_size
        grid[r, c] = digit
    
    # Custom colormap for digits 0-9
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors[:10])
    bounds = np.arange(-0.5, 10.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')
    
    # Add grid lines
    for x in range(grid_size + 1):
        ax.axhline(x - 0.5, color='black', linewidth=2)
        ax.axvline(x - 0.5, color='black', linewidth=2)
    
    # Add zone labels
    for r in range(grid_size):
        for c in range(grid_size):
            z = r * grid_size + c
            digit, weight = zone_assignment[z]
            ax.text(c, r - 0.15, f'Z{z}', ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
            ax.text(c, r + 0.2, f'{digit}', ha='center', va='center',
                   fontsize=18, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.4))
    
    ax.set_title(f'Zone → Digit Assignment\n(find brightest zone, read its digit)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=range(10), shrink=0.8)
    cbar.set_label('Assigned Digit')
    
    # Right: Weight strength per zone
    ax2 = axes[1]
    weight_grid = np.zeros((grid_size, grid_size))
    for z, (digit, weight) in zone_assignment.items():
        r, c = z // grid_size, z % grid_size
        weight_grid[r, c] = weight
    
    im2 = ax2.imshow(weight_grid, cmap='YlOrRd', aspect='equal')
    
    for x in range(grid_size + 1):
        ax2.axhline(x - 0.5, color='black', linewidth=2)
        ax2.axvline(x - 0.5, color='black', linewidth=2)
    
    for r in range(grid_size):
        for c in range(grid_size):
            z = r * grid_size + c
            digit, weight = zone_assignment[z]
            color = 'white' if weight > (weight_grid.max() + weight_grid.min()) / 2 else 'black'
            ax2.text(c, r, f'{digit}\nW={weight:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)
    
    ax2.set_title('Zone Confidence\n(higher = more committed to digit)', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, shrink=0.8, label='Weight')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def analyze_maxzone_hits(model, x_test, y_test, device='cpu', batch_size=100):
    """For maxzone: count which zone is brightest for each digit in test data."""
    model.eval()
    extractor = model.detector.extractor
    num_zones = extractor.num_zones
    
    # hits[digit][zone] = count of times that zone was brightest for this digit
    hits = {d: np.zeros(num_zones) for d in range(10)}
    
    with torch.no_grad():
        for i in range(0, x_test.shape[0], batch_size):
            batch_x = x_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            if batch_x.dim() == 4:
                batch_x = batch_x.squeeze(1)
            field = batch_x.to(torch.complex64)
            
            for layer_idx, layer in enumerate(model.layers):
                amplitude = layer.get_amplitude(
                    binary=model.is_binary,
                    binary_method='hard',
                    binary_sharpness=10.0,
                    min_transmission=model.min_transmission if model.min_transmission > 0 else None,
                    max_transmission=model.max_transmission if model.max_transmission < 1 else None
                )
                field = field * amplitude
                field = propagate(field, model.transfer_functions[layer_idx])
            
            intensity = torch.abs(field) ** 2
            zone_intensities = extractor.extract(intensity)
            brightest = zone_intensities.argmax(dim=1).cpu().numpy()
            
            for label, zone in zip(batch_y.numpy(), brightest):
                hits[label][zone] += 1
    
    return hits


def print_maxzone_hit_summary(hits, zone_assignment, grid_size):
    """Print how often each digit's brightest zone matches its assigned zone."""
    print(f"\n{'='*50}")
    print(f"MAXZONE HIT ANALYSIS")
    print(f"{'='*50}")
    
    # Build reverse map: digit → list of zones assigned to it
    digit_to_zones = {}
    for z, (digit, _) in zone_assignment.items():
        if digit not in digit_to_zones:
            digit_to_zones[digit] = []
        digit_to_zones[digit].append(z)
    
    for digit in range(10):
        total = int(hits[digit].sum())
        if total == 0:
            continue
        
        # Top 3 zones this digit activates
        top_zones = np.argsort(hits[digit])[::-1][:3]
        assigned_zones = digit_to_zones.get(digit, [])
        
        # Count hits on assigned zones
        assigned_hits = sum(int(hits[digit][z]) for z in assigned_zones)
        assigned_pct = 100.0 * assigned_hits / total
        
        top_str = ', '.join(f'Z{z}({int(hits[digit][z])})' for z in top_zones)
        print(f"  Digit {digit}: {assigned_pct:5.1f}% on assigned zones | Top: {top_str}")


def save_zone_assignment_txt(zone_assignment, grid_size, save_path=None):
    """Save zone assignment lookup table to text file."""
    if save_path is None:
        return
    
    with open(save_path, 'w') as f:
        f.write("MAXZONE ASSIGNMENT LOOKUP TABLE\n")
        f.write("=" * 40 + "\n")
        f.write("Find the brightest zone, look up its digit.\n\n")
        
        f.write(f"Grid layout ({grid_size}×{grid_size}):\n")
        for r in range(grid_size):
            row_str = "  "
            for c in range(grid_size):
                z = r * grid_size + c
                digit, weight = zone_assignment[z]
                row_str += f"  [{z:2d}→{digit}]"
            f.write(row_str + "\n")
        
        f.write(f"\nDetailed assignment:\n")
        for z in range(grid_size * grid_size):
            digit, weight = zone_assignment[z]
            f.write(f"  Zone {z:2d} → Digit {digit}  (W={weight:+.3f})\n")
    
    print(f"  Saved: {save_path}")


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

def forward_pass(image, masks, layer_spacings, 
                 use_fill_factor=False, use_contrast_limits=False, use_binary=False):
    """Propagate light through LCD layers. Returns list of stage results.
    
    Args:
        layer_spacings: list of distances (one per layer), or single float for uniform spacing.
    """
    # Support single float for backward compatibility
    if isinstance(layer_spacings, (int, float)):
        layer_spacings = [layer_spacings] * len(masks)
    
    # Precompute transfer functions per layer
    H_list = [create_transfer_function(N, WAVELENGTH, PIXEL_PITCH, d) for d in layer_spacings]
    
    field = image.to(torch.complex64)
    results = [{'name': 'Input', 'intensity': torch.abs(field)**2, 'field': field.clone(), 'mask': None}]
    
    for i, mask in enumerate(masks):
        effective_mask = mask.clone()
        if use_binary:
            effective_mask = apply_binary_quantization(effective_mask, method='hard')
        if use_contrast_limits:
            effective_mask = apply_contrast_limits(effective_mask, MIN_TRANSMISSION, MAX_TRANSMISSION)
        
        field = field * effective_mask
        if use_fill_factor and FILL_FACTOR < 1.0:
            field = apply_fill_factor(field, fill_factor=FILL_FACTOR)
        
        results.append({'name': f'After Mask {i+1}', 'intensity': torch.abs(field)**2, 
                       'field': field.clone(), 'mask': effective_mask})
        
        field = propagate(field, H_list[i])
        results.append({'name': f'After Prop {i+1}', 'intensity': torch.abs(field)**2,
                       'field': field.clone(), 'mask': None})
    
    results[-1]['name'] = 'Camera'
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_zone_grid(ax, intensity, grid_size=3):
    """Draw zone grid overlay on intensity image with zone labels."""
    h, w = intensity.shape
    zone_h, zone_w = h // grid_size, w // grid_size
    
    # Draw the intensity image
    ax.imshow(intensity.numpy(), cmap='hot')
    
    # Draw grid lines
    for i in range(1, grid_size):
        ax.axhline(i * zone_h, color='cyan', linewidth=1.5, alpha=0.8)
        ax.axvline(i * zone_w, color='cyan', linewidth=1.5, alpha=0.8)
    
    # Draw border
    ax.axhline(0, color='cyan', linewidth=2)
    ax.axhline(h, color='cyan', linewidth=2)
    ax.axvline(0, color='cyan', linewidth=2)
    ax.axvline(w, color='cyan', linewidth=2)
    
    # Add zone labels and intensity values
    for row in range(grid_size):
        for col in range(grid_size):
            zone_idx = row * grid_size + col
            # Calculate zone intensity
            y1, y2 = row * zone_h, (row + 1) * zone_h
            x1, x2 = col * zone_w, (col + 1) * zone_w
            zone_intensity = intensity[y1:y2, x1:x2].mean().item()
            
            # Position for label
            cy, cx = (row + 0.5) * zone_h, (col + 0.5) * zone_w
            
            # Zone label
            ax.text(cx, cy - zone_h * 0.15, f'Z{zone_idx}', 
                   ha='center', va='center', color='cyan', fontsize=8, fontweight='bold')
            # Intensity value
            ax.text(cx, cy + zone_h * 0.15, f'{zone_intensity:.3f}', 
                   ha='center', va='center', color='white', fontsize=7)


def visualize(image, masks, layer_spacings, digit_label=None, save_path=None,
              use_fill_factor=False, use_contrast_limits=False, use_binary=False,
              grid_size=3):
    """Generate 3-row plot: intensity, phase, mask/zones."""
    results = forward_pass(image, masks, layer_spacings, use_fill_factor, use_contrast_limits, use_binary)
    num_stages = len(results)
    
    fig, axes = plt.subplots(3, num_stages, figsize=(3 * num_stages, 9))
    
    # Track intensity range for consistent colormap
    all_intensities = [stage['intensity'] for stage in results]
    vmax_intensity = max(i.max().item() for i in all_intensities)
    
    for col, stage in enumerate(results):
        intensity, field, mask = stage['intensity'], stage['field'], stage['mask']
        is_camera = (stage['name'] == 'Camera')
        is_input = (stage['name'] == 'Input')
        
        # Row 0: Intensity
        axes[0, col].imshow(intensity.numpy(), cmap='hot', vmin=0, vmax=vmax_intensity)
        axes[0, col].set_title(stage['name'], fontsize=10)
        axes[0, col].axis('off')
        
        # Row 1: Phase
        im_phase = axes[1, col].imshow(torch.angle(field).numpy(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[1, col].axis('off')
        
        # Row 2: Mask (for mask stages), Zone grid (for camera), or Input image (for input)
        if mask is not None:
            # Show the LCD mask
            axes[2, col].imshow(mask.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2, col].set_title('LCD Mask', fontsize=8)
        elif is_camera:
            # Show zone grid overlay for camera
            draw_zone_grid(axes[2, col], intensity, grid_size=grid_size)
            axes[2, col].set_title(f'Detector Zones ({grid_size}×{grid_size})', fontsize=8)
        elif is_input:
            # Show input image info
            axes[2, col].imshow(intensity.numpy(), cmap='gray')
            axes[2, col].set_title(f'Input ({intensity.shape[0]}×{intensity.shape[1]})', fontsize=8)
        else:
            # For propagation stages, show intensity with different colormap
            axes[2, col].imshow(intensity.numpy(), cmap='viridis')
            axes[2, col].set_title('Propagated', fontsize=8)
        axes[2, col].axis('off')
    
    # Row labels with symbols
    axes[0, 0].set_ylabel('Intensity |E|²', fontsize=11)
    axes[1, 0].set_ylabel('Phase ∠E', fontsize=11)
    axes[2, 0].set_ylabel('Mask / Zones', fontsize=11)
    
    # Add phase colorbar
    cbar_ax = fig.add_axes([0.92, 0.38, 0.015, 0.24])
    cbar = fig.colorbar(im_phase, cax=cbar_ax)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(['-π', '0', '+π'])
    cbar.ax.tick_params(labelsize=8)
    
    # Title
    title = f'{len(masks)}-Layer ONN'
    if digit_label is not None:
        title += f' | Digit: {digit_label}'
    if isinstance(layer_spacings, (int, float)):
        title += f' | λ=532nm | Gap: {layer_spacings*100:.0f}cm'
    else:
        title += f' | λ=532nm | Gaps: {[f"{d*100:.0f}cm" for d in layer_spacings]}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # ===================
    # OPTIONS
    # ===================
    MASK_SOURCE = 'checkpoints_maxzone_5x5/best.pt'  # 'random', 'smooth', or checkpoint path
    USE_FILL_FACTOR = True
    USE_CONTRAST_LIMITS = True
    USE_BINARY = IS_BINARY  # Match training config (config.lcd.is_binary)
    
    # Analysis options
    RUN_ANALYSIS = True         # Run full test set analysis (confusion matrix, etc.)
    VISUALIZE_DIGITS = True     # Generate propagation visualizations for digits 0-9
    
    # ===================
    # SETUP
    # ===================
    SAVE_DIR = f'visualizations_{MASK_SOURCE}' if MASK_SOURCE in ['random', 'smooth'] else str(Path(MASK_SOURCE).parent / 'visualizations')
    
    NUM_LAYERS = config.multilayer.num_layers
    LAYER_SPACINGS = config.multilayer.layer_spacings  # Per-layer distances
    
    print("=" * 60)
    print("ONN VISUALIZATION & ANALYSIS")
    print("=" * 60)
    print(f"Config: {N}×{N}, λ={WAVELENGTH*1e9:.0f}nm, {NUM_LAYERS} layers, gaps={[f'{d*100:.1f}cm' for d in LAYER_SPACINGS]}")
    print(f"LCD: fill={FILL_FACTOR}, contrast=[{MIN_TRANSMISSION:.2f},{MAX_TRANSMISSION:.2f}], binary={USE_BINARY}")
    
    # Load masks
    if MASK_SOURCE == 'random':
        masks = create_random_masks(NUM_LAYERS)
    elif MASK_SOURCE == 'smooth':
        masks = create_smooth_masks(NUM_LAYERS)
    else:
        masks = load_masks(MASK_SOURCE)
        NUM_LAYERS = len(masks)
    
    # Load model for prediction
    model = load_model(MASK_SOURCE) if MASK_SOURCE not in ['random', 'smooth'] else None
    detection_method = None
    grid_size = 3  # Default grid size for zone visualization
    
    if model and hasattr(model.detector, 'zone_logits'):
        detection_method = 'maxzone'
    elif model and hasattr(model.detector, 'pattern_to_logits'):
        detection_method = 'binary'
    elif model and hasattr(model.detector, 'classifier'):
        detection_method = 'center'
    else:
        detection_method = 'zone'
    
    # Infer grid size from model's zone extractor
    if model and hasattr(model.detector, 'extractor'):
        num_zones = model.detector.extractor.zones.shape[0]
        grid_size = int(np.sqrt(num_zones))
    
    # Load data
    print("\nLoading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist_resized()
    
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_masks_as_images(masks, folder=f'{SAVE_DIR}/masks')
    
    # ===================
    # FULL TEST SET ANALYSIS
    # ===================
    if RUN_ANALYSIS and model:
        print("\n" + "=" * 60)
        print("FULL TEST SET ANALYSIS (10,000 images)")
        print("=" * 60)
        
        # Confusion matrix
        print("\nComputing confusion matrix...")
        cm = compute_confusion_matrix(model, x_test, y_test)
        plot_confusion_matrix(cm, save_path=f'{SAVE_DIR}/confusion_matrix.png')
        
        # Per-class accuracy
        per_class = compute_per_class_accuracy(cm)
        
        # Binary pattern analysis (only for binary detection mode)
        binary_analysis = None
        if detection_method == 'binary':
            print("\nAnalyzing binary patterns...")
            print(f"  Threshold: {model.detector.threshold} (applied to normalized 0-1 intensities)")
            patterns_per_digit = get_binary_patterns(model, x_test, y_test)
            binary_analysis = analyze_binary_patterns(patterns_per_digit)
            
            # Infer grid size from pattern length
            pattern_len = len(binary_analysis[0]['top_pattern']) if binary_analysis[0]['top_pattern'] else 9
            grid_size = int(np.sqrt(pattern_len))
            
            plot_binary_pattern_summary(binary_analysis, grid_size=grid_size,
                                       save_path=f'{SAVE_DIR}/binary_patterns.png')
            plot_pattern_to_logits(model, save_path=f'{SAVE_DIR}/pattern_logits.png')
            
            # NEW: Detailed pattern-digit mapping visualizations
            plot_pattern_digit_mapping(model, grid_size=grid_size,
                                      save_path=f'{SAVE_DIR}/pattern_digit_mapping_learned.png')
            plot_observed_pattern_mapping(binary_analysis, grid_size=grid_size,
                                         save_path=f'{SAVE_DIR}/pattern_digit_mapping_observed.png')
            
            # Pattern assignment: which patterns belong to which digit
            assignment = get_pattern_assignment(model, grid_size=grid_size)
            if assignment:
                plot_pattern_assignment(assignment, grid_size=grid_size,
                                      save_path=f'{SAVE_DIR}/pattern_assignment.png')
                save_pattern_assignment_txt(assignment, grid_size=grid_size,
                                           save_path=f'{SAVE_DIR}/pattern_assignment.txt')
                print_pattern_assignment(assignment, grid_size=grid_size)
            
            # Print pattern overlap analysis
            print_pattern_overlap(binary_analysis)
        
        # MaxZone analysis
        if detection_method == 'maxzone':
            print("\nAnalyzing MaxZone assignments...")
            zone_assignment = model.detector.get_zone_assignment()
            print_zone_assignment(zone_assignment, grid_size)
            plot_zone_assignment_map(zone_assignment, grid_size,
                                    save_path=f'{SAVE_DIR}/zone_assignment_map.png')
            
            # Analyze which zone each test sample activates
            maxzone_analysis = analyze_maxzone_hits(model, x_test, y_test)
            print_maxzone_hit_summary(maxzone_analysis, zone_assignment, grid_size)
            save_zone_assignment_txt(zone_assignment, grid_size,
                                    save_path=f'{SAVE_DIR}/zone_assignment.txt')
        
        # Zone intensity analysis (ALL modes)
        print("\nAnalyzing zone intensities...")
        zone_intensities_per_digit, num_zones = get_zone_intensities(model, x_test, y_test)
        avg_intensities, std_intensities = compute_avg_zone_intensities(zone_intensities_per_digit)
        
        # Infer grid size from num_zones
        grid_size_zones = int(np.sqrt(num_zones))
        
        plot_zone_intensity_heatmap(avg_intensities, grid_size=grid_size_zones,
                                    save_path=f'{SAVE_DIR}/zone_intensity_heatmap.png')
        plot_zone_intensity_comparison(avg_intensities, std_intensities, grid_size=grid_size_zones,
                                       save_path=f'{SAVE_DIR}/zone_intensity_comparison.png')
        
        # Print summary
        print_analysis_summary(cm, per_class, binary_analysis)
        
        # Save text summary
        with open(f'{SAVE_DIR}/analysis_summary.txt', 'w') as f:
            total = cm.sum().item()
            correct = cm.diag().sum().item()
            f.write(f"Detection Method: {detection_method}\n")
            f.write(f"Overall Accuracy: {correct}/{total} = {100.0*correct/total:.2f}%\n\n")
            f.write("Per-Class Accuracy:\n")
            for digit, acc in per_class.items():
                f.write(f"  Digit {digit}: {acc:.2f}%\n")
            
            f.write(f"\nZone Intensities ({grid_size_zones}x{grid_size_zones} = {num_zones} zones):\n")
            for digit in range(10):
                avg = avg_intensities[digit]
                f.write(f"  Digit {digit}: [{', '.join(f'{v:.3f}' for v in avg)}]\n")
            
            if binary_analysis:
                f.write("\nBinary Patterns (most common per digit):\n")
                for digit in range(10):
                    info = binary_analysis[digit]
                    pattern = info['top_pattern']
                    pattern_str = ''.join(str(b) for b in pattern) if pattern else 'N/A'
                    coverage = 100.0 * info['top_count'] / info['total'] if info['total'] > 0 else 0
                    f.write(f"  Digit {digit}: [{pattern_str}] ({coverage:.0f}%, {info['unique']} unique)\n")
        print(f"  Saved: {SAVE_DIR}/analysis_summary.txt")
    
    # ===================
    # DIGIT VISUALIZATIONS
    # ===================
    if VISUALIZE_DIGITS:
        print(f"\n{'='*60}")
        print("PROPAGATION VISUALIZATIONS")
        print("=" * 60)
        print("Generating visualizations for digits 0-9...")
        
        correct = 0
        for digit in range(10):
            idx = (y_train == digit).nonzero()[0][0]
            img = x_train[idx, 0]
            
            pred_str = ""
            if model:
                pred, _ = predict(model, img)
                pred_str = f" ✓" if pred == digit else f" ✗ (pred={pred})"
                correct += pred == digit
            
            print(f"  Digit {digit}{pred_str}")
            visualize(img, masks, LAYER_SPACINGS, digit_label=digit,
                      save_path=f'{SAVE_DIR}/digit_{digit}.png',
                      use_fill_factor=USE_FILL_FACTOR,
                      use_contrast_limits=USE_CONTRAST_LIMITS,
                      use_binary=USE_BINARY,
                      grid_size=grid_size)
        
        if model:
            print(f"\nSample accuracy: {correct}/10 = {correct*10}%")
    
    print(f"\n{'='*60}")
    print(f"All results saved to '{SAVE_DIR}/'")
    print("Done!")