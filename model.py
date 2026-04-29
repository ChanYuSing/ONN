"""
model.py - Optical Neural Network Model
========================================

Trainable PyTorch module for amplitude-based optical neural network.

Architecture:
    Input (200×200) → [LCD Layer 1] → propagate → [LCD Layer 2] → ... → Detector → 10 logits

Components:
    - AmplitudeLayer: Trainable LCD mask (sigmoid → [0,1])
    - Detectors: zone, center, binary methods
    - OpticalNeuralNetwork: Full model

Usage:
    model = OpticalNeuralNetwork(config, detection_method='center')
    logits = model(images)  # [batch, 1, N, N] → [batch, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config
from physics import (
    create_transfer_function,
    propagate,
    apply_fill_factor,
    apply_contrast_limits,
    apply_binary_quantization
)


# =============================================================================
# DETECTION METHODS
# =============================================================================

def create_grid_zones(N, rows, cols):
    """Create n×n grid of binary zone masks. Returns [rows*cols, N, N]."""
    num_zones = rows * cols
    zones = torch.zeros(num_zones, N, N)
    
    h, w = N // rows, N // cols
    
    for i in range(num_zones):
        r, c = i // cols, i % cols
        r_start, r_end = r * h, (r + 1) * h if r < rows - 1 else N
        c_start, c_end = c * w, (c + 1) * w if c < cols - 1 else N
        zones[i, r_start:r_end, c_start:c_end] = 1.0
    
    return zones


class ZoneIntensityExtractor(nn.Module):
    """Extract average intensity from each zone in a grid."""
    
    def __init__(self, N, rows=4, cols=4):
        super().__init__()
        self.N = N
        self.rows = rows
        self.cols = cols
        self.num_zones = rows * cols
        
        zones = create_grid_zones(N, rows, cols)
        self.register_buffer('zones', zones)
        self.register_buffer('zone_areas', zones.sum(dim=(1, 2)))
    
    def extract(self, intensity):
        """intensity [batch, N, N] → zone_intensities [batch, num_zones]"""
        intensity = intensity.unsqueeze(1)  # [batch, 1, N, N]
        zones = self.zones.unsqueeze(0)     # [1, num_zones, N, N]
        zone_intensities = (intensity * zones).sum(dim=(2, 3))
        return zone_intensities / (self.zone_areas + 1e-8)


class ZoneDetector(nn.Module):
    """Direct zone-to-class mapping. 10 classes → 2×5 grid."""
    
    def __init__(self, N, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Find grid dimensions
        if num_classes == 10:
            rows, cols = 2, 5
        elif num_classes <= 4:
            rows, cols = 2, 2
        elif num_classes <= 9:
            rows, cols = 3, 3
        else:
            rows = int(np.ceil(np.sqrt(num_classes)))
            cols = int(np.ceil(num_classes / rows))
        
        self.extractor = ZoneIntensityExtractor(N, rows, cols)
    
    def forward(self, intensity):
        """intensity [batch, N, N] → logits [batch, num_classes]"""
        zone_intensities = self.extractor.extract(intensity)
        # Use log-intensity as logits: physically meaningful (intensity is log-scale)
        # and allows negative logits to suppress wrong classes
        return torch.log(zone_intensities[:, :self.num_classes] + 1e-8)


class CenterBasedDetector(nn.Module):
    """Linear projection from zone intensities to class logits."""
    
    def __init__(self, N, num_classes=10, rows=4, cols=4):
        super().__init__()
        self.extractor = ZoneIntensityExtractor(N, rows, cols)
        self.classifier = nn.Linear(rows * cols, num_classes)
    
    def forward(self, intensity):
        """intensity [batch, N, N] → logits [batch, num_classes]"""
        zone_intensities = self.extractor.extract(intensity)
        return self.classifier(zone_intensities)


class BinaryEncodingDetector(nn.Module):
    """
    Binary threshold encoding: zone intensities → binary pattern → class.
    For 3×3 grid: 9 zones → 2^9 = 512 patterns → learned mapping to 10 classes.
    """
    
    def __init__(self, N, num_classes=10, rows=2, cols=2, threshold=0.5):
        super().__init__()
        self.num_zones = rows * cols
        self.threshold = threshold
        self.sharpness = 10.0
        
        self.extractor = ZoneIntensityExtractor(N, rows, cols)
        
        # Learnable: pattern → logits
        self.num_patterns = 2 ** self.num_zones
        self.pattern_to_logits = nn.Parameter(torch.randn(self.num_patterns, num_classes) * 0.1)
        
        # Precompute all binary patterns once (avoids rebuilding every forward pass)
        all_patterns = torch.zeros(self.num_patterns, self.num_zones)
        for i in range(self.num_patterns):
            for j in range(self.num_zones):
                all_patterns[i, j] = (i >> j) & 1
        self.register_buffer('all_patterns', all_patterns)
    
    def _soft_binary(self, zone_intensities):
        """Differentiable soft binarization to ~0 or ~1."""
        z_min = zone_intensities.min(dim=1, keepdim=True)[0]
        z_max = zone_intensities.max(dim=1, keepdim=True)[0]
        z_norm = (zone_intensities - z_min) / (z_max - z_min + 1e-8)
        return torch.sigmoid(self.sharpness * (z_norm - self.threshold))
    
    def forward(self, intensity):
        """intensity [batch, N, N] → logits [batch, num_classes]"""
        zone_intensities = self.extractor.extract(intensity)
        binary_soft = self._soft_binary(zone_intensities)
        
        # Soft matching: distance to each pattern → weights
        pattern_dist = torch.cdist(binary_soft, self.all_patterns, p=2)
        pattern_weights = F.softmax(-pattern_dist * self.sharpness, dim=1)
        
        return torch.matmul(pattern_weights, self.pattern_to_logits)


class MaxZoneDetector(nn.Module):
    """
    Max-zone detection: find brightest zone → look up class from learned table.
    
    Uses more zones than classes (e.g., 5×5=25 for 10 classes).
    Each zone has a learned class-logit vector.
    
    Uses Gumbel-Softmax with hard=True during training to force the model
    to commit to a single brightest zone (not blend all zones). Gumbel noise
    provides exploration; STE provides gradients. Eval uses hard argmax.
    
    logit_scale amplifies normalized intensities so they dominate Gumbel noise
    (without scaling, [0,1] logits are drowned by Gumbel noise std≈1.28).
    Temperature τ anneals from 2.0 → 0.5 (controls backward gradient spread).
    """
    
    def __init__(self, N, num_classes=10, rows=5, cols=5, temperature=2.0):
        super().__init__()
        self.num_zones = rows * cols
        self.num_classes = num_classes
        self.temperature = temperature  # Will be updated by training loop
        self.logit_scale = nn.Parameter(torch.tensor(10.0))  # Learnable: model finds optimal scale
        
        self.extractor = ZoneIntensityExtractor(N, rows, cols)
        
        # Initialize zone_logits with round-robin class assignment.
        # Each zone starts strongly assigned to one class (zone i → class i % num_classes).
        # This prevents early collapse where some classes get 0 zones.
        init_logits = torch.full((self.num_zones, num_classes), -1.0)
        for z in range(self.num_zones):
            init_logits[z, z % num_classes] = 1.0
        self.zone_logits = nn.Parameter(init_logits)
    
    def forward(self, intensity):
        """intensity [batch, N, N] → logits [batch, num_classes]"""
        zone_intensities = self.extractor.extract(intensity)  # [batch, num_zones]
        
        # Normalize to [0, 1]
        z_min = zone_intensities.min(dim=1, keepdim=True)[0]
        z_max = zone_intensities.max(dim=1, keepdim=True)[0]
        z_norm = (zone_intensities - z_min) / (z_max - z_min + 1e-8)
        
        # Scale up so signal dominates Gumbel noise (~1.28 std)
        scaled_logits = z_norm * self.logit_scale
        
        if self.training:
            # Gumbel-Softmax: hard one-hot forward, soft gradient backward
            attention_weights = F.gumbel_softmax(
                scaled_logits, tau=self.temperature, hard=True
            )
        else:
            # Eval: deterministic hard argmax
            idx = scaled_logits.argmax(dim=1)
            attention_weights = F.one_hot(idx, self.num_zones).float()
        
        logits = torch.matmul(attention_weights, self.zone_logits)  # [batch, num_classes]
        
        return logits
    
    def get_zone_assignment(self):
        """Return dict: {zone_idx: (assigned_digit, weight)} based on learned logits."""
        assigned = self.zone_logits.detach().cpu()
        assignment = {}
        for z in range(self.num_zones):
            digit = assigned[z].argmax().item()
            weight = assigned[z, digit].item()
            assignment[z] = (digit, weight)
        return assignment


def create_detector(method, N, num_classes=10, **kwargs):
    """Factory: create detector by method name ('zone', 'center', 'binary', 'maxzone')."""
    if method == 'zone':
        return ZoneDetector(N, num_classes)
    elif method == 'center':
        return CenterBasedDetector(N, num_classes, kwargs.get('rows', 4), kwargs.get('cols', 4))
    elif method == 'binary':
        return BinaryEncodingDetector(N, num_classes, kwargs.get('rows', 2), kwargs.get('cols', 2))
    elif method == 'maxzone':
        return MaxZoneDetector(N, num_classes, kwargs.get('rows', 5), kwargs.get('cols', 5))
    else:
        raise ValueError(f"Unknown detection method: {method}")


# =============================================================================
# AMPLITUDE LAYER
# =============================================================================

class AmplitudeLayer(nn.Module):
    """Trainable LCD amplitude mask. Raw params → sigmoid → [0, 1]."""
    
    def __init__(self, N, init_type='smooth'):
        super().__init__()
        self.N = N
        
        if init_type == 'smooth':
            # Smooth initialization (low frequency pattern)
            low_res = N // 10
            raw_low = torch.randn(low_res, low_res) * 0.5
            raw = F.interpolate(
                raw_low.unsqueeze(0).unsqueeze(0),
                size=(N, N), mode='bilinear', align_corners=False
            )[0, 0]
        elif init_type == 'uniform':
            raw = torch.zeros(N, N)  # sigmoid(0) = 0.5
        elif init_type == 'random':
            raw = torch.randn(N, N) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        self.raw = nn.Parameter(raw)
    
    def get_amplitude(self, binary=False, binary_method='sigmoid', binary_sharpness=10.0,
                      min_transmission=None, max_transmission=None):
        """Get amplitude mask with optional LCD effects."""
        amplitude = torch.sigmoid(self.raw)
        
        if binary:
            amplitude = apply_binary_quantization(amplitude, method=binary_method, sharpness=binary_sharpness)
        
        if min_transmission is not None and max_transmission is not None:
            amplitude = apply_contrast_limits(amplitude, min_transmission, max_transmission)
        
        return amplitude
    
    def forward(self, field, **kwargs):
        """Apply amplitude mask to optical field."""
        return field * self.get_amplitude(**kwargs)


# =============================================================================
# OPTICAL NEURAL NETWORK
# =============================================================================

class OpticalNeuralNetwork(nn.Module):
    """
    Multi-layer ONN: Input → [Mask → Propagate] × N → Detect → Logits
    
    Args:
        config: Config object (hardware/training params)
        detection_method: 'zone', 'center', 'binary', or 'maxzone'
        **detection_kwargs: rows, cols for detector grid
    """
    
    def __init__(self, config=None, detection_method='center', **detection_kwargs):
        super().__init__()
        
        if config is None:
            config = Config()
        self.config = config
        
        # From config
        self.N = config.lcd.resolution[0]
        self.num_layers = config.multilayer.num_layers
        self.layer_spacings = config.multilayer.layer_spacings
        self.wavelength = config.light.wavelength
        self.pixel_pitch = config.lcd.pixel_pitch
        self.fill_factor = config.lcd.fill_factor
        self.min_transmission = config.lcd.min_transmission
        self.max_transmission = config.lcd.max_transmission
        self.is_binary = config.lcd.is_binary
        
        self.num_classes = 10  # MNIST
        self.detection_method = detection_method
        self.detection_kwargs = detection_kwargs
        
        # Trainable layers
        self.layers = nn.ModuleList([
            AmplitudeLayer(self.N, init_type='smooth')
            for _ in range(self.num_layers)
        ])
        
        # Precomputed transfer functions
        self.register_buffer('transfer_functions', self._create_transfer_functions())
        
        # Detector
        self.detector = create_detector(detection_method, self.N, self.num_classes, **detection_kwargs)
    
    def _create_transfer_functions(self):
        """Precompute H for each layer spacing."""
        H_list = [
            create_transfer_function(self.N, self.wavelength, self.pixel_pitch, spacing)
            for spacing in self.layer_spacings
        ]
        return torch.stack(H_list)
    
    def get_masks(self):
        """Get all amplitude masks (detached, for visualization)."""
        return [
            layer.get_amplitude(
                binary=self.is_binary,
                min_transmission=self.min_transmission if self.min_transmission > 0 else None,
                max_transmission=self.max_transmission if self.max_transmission < 1 else None
            ).detach()
            for layer in self.layers
        ]
    
    def forward(self, x, return_intermediate=False):
        """
        Forward pass: images → logits.
        
        Args:
            x: [batch, 1, N, N] or [batch, N, N]
            return_intermediate: Return intermediate fields for visualization
        
        Returns:
            logits: [batch, num_classes]
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        
        field = x.to(torch.complex64)
        intermediates = [field.clone()] if return_intermediate else None
        
        # Propagate through layers
        for i, layer in enumerate(self.layers):
            amplitude = layer.get_amplitude(
                binary=self.is_binary,
                binary_method='sigmoid' if self.training else 'hard',
                binary_sharpness=10.0,
                min_transmission=self.min_transmission if self.min_transmission > 0 else None,
                max_transmission=self.max_transmission if self.max_transmission < 1 else None
            )
            
            field = field * amplitude
            
            if self.fill_factor < 1.0:
                field = apply_fill_factor(field, self.fill_factor)
            
            if return_intermediate:
                intermediates.append(field.clone())
            
            field = propagate(field, self.transfer_functions[i])
            
            if return_intermediate:
                intermediates.append(field.clone())
        
        # Detect
        intensity = torch.abs(field) ** 2
        logits = self.detector(intensity)
        
        return (logits, intermediates) if return_intermediate else logits
    
    def save_masks(self, filepath):
        """Save trained masks to file."""
        torch.save({'masks': self.get_masks(), 'config': {
            'num_layers': self.num_layers, 'N': self.N,
            'wavelength': self.wavelength, 'pixel_pitch': self.pixel_pitch,
        }}, filepath)
    
    def load_masks(self, filepath):
        """Load masks from file (inverse sigmoid to get raw params)."""
        masks = torch.load(filepath)['masks']
        for i, mask in enumerate(masks):
            amplitude = mask.clamp(1e-6, 1-1e-6)
            self.layers[i].raw.data = torch.log(amplitude / (1 - amplitude))


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("MODEL TEST")
    print("=" * 60)
    
    config = Config()
    model = OpticalNeuralNetwork(config)
    
    print(f"\nModel: {model.num_layers} layers, {model.N}×{model.N} grid")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass test
    x = torch.rand(4, 1, model.N, model.N)
    logits = model(x)
    print(f"\nForward: {x.shape} → {logits.shape}")
    
    # Gradient test
    loss = F.cross_entropy(logits, torch.randint(0, 10, (4,)))
    loss.backward()
    print(f"Loss: {loss.item():.4f}, grad norm: {model.layers[0].raw.grad.norm():.4f}")
    
    # Visualize zones
    extractor = model.detector.extractor
    zone_vis = sum((i + 1) * z for i, z in enumerate(extractor.zones))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(zone_vis.numpy(), cmap='tab20')
    axes[0].set_title(f'Detection Zones ({extractor.rows}×{extractor.cols})')
    axes[0].axis('off')
    
    axes[1].imshow(model.get_masks()[0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Layer 1 Mask (initial)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_zones.png', dpi=150)
    print("\nSaved: model_zones.png")
    print("=" * 60)