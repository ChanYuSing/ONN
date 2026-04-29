"""
physics.py - Light Propagation for Optical Neural Network
==========================================================

Simulates how light diffracts as it travels between LCD layers.

Core method: Angular Spectrum
    1. FFT(field) → frequency domain
    2. Multiply by transfer function H
    3. IFFT → spatial domain

    H(fx,fy) = exp(i·k·z·√(1 - λ²fx² - λ²fy²))

All parameters come from config.py.

Usage:
    from physics import propagate, create_transfer_function
    H = create_transfer_function(N, wavelength, pixel_pitch, distance)
    output = propagate(input_field, H)
"""

import torch
import numpy as np

from config import Config

_config = Config()


# =============================================================================
# CORE PROPAGATION
# =============================================================================

def create_transfer_function(N, wavelength, pixel_pitch, distance, device='cpu'):
    """
    Create transfer function H for angular spectrum propagation.
    
    Args:
        N: Grid size (200)
        wavelength: Light wavelength in meters
        pixel_pitch: Pixel size in meters  
        distance: Propagation distance in meters
        device: 'cpu' or 'cuda'
    
    Returns:
        H: Complex [N, N] transfer function
    """
    k = 2 * np.pi / wavelength
    
    # Spatial frequencies
    fx = torch.fft.fftfreq(N, d=pixel_pitch, device=device)
    fy = torch.fft.fftfreq(N, d=pixel_pitch, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    
    # sqrt(1 - (λfx)² - (λfy)²)
    sqrt_arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    
    # Filter evanescent waves (high frequencies that don't propagate)
    propagating = sqrt_arg >= 0
    sqrt_arg = torch.clamp(sqrt_arg, min=0)
    
    # H = exp(i·k·z·sqrt(...))
    H = torch.exp(1j * k * distance * torch.sqrt(sqrt_arg))
    H = torch.where(propagating, H, torch.zeros_like(H))
    
    return H


def propagate(field, H):
    """
    Propagate field using angular spectrum: FFT → multiply H → IFFT.
    
    Args:
        field: Complex [batch, N, N] or [N, N]
        H: Transfer function [N, N]
    
    Returns:
        Propagated field, same shape as input
    """
    squeeze = field.dim() == 2
    if squeeze:
        field = field.unsqueeze(0)
    
    output = torch.fft.ifft2(torch.fft.fft2(field) * H.unsqueeze(0))
    
    return output.squeeze(0) if squeeze else output


def propagate_distance(field, distance, wavelength=None, pixel_pitch=None):
    """
    Propagate field a given distance (convenience function).
    
    Args:
        field: Complex [batch, N, N] or [N, N]
        distance: Propagation distance in meters
        wavelength: From config if None
        pixel_pitch: From config if None
    """
    if wavelength is None:
        wavelength = _config.light.wavelength
    if pixel_pitch is None:
        pixel_pitch = _config.lcd.pixel_pitch
    
    H = create_transfer_function(field.shape[-1], wavelength, pixel_pitch, distance, field.device)
    return propagate(field, H)


# =============================================================================
# LCD EFFECTS
# =============================================================================

def apply_fill_factor(field, fill_factor=0.9):
    """
    Simulate pixel gaps (fill_factor < 1.0 means gaps between pixels).
    """
    if fill_factor >= 1.0:
        return field
    
    N = field.shape[-1]
    device = field.device
    
    x = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    
    # Position within each pixel [0, 1)
    px, py = (X * N) % 1, (Y * N) % 1
    
    # Only pass light in central region
    margin = (1 - fill_factor) / 2
    aperture = ((px >= margin) & (px < 1 - margin) & 
                (py >= margin) & (py < 1 - margin)).float()
    
    return field * aperture


def apply_contrast_limits(amplitude, min_transmission=0.02, max_transmission=0.95):
    """
    Scale amplitude to realistic LCD transmission range.
    Real LCDs can't achieve perfect black (0) or white (1).
    """
    return min_transmission + amplitude * (max_transmission - min_transmission)


def apply_binary_quantization(amplitude, threshold=0.5, method='ste', sharpness=10.0):
    """
    Quantize to binary (0 or 1) for binary LCD displays.
    
    Methods:
        'hard'   : Step function (inference)
        'ste'    : Straight-through estimator (training)
        'sigmoid': Soft approximation (training)
    """
    if method == 'hard':
        return (amplitude >= threshold).float()
    
    elif method == 'ste':
        # Forward: hard, Backward: pass gradient through
        hard = (amplitude >= threshold).float()
        return hard.detach() + amplitude - amplitude.detach()
    
    elif method == 'sigmoid':
        return torch.sigmoid(sharpness * (amplitude - threshold))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_lcd_effects(amplitude, fill_factor=None, min_transmission=None, 
                      max_transmission=None, binary=False, binary_method='ste', 
                      binary_sharpness=10.0):
    """Apply all LCD effects: binary → contrast limits."""
    result = amplitude
    
    if binary:
        result = apply_binary_quantization(result, method=binary_method, sharpness=binary_sharpness)
    
    if min_transmission is not None and max_transmission is not None:
        result = apply_contrast_limits(result, min_transmission, max_transmission)
    
    return result


# =============================================================================
# DEMO (run: python physics.py)
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("LIGHT PROPAGATION PHYSICS - DEMONSTRATION")
    print("="*60)
    
    # Parameters from config
    N = _config.lcd.resolution[0]
    wavelength = _config.light.wavelength
    pixel_pitch = _config.lcd.pixel_pitch
    
    print(f"\nParameters (from config):")
    print(f"  Grid size: {N}×{N}")
    print(f"  Wavelength: {wavelength*1e9:.0f} nm")
    print(f"  Pixel pitch: {pixel_pitch*1e6:.1f} μm")
    print(f"  Physical size: {N * pixel_pitch * 1e3:.2f} mm")
    
    # Create test patterns
    print("\n" + "-"*60)
    print("TEST 1: Point Source Diffraction")
    print("-"*60)
    
    # Single bright point in center
    point_source = torch.zeros(N, N, dtype=torch.complex64)
    point_source[N//2, N//2] = 1.0
    
    # Propagate at different distances
    distances = [1e-2, 3e-2, 5e-2, 10e-2]  # 1cm, 3cm, 5cm, 10cm
    
    fig, axes = plt.subplots(2, len(distances)+1, figsize=(15, 6))
    
    # Show input
    axes[0, 0].imshow(point_source.abs().numpy(), cmap='hot')
    axes[0, 0].set_title('Input\n(point source)')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(point_source.angle().numpy(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Phase')
    axes[1, 0].axis('off')
    
    for i, z in enumerate(distances):
        H = create_transfer_function(N, wavelength, pixel_pitch, z)
        output = propagate(point_source, H)
        
        intensity = output.abs()**2
        phase = output.angle()
        
        axes[0, i+1].imshow(intensity.numpy(), cmap='hot')
        axes[0, i+1].set_title(f'z = {z*100:.0f} cm')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(phase.numpy(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[1, i+1].set_title(f'Phase')
        axes[1, i+1].axis('off')
    
    axes[0, 0].set_ylabel('Intensity |E|²', fontsize=12)
    axes[1, 0].set_ylabel('Phase ∠E', fontsize=12)
    
    plt.suptitle('Point Source Diffraction (light spreads as it propagates)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_point_diffraction.png', dpi=150)
    print("Saved: physics_point_diffraction.png")
    plt.show()
    
    # Test 2: Slit diffraction
    print("\n" + "-"*60)
    print("TEST 2: Single Slit Diffraction")
    print("-"*60)
    
    # Vertical slit (10 pixels wide)
    slit = torch.zeros(N, N, dtype=torch.complex64)
    slit_width = 10
    slit[:, N//2 - slit_width//2 : N//2 + slit_width//2] = 1.0
    
    fig, axes = plt.subplots(1, len(distances)+1, figsize=(15, 3))
    
    axes[0].imshow(slit.abs().numpy(), cmap='gray')
    axes[0].set_title(f'Input slit\n(width={slit_width}px)')
    axes[0].axis('off')
    
    for i, z in enumerate(distances):
        output = propagate_distance(slit, z, wavelength, pixel_pitch)
        intensity = output.abs()**2
        
        axes[i+1].imshow(intensity.numpy(), cmap='hot')
        axes[i+1].set_title(f'z = {z*100:.0f} cm')
        axes[i+1].axis('off')
    
    plt.suptitle('Single Slit Diffraction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_slit_diffraction.png', dpi=150)
    print("Saved: physics_slit_diffraction.png")
    plt.show()
    
    # Test 3: MNIST digit propagation
    print("\n" + "-"*60)
    print("TEST 3: MNIST Digit Propagation")
    print("-"*60)
    
    # Load a sample digit
    from data import load_mnist_resized
    x_train, y_train, _, _ = load_mnist_resized()
    
    # Get one of each digit
    fig, axes = plt.subplots(10, 5, figsize=(12, 24))
    
    for digit in range(10):
        idx = (y_train == digit).nonzero()[0][0]
        img = x_train[idx, 0]  # [200, 200]
        
        # Convert to complex field (amplitude only, no phase)
        field = img.to(torch.complex64)
        
        # Show original
        axes[digit, 0].imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[digit, 0].set_title('Input' if digit == 0 else '')
        axes[digit, 0].set_ylabel(f'Digit {digit}', fontsize=12)
        axes[digit, 0].axis('off')
        
        # Propagate at different distances
        test_distances = [2e-2, 5e-2, 8e-2, 12e-2]
        for i, z in enumerate(test_distances):
            output = propagate_distance(field, z, wavelength, pixel_pitch)
            intensity = output.abs()**2
            
            # Normalize for visualization
            intensity = intensity / intensity.max()
            
            axes[digit, i+1].imshow(intensity.numpy(), cmap='hot')
            if digit == 0:
                axes[digit, i+1].set_title(f'z={z*100:.0f}cm')
            axes[digit, i+1].axis('off')
    
    plt.suptitle('MNIST Digits After Light Propagation\n(digits become blurry due to diffraction)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_mnist_propagation.png', dpi=150)
    print("Saved: physics_mnist_propagation.png")
    plt.show()

