"""
Export trained ONN data to MATLAB .mat format.

Exports:
  - masks.mat: trained amplitude masks (after sigmoid)
  - digits.mat: 2 sample digits per class (0-9) from test set
  - params.mat: all physical parameters
  - test_set.mat: full 10k test images + labels (for full validation)

Usage:
    python export_matlab.py
"""

import torch
import numpy as np
import os

from config import Config
from data import load_mnist_resized
from physics import create_transfer_function, propagate
from model import create_grid_zones

try:
    from scipy.io import savemat
except ImportError:
    print("scipy required: pip install scipy")
    exit(1)


def main():
    config = Config()
    output_dir = "matlab_export"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load trained masks ---
    ckpt_path = "checkpoints_maxzone_5x5/best.pt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    masks = []
    for i in range(config.multilayer.num_layers):
        key = f"layers.{i}.raw"
        raw = checkpoint['model_state_dict'][key]
        mask = torch.sigmoid(raw).numpy()
        masks.append(mask)
        print(f"Mask {i+1}: shape={mask.shape}, min={mask.min():.4f}, max={mask.max():.4f}")

    # Save masks
    savemat(os.path.join(output_dir, "masks.mat"), {
        'mask1': masks[0],
        'mask2': masks[1],
    })
    print(f"  Saved: {output_dir}/masks.mat")

    # --- Physical parameters ---
    pixel_pitch = config.lcd.pixel_pitch  # meters
    wavelength = config.light.wavelength  # meters
    gap = config.multilayer.layer_spacings[0]  # meters
    N = config.lcd.resolution[0]

    # --- Transfer function H (float32 precision — CRITICAL for MATLAB) ---
    # The ONN masks were optimised against float32 H (torch.fft with float32 pixel_pitch).
    # Recomputing H in MATLAB float64 gives slightly different phases (~0.07 rad ULP error
    # at k·z ≈ 590,000 rad) which shifts the output intensity pattern → wrong zone wins.
    # Solution: export the exact float32 H from the checkpoint, load it in MATLAB.
    H_torch = create_transfer_function(N, wavelength, pixel_pitch, gap, device='cpu')
    # Cast complex64 → complex128: values are exact (upcast, no recomputation).
    savemat(os.path.join(output_dir, "H.mat"), {
        'H': H_torch.numpy().astype(np.complex128),
    })
    print(f"  Saved: {output_dir}/H.mat (float32-precision transfer function for MATLAB)")

    savemat(os.path.join(output_dir, "params.mat"), {
        'wavelength': wavelength,
        'pixel_pitch': pixel_pitch,
        'gap': gap,
        'N': N,
        'active_area': config.lcd.active_area[0],
        'num_layers': config.multilayer.num_layers,
    })
    print(f"  Saved: {output_dir}/params.mat")

    # --- Load MNIST ---
    x_train, y_train, x_test, y_test = load_mnist_resized()

    # Save sample digits (2 per class)
    samples = {}
    for digit in range(10):
        indices = (y_test == digit).nonzero(as_tuple=True)[0]
        for i in range(2):
            idx = indices[i].item()
            img = x_test[idx, 0].numpy()  # [200, 200]
            samples[f'digit{digit}_sample{i}'] = img
    
    savemat(os.path.join(output_dir, "digits.mat"), samples)
    print(f"  Saved: {output_dir}/digits.mat (20 sample images)")

    # Save full test set for accuracy comparison
    print("  Saving full test set (may take a moment)...")
    savemat(os.path.join(output_dir, "test_set.mat"), {
        'x_test': x_test[:, 0].numpy(),  # [10000, 200, 200]
        'y_test': y_test.numpy(),          # [10000]
    }, do_compression=True)
    print(f"  Saved: {output_dir}/test_set.mat (10000 images, compressed)")

    # --- Zone assignment table ---
    # Load the detector's zone_logits to export the mapping
    zone_logits_key = 'detector.zone_logits'
    if zone_logits_key in checkpoint['model_state_dict']:
        zone_logits = checkpoint['model_state_dict'][zone_logits_key].numpy()
        zone_to_digit = zone_logits.argmax(axis=1)  # [25]
        savemat(os.path.join(output_dir, "zone_map.mat"), {
            'zone_logits': zone_logits,     # [25, 10]
            'zone_to_digit': zone_to_digit, # [25]
            'grid_size': 5,
        })
        print(f"  Saved: {output_dir}/zone_map.mat")

    # --- Python reference propagation for digit 0 ---
    # Run the exact same computation as Python's ONN, save for MATLAB comparison
    print("  Computing Python reference propagation for digit 0...")
    config2 = Config()
    N2 = config2.lcd.resolution[0]
    wl = config2.light.wavelength
    pp = config2.lcd.pixel_pitch
    gap2 = config2.multilayer.layer_spacings[0]

    H_ref = create_transfer_function(N2, wl, pp, gap2, device='cpu')

    # Get first digit-0 image from test set
    idx0 = (y_test == 0).nonzero(as_tuple=True)[0][0].item()
    img0 = x_test[idx0, 0]  # [200, 200]

    # Load masks as tensors
    m1 = torch.tensor(masks[0])  # [200, 200]
    m2 = torch.tensor(masks[1])  # [200, 200]

    # Forward pass
    f = img0.to(torch.complex64)
    f = f * m1
    f = propagate(f, H_ref)
    f = f * m2
    f = propagate(f, H_ref)
    intensity_ref = f.abs().pow(2)  # [200, 200]

    # Zone intensities from Python
    rows_z, cols_z = 5, 5
    zones = create_grid_zones(N2, rows_z, cols_z)  # [25, 200, 200]
    zone_areas = zones.sum(dim=(1, 2))
    zone_int_ref = (intensity_ref.unsqueeze(0) * zones).sum(dim=(1, 2)) / (zone_areas + 1e-8)
    winner_py = zone_int_ref.argmax().item()
    print(f"  Python: winner zone {winner_py} (0-indexed) → digit {zone_to_digit[winner_py]}, "
          f"intensity={zone_int_ref[winner_py]:.6e}")

    savemat(os.path.join(output_dir, "python_reference.mat"), {
        'intensity_ref': intensity_ref.numpy(),     # [200, 200] Python-computed output
        'zone_ints_ref': zone_int_ref.numpy(),      # [25] zone intensities (0-indexed)
        'winner_zone': winner_py,                   # 0-indexed winner zone
        'input_image': img0.numpy(),                # [200, 200] input image (before masks)
        'mask1_py': masks[0],                       # [200, 200] mask1 in Python orientation
        'mask2_py': masks[1],                       # [200, 200] mask2 in Python orientation
    })
    print(f"  Saved: {output_dir}/python_reference.mat")

    print(f"\nAll exports in: {output_dir}/")
    print("Open MATLAB, run: onn_simulate.m")


if __name__ == "__main__":
    main()
