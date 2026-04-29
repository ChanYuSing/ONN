"""
validate_numpy.py - Independent NumPy ASM validation of trained ONN
====================================================================
Step 1: Run the actual PyTorch model to confirm 90.85% (ground truth).
Step 2: Extract H from checkpoint, compare with freshly computed H.
Step 3: Run numpy with the EXACT H from the checkpoint (float32→float64 cast).
Step 4: If still wrong, expose the step-by-step field difference.

Root cause hypothesis: H stored in checkpoint (complex64/float32) may differ
from freshly computed H (complex128/float64) due to limited float32 phase
precision.  k*z ≈ 590,000 rad with float32 ULP ≈ 0.06 rad → ~6% phase error
per frequency bin, which compounds across two FFT propagations.
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from data import load_mnist_resized
from model import OpticalNeuralNetwork
from physics import create_transfer_function


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def numpy_propagate(field, H):
    return np.fft.ifft2(np.fft.fft2(field) * H)


def zone_winner(intensity, N=200, grid_sz=5):
    zone_size = N // grid_sz
    zi = np.zeros(grid_sz * grid_sz)
    for zy in range(grid_sz):
        for zx in range(grid_sz):
            r1, r2 = zy * zone_size, (zy + 1) * zone_size
            c1, c2 = zx * zone_size, (zx + 1) * zone_size
            zi[zy * grid_sz + zx] = intensity[r1:r2, c1:c2].sum()
    return zi.argmax(), zi


def numpy_forward(img, masks_np, H):
    f = img.astype(complex)
    f = f * masks_np[0]
    f = numpy_propagate(f, H)
    f = f * masks_np[1]
    f = numpy_propagate(f, H)
    return np.abs(f) ** 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    config = Config()
    N     = config.lcd.resolution[0]
    wl    = config.light.wavelength
    pp    = config.lcd.pixel_pitch
    gap   = config.multilayer.layer_spacings[0]

    ckpt_path = "checkpoints_maxzone_5x5/best.pt"
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    print("=== ONN Validation ===\n")

    # -----------------------------------------------------------------------
    # STEP 1 — Run actual PyTorch model (ground truth)
    # -----------------------------------------------------------------------
    print("--- Step 1: PyTorch model forward pass (ground truth) ---")
    model = OpticalNeuralNetwork(config, detection_method='maxzone', rows=5, cols=5)
    model.load_state_dict(state)
    model.eval()

    _, _, x_test, y_test = load_mnist_resized()
    x_np = x_test[:, 0].numpy()   # [10000, 200, 200]
    y_np = y_test.numpy()

    correct_torch = 0
    BATCH = 500
    with torch.no_grad():
        for start in range(0, len(x_np), BATCH):
            imgs = x_test[start:start+BATCH]       # [B, 1, 200, 200]
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            correct_torch += (preds == y_test[start:start+BATCH]).sum().item()
    acc_torch = 100.0 * correct_torch / len(x_np)
    print(f"  PyTorch accuracy: {acc_torch:.2f}%  (expected ~90.85%)\n")

    # -----------------------------------------------------------------------
    # STEP 2 — Compare H: saved in checkpoint vs freshly computed
    # -----------------------------------------------------------------------
    print("--- Step 2: Transfer function comparison ---")

    H_saved_torch  = state['transfer_functions'][0]          # complex64 (float32)
    H_fresh_torch  = create_transfer_function(N, wl, pp, gap)  # also complex64
    H_numpy_fresh  = np.array(H_fresh_torch.numpy(), dtype=complex)  # cast to complex128

    diff_fresh     = (H_saved_torch - H_fresh_torch).abs()
    diff_saved_np  = np.abs(H_saved_torch.numpy().astype(complex) - H_numpy_fresh)

    print(f"  H_saved  dtype:              {H_saved_torch.dtype}")
    print(f"  H(saved) vs H(fresh torch):  max_diff={diff_fresh.max():.4e}  mean_diff={diff_fresh.mean():.4e}")
    print(f"  H(saved) vs H(numpy float64): max_diff={diff_saved_np.max():.4e}  mean_diff={diff_saved_np.mean():.4e}")
    print(f"  H(numpy) shape: {H_numpy_fresh.shape}, dtype: {H_numpy_fresh.dtype}\n")

    # Extract masks
    masks_np = []
    for i in range(config.multilayer.num_layers):
        raw = state[f"layers.{i}.raw"].numpy()
        mask = 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
        masks_np.append(mask)

    zone_logits   = state['detector.zone_logits'].numpy()
    zone_to_digit = zone_logits.argmax(axis=1)

    idx0 = (y_np == 0).nonzero()[0][0]
    img0 = x_np[idx0]

    # -----------------------------------------------------------------------
    # STEP 3a — NumPy with freshly computed H (float64)
    # -----------------------------------------------------------------------
    print("--- Step 3a: NumPy  +  fresh H (float64) ---")
    I_np_fresh  = numpy_forward(img0, masks_np, H_numpy_fresh)
    w_fresh, _  = zone_winner(I_np_fresh, N)
    print(f"  Winner zone: {w_fresh} → digit {zone_to_digit[w_fresh]}  (true: 0)\n")

    # -----------------------------------------------------------------------
    # STEP 3b — NumPy with SAVED checkpoint H (cast float32→float64)
    # -----------------------------------------------------------------------
    print("--- Step 3b: NumPy  +  saved checkpoint H (float32 cast to float64) ---")
    H_saved_np   = H_saved_torch.numpy().astype(complex)
    I_np_saved   = numpy_forward(img0, masks_np, H_saved_np)
    w_saved, _   = zone_winner(I_np_saved, N)
    print(f"  Winner zone: {w_saved} → digit {zone_to_digit[w_saved]}  (true: 0)\n")

    # -----------------------------------------------------------------------
    # STEP 3c — NumPy with saved H keeping complex64 precision (no cast)
    # -----------------------------------------------------------------------
    print("--- Step 3c: NumPy  +  saved checkpoint H  (kept as complex64/float32) ---")
    H_saved_f32  = H_saved_torch.numpy()           # complex64 numpy array
    img0_f32     = img0.astype(np.float32)
    f = img0_f32.astype(np.complex64)
    f = f * masks_np[0].astype(np.float32)
    f = np.fft.ifft2(np.fft.fft2(f) * H_saved_f32)
    f = f * masks_np[1].astype(np.float32)
    f = np.fft.ifft2(np.fft.fft2(f) * H_saved_f32)
    I_np_f32     = np.abs(f) ** 2
    w_f32, _     = zone_winner(I_np_f32.real, N)
    print(f"  Winner zone: {w_f32} → digit {zone_to_digit[w_f32]}  (true: 0)\n")

    # -----------------------------------------------------------------------
    # STEP 4 — Step-by-step field comparison: numpy vs torch
    # -----------------------------------------------------------------------
    print("--- Step 4: Step-by-step field error (numpy float64 vs torch float32) ---")
    img0_t = torch.tensor(img0).to(torch.complex64)
    m1_t   = torch.tensor(masks_np[0], dtype=torch.float32)
    m2_t   = torch.tensor(masks_np[1], dtype=torch.float32)
    H_t    = H_saved_torch

    f_t = img0_t * m1_t
    f_n = img0.astype(complex) * masks_np[0]
    e1 = np.abs(f_t.numpy().astype(complex) - f_n)
    print(f"  After mask1:  max_err={e1.max():.4e}  rel_err={e1.max()/(np.abs(f_n).max()+1e-20):.4e}")

    f_t = torch.fft.ifft2(torch.fft.fft2(f_t) * H_t)
    f_n = numpy_propagate(f_n, H_numpy_fresh)
    e2 = np.abs(f_t.numpy().astype(complex) - f_n)
    print(f"  After prop1:  max_err={e2.max():.4e}  rel_err={e2.max()/(np.abs(f_n).max()+1e-20):.4e}")

    f_t = f_t * m2_t
    f_n = f_n * masks_np[1]
    e3 = np.abs(f_t.numpy().astype(complex) - f_n)
    print(f"  After mask2:  max_err={e3.max():.4e}  rel_err={e3.max()/(np.abs(f_n).max()+1e-20):.4e}")

    f_t = torch.fft.ifft2(torch.fft.fft2(f_t) * H_t)
    f_n = numpy_propagate(f_n, H_numpy_fresh)
    e4 = np.abs(f_t.numpy().astype(complex) - f_n)
    print(f"  After prop2:  max_err={e4.max():.4e}  rel_err={e4.max()/(np.abs(f_n).max()+1e-20):.4e}")
    print()

    # -----------------------------------------------------------------------
    # STEP 5 — Full accuracy using saved H (cast float64)
    # -----------------------------------------------------------------------
    print("--- Step 5: Full accuracy — NumPy with saved-checkpoint H (float64 cast) ---")
    correct = 0
    for n in range(len(x_np)):
        I = numpy_forward(x_np[n], masks_np, H_saved_np)
        w, _ = zone_winner(I, N)
        if zone_to_digit[w] == y_np[n]:
            correct += 1
        if (n + 1) % 1000 == 0:
            print(f"  {n+1:5d}/10000  acc={100*correct/(n+1):.2f}%")

    acc_np = 100.0 * correct / len(x_np)
    print(f"\n  NumPy (saved H, f64) accuracy: {acc_np:.2f}%")
    print(f"  PyTorch accuracy:              {acc_torch:.2f}%")
    print(f"  Match: {'YES ✓' if abs(acc_np - acc_torch) < 1.0 else 'NO ✗ — investigate further'}")


if __name__ == "__main__":
    main()
