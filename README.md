# LCD-Based Optical Neural Network for MNIST Classification

**ELEC4848 Senior Design Project — The University of Hong Kong**

A differentiable simulation framework for a two-layer amplitude-modulating **Optical Neural Network (ONN)** built from off-the-shelf 200 × 200 monochrome LCDs. Achieves **90.85 % test accuracy** on MNIST — within 0.9 percentage points of the five-layer phase-mask D²NN benchmark of Lin et al. (2018) — at over two orders of magnitude lower hardware cost.

---

## Overview

Light propagation physics (Angular Spectrum Method) is implemented as differentiable PyTorch operations, so LCD amplitude masks are directly optimised by backpropagation. The trained masks can be exported as PNG images and displayed on physical LCD screens for hardware deployment.

```
532 nm laser → [LCD layer 1] ──5 cm──▶ [LCD layer 2] ──5 cm──▶ Camera → 10-class output
               200×200 px                200×200 px               zone detection
               138.3 µm pitch            138.3 µm pitch
```

### Key results

| Detection method | Test accuracy |
|---|---|
| Direct zone (2 × 5) | 28.6 % |
| Center-based (3 × 3) | 79.1 % |
| Binary encoding (3 × 3) | 82.3 % |
| Max-zone (4 × 4) | 87.4 % |
| **Max-zone (5 × 5)** | **90.85 %** |
| Lin et al. benchmark (5 phase layers) | 91.75 % |

---

## Repository structure

```
ONN/
├── config.py                  # All hardware & training parameters (single source of truth)
├── physics.py                 # Angular Spectrum Method propagation (PyTorch, differentiable)
├── model.py                   # AmplitudeLayer, 4 detector types, OpticalNeuralNetwork
├── data.py                    # MNIST loader: 28×28 → resize → centre-pad to 200×200
├── train.py                   # Training loop (cosine LR, early stopping, checkpointing)
├── visualize_propagation.py   # Post-training analysis: confusion matrix, zone heatmaps, mask export
├── generate_figures.py        # Generates all report figures (system diagram, training curves, etc.)
├── validate_numpy.py          # Independent NumPy re-implementation for numerical validation
├── export_matlab.py           # Exports trained masks + test set to .mat for MATLAB validation
├── matlab/
│   └── onn_simulate.m         # MATLAB independent validation script
├── requirements.txt
└── .gitignore
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

Default: max-zone 5 × 5 detector, 150 epochs, cosine LR annealing, early stopping (patience 30).  
Checkpoints and mask PNGs are saved to `checkpoints_maxzone_5x5/`.

Edit the constants at the top of `train.py` to switch detection method or grid size:

```python
DETECTION_METHOD = 'maxzone'   # 'zone' | 'center' | 'binary' | 'maxzone'
DETECTOR_ROWS    = 5
DETECTOR_COLS    = 5
```

### 3. Visualise results

```bash
python visualize_propagation.py
```

Generates confusion matrix, per-class accuracy, zone intensity heatmaps, and exports trained masks as `mask_1.png` / `mask_2.png` for direct LCD display.

### 4. Reproduce report figures

```bash
python generate_figures.py
```

Generates all six report figures (system architecture schematic, zone assignment map, training curves, per-class accuracy, confusion matrix, trained masks) into `report/figures/`.

### 5. Independent numerical validation

```bash
python validate_numpy.py   # NumPy re-implementation (verifies float32 precision dependency)
python export_matlab.py    # export masks + test set → matlab/export/*.mat
# then: cd matlab && run onn_simulate.m in MATLAB
```

> **Precision note:** The trained float32 transfer function H must be preserved exactly.
> Recomputing H in float64 causes an accuracy collapse from ~91 % to ~43 % due to float32
> phase-quantisation error of ~0.06 rad per bin compounding across two FFT propagations.

---

## Hardware target

| Parameter | Value |
|---|---|
| LCD model | 1.54″ monochrome reflective LCD |
| Resolution | 200 × 200 pixels |
| Pixel pitch | 138.3 µm |
| Active area | 27.66 mm × 27.66 mm |
| Light source | 532 nm green laser |
| Inter-layer spacing | 50 mm |
| Number of layers | 2 |
| Cost per LCD | < USD 50 |

---

## Physics: Angular Spectrum Method

Each gap between layers applies the transfer function:

$$H(f_x, f_y) = \exp\!\left(i \,k\, z \sqrt{1 - \lambda^2 f_x^2 - \lambda^2 f_y^2}\right)$$

with evanescent waves ($\lambda^2 f_x^2 + \lambda^2 f_y^2 > 1$) zeroed out. The full pipeline per forward pass is:

1. Cast input image to complex field $E_0$
2. For each layer $\ell$:  
   a. Multiply by sigmoid-activated amplitude mask  
   b. FFT → multiply $H_\ell$ → IFFT (propagate)
3. Compute intensity $|E_{\text{out}}|^2$
4. Extract zone intensities → detector → cross-entropy loss

The transfer functions are precomputed once and stored as buffers; only the mask parameters are trained.

---

## Configuration

All parameters live in `config.py`. Important knobs:

```python
# config.py
MultiLayerConfig(
    num_layers     = 2,
    layer_spacings = [5e-2, 5e-2],   # metres
)
LCDConfig(
    resolution  = (200, 200),
    pixel_pitch = 138.3e-6,          # metres
    is_binary   = False,             # True → STE binarisation during training
)
LightSourceConfig(
    wavelength = 532e-9,             # metres
)
TrainingConfig(
    epochs         = 150,
    learning_rate  = 0.01,
    batch_size     = 64,
)
```

---

## Citation / Reference

This project replicates and extends:

> Y. Lin, Y. Rivenson, N. T. Yardimci, M. Veli, Y. Luo, M. Jarrahi, and A. Ozcan,  
> "All-optical machine learning using diffractive deep neural networks,"  
> *Science*, vol. 361, no. 6406, pp. 1004–1008, 2018.  
> DOI: [10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084)

---

## License

This project is submitted as academic coursework. Source code is provided for reference.
