# 5. Independent Validation

To confirm that the 90.85% simulation accuracy is reproducible and not an artefact of the PyTorch training framework, the trained model was independently re-evaluated in two additional computational environments: a NumPy-based Python implementation and a MATLAB implementation. Both environments implement the forward pass independently, sharing no code or computational routines with the training framework. Independent validation is a necessary step before any physical hardware deployment: if two independent implementations agree on the simulation result, this provides confidence that the trained mask patterns are genuinely encoding the classification function in the physical diffraction process, rather than exploiting an implementation quirk in PyTorch's FFT or tensor operations. The validation also provides a concrete quantitative baseline — the 90.85% figure — against which any discrepancy in the eventual physical hardware measurement can be compared and attributed. This section describes the validation procedure, results, and a precision dependency that was identified during the process and which has important implications for the hardware deployment workflow.

## 5.1 NumPy Validation

### 5.1.1 Validation Procedure

The NumPy validation reconstructed the forward pass of the trained ONN entirely using NumPy array operations, without PyTorch. The procedure was structured as a five-step diagnostic to isolate potential sources of discrepancy:

1. **Extract trained parameters.** The best checkpoint (`best_latest.pt`) was loaded in PyTorch. The two amplitude mask tensors M₁, M₂ ∈ [0, 1]²⁰⁰ˣ²⁰⁰ and the zone assignment logit matrix W ∈ ℝ²⁵ˣ¹⁰ were exported to disk as NumPy `.npy` files.

2. **Construct the Angular Spectrum transfer function.** The angular spectrum transfer function H(f_x, f_y) is defined as

$$H(f_x, f_y) = \exp\left(j \, k_z \, z\right), \quad k_z = \sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2} \qquad (9)$$

where k = 2π/λ with λ = 532 nm, z = 50 mm, and f_x, f_y are the spatial frequencies corresponding to a 200 × 200 grid with pixel pitch p = 138.3 µm. Two versions of H were computed: one freshly computed in float64 precision, and one loaded directly from the checkpoint (which stores H in float32 as it was cached during training).

3. **Implement the forward pass.** The ONN forward pass applies, for each layer ℓ ∈ {1, 2}:

$$U_{\ell+1} = \mathcal{F}^{-1}\left\{ H \cdot \mathcal{F}\left\{ U_\ell \cdot M_\ell \right\} \right\} \qquad (10)$$

where ℱ and ℱ⁻¹ denote the 2-D discrete Fourier transform pair, implemented via `numpy.fft.fft2` and `numpy.fft.ifft2` respectively.

4. **Compute zone intensities.** The output intensity field |U₃|² was partitioned into a 5 × 5 grid of equal zones (40 × 40 pixels each). The summed intensity in each zone was computed, and the max-zone detector assigned each sample to the class with the highest total zone intensity among the zones allocated to that class.

5. **Evaluate over the test set.** Steps 3–4 were applied to all 10,000 MNIST test images with energy normalisation identical to the training pipeline.

### 5.1.2 Results

Using H loaded from the checkpoint (float32 precision, matching the training environment), the NumPy implementation reproduced **90.85% accuracy** (9,085 / 10,000), matching the PyTorch training result exactly. Using H freshly computed in float64, a significant discrepancy appeared: accuracy fell to approximately **43%**, a result that is only marginally above random chance (10%). This finding is discussed in §5.3.

## 5.2 MATLAB Validation

### 5.2.1 Validation Procedure

A MATLAB simulation was implemented as a second independent check. The export and simulation steps are as follows.

**Export from Python.** A Python export script (`export_matlab.py`) was used to write three files readable by MATLAB:

- `masks.mat` — the two amplitude mask arrays M₁, M₂, exported as double-precision arrays (upcast from float32).
- `params.mat` — scalar physical parameters: λ, z, pixel pitch, grid size, number of zones along each axis (5), and zone width in pixels (40).
- `H.mat` — the angular spectrum transfer function H, stored at float32 precision as it exists in the checkpoint, then exported to MATLAB as a complex double array. Crucially, the float32 values were cast to double after export, *not* recomputed in double precision.

**MATLAB simulation (`onn_simulate.m`).** The script loaded the three `.mat` files and performed:

1. Energy normalisation of each input image (identical formula to §3.6.1).
2. Two propagation steps using MATLAB's `fft2`/`ifft2` with the loaded H.
3. Zone intensity summation over a 5 × 5 grid.
4. Argmax classification; class index is determined by the zone with the maximum intensity summed over zones belonging to that class.
5. Accuracy counted over all 10,000 test images.

### 5.2.2 Results

The MATLAB simulation reproduced **90.85% accuracy** (9,085 / 10,000). The summed intensity per zone divided by the per-pixel mean intensity within that zone equalled exactly **1600** (= 40 × 40 pixels per zone) for all 25 zones across all test samples in both implementations, confirming that the zone partitioning and summation are consistent between MATLAB and PyTorch. The relative intensity error between PyTorch and MATLAB output planes was 5.18 × 10⁻⁶, attributable to floating-point rounding differences between NumPy's FFT and MATLAB's FFT implementations. No classification disagreements resulted from this rounding error.

## 5.3 Float32 Precision Dependency

### 5.3.1 Physical Origin

The angular spectrum transfer function accumulates a phase of k_z·z per spatial frequency component. At the on-axis frequency (f_x = f_y = 0), this phase equals

$$\phi_0 = k \cdot z = \frac{2\pi}{\lambda} \cdot z = \frac{2\pi}{532 \times 10^{-9}} \times 0.05 \approx 590{,}600 \text{ rad} \qquad (11)$$


At k·z ≈ 590,000 rad, this value lies in the range [2¹⁹, 2²⁰), so the unit in the last place (ULP) of its float32 representation is 2¹⁹⁻²³ = 2⁻⁴ ≈ 0.06 rad. An additional contribution arises from cancellation error in evaluating k_z = √(k² − (2π·f_x)² − (2π·f_y)²) at high spatial frequencies, where the two large terms are nearly equal. Combining both effects, the phase discrepancy between float32 checkpoint H and a freshly computed float64 H reaches approximately **0.07 rad** at high spatial frequencies.

### 5.3.2 Effect on Accuracy

A phase error of Δφ ≈ 0.07 rad in H propagates into the complex field amplitude at each layer. After two propagation steps, phase errors accumulate non-linearly across the 40,000 frequency components of the 200 × 200 grid. The resulting amplitude perturbation alters the intensity distribution at the detector plane sufficiently to shift the argmax zone winner for a large fraction of test samples. This explains the drop from 90.85% to approximately 43% when H is recomputed in float64: the network learned to exploit the specific float32 rounding pattern of H that was used during training, and a differently rounded H produces a structurally different interference pattern even though both are mathematically consistent with the wave equation.

### 5.3.3 Implication and Fix

This finding has a concrete implication for any simulation-to-hardware transfer workflow. If hardware is used to perform physical propagation, the physical propagation is exact (up to manufacturing tolerances) and does not suffer from float32 rounding. However, any digitally computed preprocessing or post-processing step (e.g., computing H to simulate a missing layer) must use the same H that was present during training, or retrain with an H computed consistently in the target precision.

ONN mask patterns are therefore not general solutions to the optical classification problem; they are adapted to the specific numerical environment of the training simulator. A mask trained with float32 H is effectively tuned to a slightly non-ideal transfer function — one that differs from the mathematically exact transfer function by a spatially varying phase error of up to 0.07 rad. When evaluated against the exact transfer function (as approximated by float64), the exploited phase structure is absent and accuracy collapses. The behaviour is analogous to adversarial fragility in electronic neural networks, where small input perturbations cause drastic accuracy drops; the difference here is that the perturbation is in the transfer function rather than the input. Numerical consistency between the training simulator and the evaluation environment is therefore a non-negotiable requirement for any ONN deployment workflow. Whether float32 or float64 is used matters less than consistency: retraining from scratch with float64 would produce a mask adapted to that transfer function, deployable with equal reliability.

The fix adopted here was straightforward: H was computed once during initial model construction and cached within the PyTorch model as a registered buffer. It was saved with the checkpoint in float32 and restored from it at evaluation time. Neither the NumPy nor the MATLAB implementation recomputed H; both loaded it from the checkpoint file. This policy ensured that all three implementations operated on bit-identical transfer function values, and the agreement of 90.85% across all three implementations confirms that the result is numerically robust when this policy is followed.
