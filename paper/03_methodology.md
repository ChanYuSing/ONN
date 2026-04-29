 III. Methodology

 A. System Architecture

The ONN is a linear optical train: a 532 nm coherent source illuminates the 28 × 28 MNIST input [11] (upsampled to 140 × 140 and zero-padded to 200 × 200) which traverses two amplitude-modulating LCD layers separated by a 50 mm free-space gap; the output intensity field is read out by a CMOS camera. The full layout is shown in Fig. 1, and the hardware specifications are summarised in TABLE I.

**Fig. 1.**  System architecture of the two-layer LCD-based ONN. *(insert `report/figures/figure1_system_architecture.png`)*

**TABLE I**
*Hardware Specifications of the LCD Optical System*

| Parameter | Value |
|---|---|
| Laser wavelength | 532 nm |
| LCD resolution | 200 × 200 pixels |
| LCD active area | 27.66 × 27.66 mm |
| Pixel pitch Δx | 138.3 µm |
| Number of layers | 2 |
| Inter-layer spacing z | 50 mm |
| LCD type | Monochrome transmissive |

 B. Light Propagation Model

Propagation between layers is computed using the Angular Spectrum Method [3], which is exact within the scalar diffraction approximation at any propagation distance and angle. Each plane-wave component of the input field is propagated independently in Fourier space using the transfer function

$$H(k_x, k_y) = \exp\left(j z \sqrt{k^2 - k_x^2 - k_y^2}\right), \quad k_x^2 + k_y^2 \leq k^2 \qquad (1)$$

where k = 2π/λ and evanescent components (k_x² + k_y² > k²) are set to zero to suppress numerical blow-up. The discrete grid satisfies the ASM sampling criterion z ≤ Δx²/λ = 35.9 m by a wide margin at z = 50 mm. Compared with the Fresnel integral, ASM costs one extra FFT per layer — negligible on a GPU — and remains accurate at all angles, justifying its selection over the paraxial Fresnel form (TABLE II).

**TABLE II**
*Comparison of Light Propagation Methods*

| Method | Near-field valid | Accuracy | Cost |
|---|---|---|---|
| Fresnel integral | No | Approximate | 1 FFT |
| Angular Spectrum | Yes | Exact (scalar) | 2 FFTs |

 C. Amplitude Modulation and LCD Selection

Each layer applies pixel-wise amplitude modulation U_out(x, y) = U_in(x, y) · T(x, y) with T ∈ [0, 1]. T is parameterised as T = σ(r) where r is a free real tensor and σ is the sigmoid function, eliminating the need for explicit clamping during optimisation. Monochrome LCD was selected over phase SLMs (> USD 10,000), DMDs (binary-only, > USD 5,000), and 3D-printed phase masks (non-reconfigurable) on cost (< USD 50), continuous-amplitude programmability, and electronic reconfigurability without disturbing the optical alignment.

Real LCDs deviate from the ideal model in three ways. **Fill factor**: only the central fraction F of each pixel transmits light:

$$T_{\text{eff}}(x,y) = T(x,y) \cdot \mathrm{ff}(x,y), \quad \mathrm{ff}(x,y) = \begin{cases} 1 & \text{if } (x/\Delta x) \bmod 1 \in [(1{-}F)/2,(1{+}F)/2] \\ 0 & \text{otherwise} \end{cases} \qquad (2)$$

**Contrast limits**: T_clipped = T_min + T·(T_max − T_min). **Binary quantisation** (on/off LCDs): a straight-through estimator passes the hard threshold in the forward pass while propagating gradients through the continuous amplitude. The framework activates these constraints when physical characterisation data are available; the present results assume continuous, full-contrast amplitude.

 D. Two-Layer Design and Detection Schemes

A single amplitude layer performs only element-wise scaling. Two layers separated by free-space propagation introduce diffraction-based spatial mixing sufficient for MNIST classification. The two-layer choice was driven by the hardware budget: each additional layer adds an alignment-sensitive gap and one more LCD; two layers represent the minimum-risk path to a functional first prototype. Five detection architectures, all designed for trivial-cost physical read-out, were evaluated.

- **Direct zone**: 2 × 5 fixed grid; each zone intensity is treated as the corresponding class logit. Requires 10 photodiodes; no learnable detector parameters.
- **Centre-based (3 × 3)**: 9 zones feed a learnable 9 × 10 linear projection.
- **Binary encoding (3 × 3)**: 9 soft-thresholded zones index a learnable 2⁹ → class lookup table; at inference the hard 9-bit pattern is read by an EEPROM lookup.
- **Max-zone (M × N)**: each zone is assigned to one of 10 classes via a learnable logit matrix W ∈ ℝ^((M·N) × 10); inference reports the class of the brightest zone — a single max-detection over photodiodes. Evaluated at M × N = 4 × 4 and 5 × 5.

For the max-zone detector, hard argmax during training is replaced by Gumbel-Softmax with a temperature τ coupled to the learning rate, τ = 0.5 + 1.5·(η/η₀). Zone intensities are scaled by a learnable factor s (initialised to 10.0) so that the optical signal dominates Gumbel noise. To pre-empt zone collapse, the logit matrix W is initialised round-robin so that every class holds at least one zone before training begins.

 E. Training Procedure

Each MNIST input is energy-normalised at load time:

$$x_i' = x_i \cdot \frac{\bar{E}}{E_i + \delta}, \quad E_i = \sum_p x_i[p], \quad \delta = 10^{-8} \qquad (3)$$

where Ē is the mean total intensity over the training set. This step is physically motivated: a uniform laser delivers the same total power to every input, so the class-dependent brightness present in raw MNIST is an artefact of the dataset and would otherwise be exploited by the detector as a spurious cue.

All parameters are optimised with AdamW [12] in two parameter groups: amplitude masks at η_mask = 0.01 with zero weight decay (the sigmoid already constrains T ∈ [0, 1]); detector parameters at η_det = 0.05 with weight decay 0.001. Learning rates are linearly warmed up from 1% over 10 epochs, then reduced by a factor of 0.5 by a `ReduceLROnPlateau` scheduler on validation accuracy with patience 10 epochs. Training terminates by early stopping after 30 epochs without improvement (max 500). The loss is cross-entropy with label smoothing ε = 0.1; gradient norm is clipped at 1.0. Batch size is 64.

 F. Software and Forward Pass

The simulation, training, and evaluation pipeline is implemented in PyTorch [13] across five Python modules: `config.py` (hyperparameters), `physics.py` (stateless ASM and LCD-effect functions), `model.py` (the `OpticalNeuralNetwork` module composing two `AmplitudeLayer` instances and one of five detector classes), `data.py` (MNIST loading, upsampling, padding), and `train.py` (training loop, checkpoint, CSV logging). The differentiable forward pass is

$$U_0 \rightarrow \text{AmpLayer}_1 \rightarrow \text{prop}(H) \rightarrow \text{AmpLayer}_2 \rightarrow \text{prop}(H) \rightarrow |\cdot|^2 \rightarrow \text{Detector} \rightarrow \text{logits} \qquad (4)$$

The transfer function H is computed once at model construction and cached as a non-learnable PyTorch buffer, ensuring that H is saved with and restored bit-identically from every checkpoint. The significance of this caching policy is examined in §IV.E.
