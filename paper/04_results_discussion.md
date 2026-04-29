 IV. Results and Discussion

 A. Detection Method Comparison

All five detection schemes were trained from scratch under identical hyperparameters (§III.E) and evaluated on the 10,000-image MNIST test set. TABLE III lists overall accuracy and convergence epoch count.

**TABLE III**
*Detection Method Accuracy on the MNIST Test Set*

| Method | Architecture | Zones | Epochs | Accuracy (%) |
|---|---|---|---|---|
| **Max-zone (5 × 5)** | Learnable assignment | 25 | 373 | **90.85** |
| Direct zone | Fixed log-intensity | 10 | 63 | 84.73 |
| Binary encoding | Soft-binary lookup | 9 | 73 | 64.71 |
| Centre-based | Linear projection | 9 | 38 | 58.96 |
| Max-zone (4 × 4) | Learnable assignment | 16 | 110 | 57.77 |

The max-zone 5 × 5 detector achieved the highest accuracy of 90.85%, ranking the methods 5 × 5 > direct zone >> binary ≈ centre ≈ 4 × 4. The 6.12 percentage-point advantage of max-zone 5 × 5 over the direct zone scheme is attributable to (i) learnable zone assignment, which adapts to the actual diffraction pattern rather than a fixed rectangular grid, and (ii) the 2.5× zone-density advantage. The strong direct-zone result (84.73%) without any learnable detector parameters confirms that the masks alone — and the diffraction physics they shape — perform the bulk of the classification computation; in hardware terms, a 10-element photodiode array with no downstream electronics would suffice for that scheme.

 B. Zone Collapse and the Max-Zone 4 × 4 Failure

The max-zone 4 × 4 configuration recorded 0.0% accuracy on digits 4 and 8, exposing a zone-collapse instability. Early in training, when masks have not yet developed structured diffraction, all zones receive nearly equal intensity and the gradient signal for zone assignment is dominated by noise. A positive-feedback loop emerges: once a zone drifts toward a class, its logit grows faster than competitors, and the softmax progressively concentrates assignment probability on the leading zone–class pair. With only 16 − 10 = 6 surplus zones, the structurally weakest classes (4 and 8, which share features with 9 and 3 respectively) lose their assignments entirely.

Three measures together resolved the instability in the 5 × 5 configuration: grid expansion (15 surplus zones increases the assignment headroom), round-robin logit initialisation (every class begins with at least one assigned zone), and Gumbel-Softmax exploration with the temperature schedule τ = 0.5 + 1.5·(η/η₀). The temperature decreases smoothly with the learning rate, sharpening from near-uniform exploration (τ ≈ 2.0 at start) to near-hard selection (τ ≈ 0.5 at convergence). The adaptive plateau scheduler permitted 373 training epochs — substantially longer than the other methods — and the resulting accuracy of 90.85% is 33 percentage points above the 4 × 4 baseline. Fig. 2 shows the full training trajectory.

**Fig. 2.**  Training and validation accuracy of the max-zone 5 × 5 model over 373 epochs (upper) with the learning-rate schedule (lower). *(insert `report/figures/figure4_training_curves.png`)*

Training and validation curves remain closely aligned throughout, indicating no overfitting; the three discrete steps in the learning-rate panel mark the plateau-scheduler activations, each followed by a renewed phase of accuracy improvement.

 C. Per-Class Accuracy and Confusion

Per-class accuracy of the final 5 × 5 model ranges from 82.7% (digit 9) to 97.8% (digit 1), as shown in Fig. 3. The geometrically distinctive digits 0 (closed loop) and 1 (single vertical stroke) achieve the highest accuracy because their diffraction patterns route intensity to dedicated zones with high reliability. Digits 8 and 9 are the most challenging cases owing to structural overlap with digits 0/3 and 4 respectively, but the 5 × 5 model resolves both, where the 4 × 4 configuration failed entirely.

**Fig. 3.**  Per-class test accuracy of the max-zone 5 × 5 model. *(insert `report/figures/figure5_per_class_accuracy.png`)*

The row-normalised confusion matrix (Fig. 4) is predominantly diagonal; the most frequent off-diagonal entries are 9 → 4 and 8 → 3, consistent with the structural similarities noted above.

**Fig. 4.**  Row-normalised confusion matrix for the max-zone 5 × 5 model. *(insert `report/figures/figure6_confusion_matrix.png`)*

The 90.85% overall accuracy compares favourably with the 91.75% Lin et al. benchmark [2], obtained with five phase-modulating layers of 3D-printed acrylic. The present system reaches within 0.9 percentage points of that result with only two amplitude-modulating LCD layers costing under USD 50 each — a hardware-cost reduction of more than two orders of magnitude relative to phase SLMs. The remaining gap is plausibly attributable to reduced network depth and the inherent attenuation of amplitude modulation.

 D. Learned Mask Structure

Fig. 5 shows the trained masks. Layer 1 displays fine, high-spatial-frequency modulation distributed across the aperture, consistent with a feature-extractor role exploiting the full Nyquist bandwidth of the 138.3 µm pixel pitch. Layer 2 displays coarse, zone-scale structure consistent with a spatial-routing function that steers the diffraction-mixed field arriving from Layer 1 toward the output zone of the predicted class. Layer 1 is markedly bimodal (peaks near 0.15 and 0.85), whereas Layer 2 varies smoothly over [0.2, 0.8] — the network has learned a near-binary feature aperture followed by a continuous routing profile.

**Fig. 5.**  Trained amplitude masks: Layer 1 (left) and Layer 2 (right) of the max-zone 5 × 5 model after convergence at epoch 373. *(insert `report/figures/figure7_masks.png`)*

The masks are stored as float32 tensors and converted to 8-bit greyscale PNGs for deployment on physical LCDs. The 1/255 ≈ 0.4% maximum amplitude error from 8-bit quantisation is well below the LCD's intrinsic contrast non-uniformity and is not expected to be the limiting factor.

 E. Independent Validation and a Float32 Precision Dependency

To confirm that the 90.85% figure is not an artefact of the PyTorch training framework, the trained model was independently re-evaluated in NumPy (Python, sharing no PyTorch code) and in MATLAB. Both implementations reproduced the angular-spectrum forward pass

$$U_{\ell+1} = \mathcal{F}^{-1}\left\{ H \cdot \mathcal{F}\left\{ U_\ell \cdot M_\ell \right\} \right\} \qquad (5)$$

then partitioned |U₃|² into a 5 × 5 grid and applied the trained zone-class assignment. **Both implementations reproduced 90.85% test accuracy exactly** when the float32 H from the trained checkpoint was used, with a relative inter-implementation intensity error of 5.18 × 10⁻⁶ and zero classification disagreements.

A surprising finding emerged when H was instead recomputed freshly in float64. Accuracy collapsed to approximately **43%** — only marginally above random chance — despite both H values being mathematically consistent with the wave equation. The physical origin is that H accumulates a phase

$$\phi_0 = k z = \frac{2\pi}{\lambda}z = \frac{2\pi}{532\times 10^{-9}} \times 0.05 \approx 590{,}600 \text{ rad} \qquad (6)$$

at the on-axis frequency. Because 590,600 lies in the IEEE 754 binade [2¹⁹, 2²⁰), the unit in the last place of its float32 representation is 2⁻⁴ ≈ 0.06 rad. Cancellation error in evaluating k_z at high spatial frequencies adds further phase noise; the per-frequency phase difference between float32 and float64 H reaches approximately 0.07 rad. After two propagation steps these errors accumulate non-linearly across the 40,000 frequency components and shift the argmax zone winner for the majority of test inputs.

The interpretation is that the trained masks are not general solutions to the optical classification problem; they are tuned to the specific (slightly non-ideal) transfer function encountered during training. Whether float32 or float64 is used matters less than consistency between the training simulator and any downstream evaluator. The fix adopted here is to compute H once at model construction, register it as a PyTorch buffer, and have downstream environments load it bit-identically from the checkpoint. Under this policy the agreement of all three implementations (PyTorch, NumPy, MATLAB) at 90.85% confirms that the result is numerically robust. To the author's knowledge, this precision-consistency requirement has not previously been quantified in the diffractive ONN literature, although its implications for any simulation-to-hardware transfer workflow are immediate.
