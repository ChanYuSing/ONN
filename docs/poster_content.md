# Poster Content Reference
# "Computing with Diffractions" — ELEC4848 SDP — Chan Yu Sing — HKU 2026

---

## IMAGES TO USE

| Ref | File path | Actual pixels | Actual ratio | Recommended width | Resulting height |
|---|---|---|---|---|---|
| **Header** | `Poster Header.jpg` | 4723 × 1063 px | 4.44 : 1 | 800 mm (full width) | 180 mm |
| **Fig 1** | `report/figures_poster/poster_fig1_system_architecture.png` | 5349 × 1950 px | **2.47 : 1** | 370 mm | 150 mm |
| **Fig 2** | `report/figures_poster/poster_fig2_propagation.png` | 10719 × 1431 px | **5.13 : 1** | 700 mm | 136 mm |
| **Fig 3** | `report/figures_poster/poster_fig3_zone_assignment.png` | 2976 × 2654 px | **0.97 : 1** | 250 mm | 258 mm |
| **Fig 5** | `report/figures_poster/poster_fig5_confusion_matrix.png` | 4140 × 3708 px | **1.01 : 1** | 240 mm | 238 mm |

> **Canva tip:** Always set width first, keep aspect ratio lock ON, never set both dimensions manually.

---

## SECTION A — TITLE BLOCK

**Main title (large, bold):**
> Computing with Diffractions

**Subtitle (smaller, italic):**
> An LCD-Based Optical Neural Network for Handwritten Digit Classification

**Author line (smallest):**
> Chan Yu Sing · Supervisor: Prof. Kenneth K.Y. Wong · ELEC4848 · The University of Hong Kong · 2026

---

## SECTION B — THE INSIGHT

**Hook line (bold, large, centred across all 3 columns):**
> **What if a laser could replace a GPU?**

**Column 1 heading:** The Problem

Training and running large neural networks is increasingly energy-intensive. Data centres are projected to consume 1,000 TWh of electricity per year by 2026 [13] — comparable to the entire annual output of several power plants. The core bottleneck is not just energy: conventional computers process data sequentially, and memory bandwidth constrains how fast information can flow between storage and computation. Every neural network inference requires billions of transistor switches to fire, all consuming power and generating heat.

**Column 2 heading:** The Physics

Light offers a fundamentally different computing substrate. A coherent beam travels at 3 × 10⁸ m/s and diffracts through apertures according to well-understood wave physics. When a 200×200 LCD modulates the amplitude of a beam, it performs 40,000 independent multiplications simultaneously — in a single exposure, passively, with no electronic switching at all. The "computation" happens as the wavefront propagates through space.

**Column 3 heading:** Why LCD?

This work uses an off-the-shelf LCD panel — cheap enough for a student lab, fast enough to reprogram in seconds.

| Technology | Cost/layer | Reconfigurable |
|---|---|---|
| 3D-printed mask | < $10 | ✗ |
| Phase SLM (LCOS) | > $10,000 | ✓ |
| **LCD (this work)** | **< $50** | **✓** |

**Contribution statement (bridge into Section C):**
> This project demonstrates that a fully reconfigurable optical neural network built from consumer LCD panels — costing under USD 100 in total — can classify handwritten digits with 90.85% accuracy.

---

## SECTION C — SYSTEM ARCHITECTURE

*Place Fig 1 on the left side — set width 370 mm, height will follow at 150 mm (ratio 2.47:1). Do not stretch.*

The key insight is that no computation happens electronically during the optical forward pass. Once the laser illuminates the first LCD, the physics of wave diffraction carries the information through space. The second LCD mask applies a second learned transformation. By the time light reaches the detector, the digit classification is encoded in the intensity pattern — readable with a photodetector array.

**Table heading:** Hardware Specifications

| Parameter | Value |
|---|---|
| Laser wavelength | 532 nm |
| LCD resolution | 200 × 200 pixels |
| Pixel pitch | 138.3 µm |
| Inter-layer spacing | 50 mm |
| Detection method | Max-zone 5×5 |
| Cost per LCD layer | < USD 50 |

*(Caption is baked into the PNG — do not add a separate caption text box in Canva.)*

---

## SECTION D — PROPAGATION STRIP

*Place Fig 2 here — set width 700 mm, height will follow at 136 mm (ratio 5.13:1). Centre horizontally on the navy strip. Do not stretch.*

**Section heading (white text):** Watching the Computation Happen

*(Caption is baked into the PNG — do not add a separate caption text box in Canva.)*

---

## SECTION E — DETECTION ALGORITHM

*Place Fig 3 here — set width 250 mm, height will follow at 258 mm (ratio 0.97:1). Nearly square — do not stretch.*

**Section heading:** Detection Algorithm

After propagating through both masks, the light arrives at a flat detector plane as a 2D intensity image. The detector is divided into a 5×5 grid of 25 zones, each assigned to one of the ten digit classes (with some classes sharing zones). After the input image propagates through both LCD masks, the average light intensity within each zone is measured. The digit class corresponding to the zone with the highest average intensity is taken as the network's prediction. This max-zone readout requires no digital computation — it can in principle be implemented with a simple photodetector array.

*(Caption is baked into the PNG — do not add a separate caption text box in Canva.)*

---

## SECTION F — CLASSIFICATION PERFORMANCE

**Section heading:** Classification Performance

**Giant number (biggest text on the poster):**
> 90.85%

**Sub-label (smaller, grey):**
> test accuracy on 10,000 MNIST images

**Highlighted comparison line (bold, prominent):**
> Matches the 5-layer phase benchmark [1] at 91.75% — using 2 layers and hardware costing 200× cheaper.

**Verification badge (teal/blue):**
> ✓ Independently verified in NumPy and MATLAB

**Body text:**

The result was independently reproduced in both NumPy and MATLAB from scratch to rule out implementation artefacts. Both reproductions agree with the PyTorch [11] training result to within 5 × 10⁻⁶ relative error.

**Disclaimer (small italic grey):**
> Note: results obtained in simulation; physical hardware assembly is the next step.

*Place Fig 5 here — set width 240 mm, height will follow at 238 mm (ratio 1.01:1). Place below the text block above. Do not stretch.*

> ⚠️ Fig 5 is nearly square. At 240 mm wide it is 238 mm tall. Place it below the 90.85% number, comparison line, and badges — not beside them — or it will overflow the column.

*(Caption is baked into the PNG — do not add a separate caption text box in Canva.)*

---

## SECTION G — CONCLUSIONS

**Section heading:** Conclusions

**Comparison table:**

| System | Layers | Modulation | Accuracy |
|---|---|---|---|
| Lin et al. [1] | 5 | Phase | 91.75% |
| **This work** | **2** | **Amplitude (LCD)** | **90.85%** |

**Body text:**

Optical computing has attracted renewed interest as an energy-efficient pathway to AI inference [2]. This work demonstrates that a two-layer amplitude-modulation Optical Neural Network (ONN) built from consumer LCD panels can match the performance of significantly more complex and expensive phase-based systems. A complete two-layer ONN system — laser, two LCD panels, detector — can be assembled for under USD 100 and retrained for a new task in hours using AdamW optimisation [12], without replacing any hardware. Because the masks are stored digitally and applied electronically, the system can be reprogrammed in seconds — no physical fabrication is needed.

Future directions include extending to three layers, evaluating on FashionMNIST, and completing physical hardware assembly to validate simulation predictions against real optical measurements.

**Future work line:**
> Future: 3-layer extension · FashionMNIST · Physical hardware assembly

---

## SECTION G (right side) — KEY FINDING BOX

*Amber/yellow background wash, no strong border*

**Box title (bold amber):**
> Key Finding: Numerical Precision Matters

**Box body:**

Recomputing the wave propagation kernel H (Angular Spectrum Method, ASM [9]) in float64 collapses accuracy from 90.85% to ~43% — despite identical mask weights. The network had learned to rely on float32 rounding artefacts baked in during training. Fix: always load H from the original training checkpoint; never recompute it independently.

---

## SECTION H — REFERENCES

> [1] X. Lin et al., "All-optical machine learning using diffractive deep neural networks," *Science*, vol. 361, pp. 1004–1008, 2018.
>
> [2] G. Wetzstein et al., "Inference in AI with deep optics and photonics," *Nature*, vol. 588, pp. 39–47, 2020.
>
> [9] J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H. Freeman, 2017.
>
> [11] A. Paszke et al., "PyTorch: High-Performance Deep Learning Library," *NeurIPS*, vol. 32, 2019.
>
> [12] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," *ICLR* 2019.
>
> [13] International Energy Agency, "Electricity 2024: Analysis and Forecast to 2026," IEA, Jan. 2024.

---

## COLOUR GUIDE

| Name | Hex | Use |
|---|---|---|
| HKU Red | `#C00000` | Title background, section header text, left border accents |
| Dark Navy | `#1A1A2E` | Propagation strip background |
| Amber | `#FFF3CD` | Key Finding box background |
| Amber text | `#8B6914` | Key Finding box title text |
| Near-Black | `#1A1A1A` | All body text, giant 90.85% |
| Teal | `#0277BD` | Verification badge |
| Grey | `#555555` | Captions, references |
