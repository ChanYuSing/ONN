# TITLE PAGE

| | |
|---|---|
| **Course** | ELEC4848 Senior Design Project |
| **Department** | Department of Electrical and Electronic Engineering |
| **Faculty** | Faculty of Engineering |
| **University** | The University of Hong Kong |
| **Project Title** | An LCD-Based Optical Neural Network for Handwritten Digit Classification |
| **Student Name** | Chan Yu Sing |
| **Student ID** | 3035930345 |
| **Supervisor** | Prof. Kenneth K.Y. Wong |
| **Date of Submission** | 20 April 2026 |

---

# ABSTRACT

This report presents the design, simulation, training, and independent numerical validation of an LCD-based diffractive optical neural network (ONN) for handwritten digit classification on MNIST. The system comprises two amplitude-modulating layers, each realised as a 200 × 200 pixel monochrome liquid-crystal display with 138.3 µm pixel pitch, illuminated by a 532 nm laser with 50 mm inter-layer spacing. Field propagation is modelled using the Angular Spectrum Method in a differentiable PyTorch framework. Five detection architectures were trained under identical conditions; the max-zone 5 × 5 detector achieved 90.85% test accuracy — within 0.9 percentage points of the 91.75% benchmark of Lin et al. [1] using five phase-modulating layers and at over two orders of magnitude lower hardware cost. Results were independently reproduced in NumPy and MATLAB, confirming that the trained float32 angular spectrum transfer function must be preserved exactly to avoid a precision-induced accuracy collapse to approximately 43%. Hardware assembly and physical characterisation are identified as the primary remaining objectives.

---

# ACKNOWLEDGEMENT

The author wishes to express sincere gratitude to Prof. Kenneth K.Y. Wong for his expert supervision and guidance throughout this project. The author also acknowledges the support provided by the Department of Electrical and Electronic Engineering at The University of Hong Kong.

---

# TABLE OF CONTENTS

| Section | Title | Page |
|---------|-------|------|
| | Abstract | i |
| | Acknowledgement | ii |
| | Table of Contents | iii |
| | List of Figures | iv |
| | List of Tables | v |
| | Abbreviations | vi |
| | List of Symbols | vii |
| **1** | **Introduction** | **1** |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Project Objectives | 1 |
| 1.3 | Report Organisation | 2 |
| 1.4 | Project Planning | 2 |
| **2** | **Related Works** | **3** |
| 2.1 | Diffractive Deep Neural Networks | 3 |
| 2.2 | Amplitude- vs. Phase-Modulating Systems | 3 |
| 2.3 | Reviews and Broader Context | 4 |
| **3** | **Methodology** | **4** |
| 3.1 | System Architecture | 4 |
| 3.2 | Light Propagation Model | 5 |
| 3.3 | Amplitude Modulation and LCD Selection | 6 |
| 3.4 | Multi-Layer Diffractive Design | 6 |
| 3.5 | Detection Methods | 7 |
| 3.6 | Training Procedure | 8 |
| 3.7 | Software Architecture | 9 |
| **4** | **Results and Discussion** | **10** |
| 4.1 | Detection Method Comparison | 10 |
| 4.2 | Max-Zone Training Analysis | 11 |
| 4.3 | Per-Class Accuracy of the Final Model | 12 |
| 4.4 | Learned Amplitude Masks | 12 |
| **5** | **Independent Validation** | **13** |
| 5.1 | NumPy Validation | 13 |
| 5.2 | MATLAB Validation | 14 |
| 5.3 | Float32 Precision Dependency | 14 |
| **6** | **Conclusion** | **15** |
| 6.1 | Summary | 15 |
| 6.2 | Significance and Contributions | 15 |
| 6.3 | Limitations | 16 |
| 6.4 | Future Work | 16 |
| | References | 17 |

---

# LIST OF FIGURES

| Fig. | Caption | Page |
|--------|---------|------|
| Fig. 1 | System architecture of the two-layer LCD-based ONN | 4 |
| Fig. 2 | Max-zone 5 × 5 detector zone-to-class assignment map | 7 |
| Fig. 3 | Training and validation accuracy curves for the max-zone 5 × 5 model over 373 epochs, with the learning rate schedule shown in the lower panel | 11 |
| Fig. 4 | Per-class test accuracy for the max-zone 5 × 5 model | 12 |
| Fig. 5 | Row-normalised confusion matrix for the max-zone 5 × 5 model | 12 |
| Fig. 6 | Trained amplitude masks: Layer 1 (left) and Layer 2 (right) | 12 |

---

# LIST OF TABLES

| Table | Caption | Page |
|-------|---------|------|
| Table 1 | Project milestone schedule | 2 |
| Table 2 | Hardware specifications of the LCD optical system | 4 |
| Table 3 | Comparison of ASM and Fresnel propagation methods | 5 |
| Table 4 | Comparison of spatial light modulator technologies | 6 |
| Table 5 | Final training hyperparameters (identical for all detection methods) | 9 |
| Table 6 | Software module structure | 9 |
| Table 7 | Detection method accuracy comparison (MNIST test set) | 10 |
| Table 8 | Per-class test accuracy (%) for all detection methods | 10 |

---

# ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| ASM | Angular Spectrum Method |
| D²NN | Diffractive Deep Neural Network |
| DMD | Digital Micromirror Device |
| FFT | Fast Fourier Transform |
| LCOS | Liquid Crystal on Silicon |
| LCD | Liquid-Crystal Display |
| MNIST | Modified National Institute of Standards and Technology (handwritten digit dataset) |
| ONN | Optical Neural Network |
| SLM | Spatial Light Modulator |

---

# LIST OF SYMBOLS

| Symbol | Definition | Unit |
|---|---|---|
| λ | Optical wavelength | m |
| k | Free-space wavenumber, k = 2π/λ | rad m⁻¹ |
| k_x, k_y | Transverse wavenumber components | rad m⁻¹ |
| k_z | Longitudinal wavenumber component, k_z = √(k² − k_x² − k_y²) | rad m⁻¹ |
| z | Propagation distance between layers | m |
| H | Angular spectrum transfer function | — |
| U | Complex optical field amplitude | — |
| T | Pixel amplitude transmission (T ∈ [0, 1]) | — |
| M_ℓ | Amplitude mask tensor for layer ℓ | — |
| W | Zone-to-class assignment logit matrix | — |
| s | Learnable logit scale factor | — |
| τ | Gumbel-Softmax temperature parameter | — |
| η | Current learning rate | — |
| η₀ | Initial learning rate | — |
| ε | Label smoothing coefficient | — |
| δ | Numerical stabiliser (prevents division by zero) | — |
| Eᵢ | Total energy (sum of pixel intensities) of image i | — |
| Ē | Mean total energy over the training set | — |
| Δx | Pixel pitch | m |
| N | Grid dimension (number of pixels per side) | — |


