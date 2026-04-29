# 1. Introduction

## 1.1 Background and Motivation

Machine learning inference has experienced explosive growth in computational demand. Modern neural networks require billions of multiply-accumulate operations per input, and large-scale deployment in data centres imposes substantial energy costs [2]. Graphics processing units consume several hundred watts per card for inference workloads at scale, and the International Energy Agency projected that global data centre electricity consumption would reach 1,000 TWh per year by 2026 [13]. This trajectory has renewed interest in alternative computing substrates that can perform inference with lower power consumption and higher throughput than transistor-based implementations.

Optical computing presents one such alternative. Coherent light propagates at c = 3 × 10⁸ m/s, and diffractive optical elements can implement complex linear transformations in parallel across the full spatial extent of an optical wavefront. A single 200 × 200 pixel spatial light modulator performs 40,000 parallel multiplications per exposure — the equivalent of a 40,000-dimensional vector multiplication — at the speed of light and with no dynamic power consumption beyond the illumination source. The computation occurs passively as light travels between layers, requiring no active switching elements. Lin et al. [1] demonstrated the feasibility of this paradigm by training Diffractive Deep Neural Networks (D²NNs) comprising 3D-printed phase masks achieving 91.75% accuracy on handwritten digit classification. Their work established the theoretical basis for treating physical diffraction as a computational primitive and showed that backpropagation through a differentiable wave-optics simulation is an effective method for designing diffractive optical processors.

However, 3D-printed phase masks cannot be reprogrammed after fabrication, which limits their utility during system development and prevents updating the network after deployment as data distributions shift. Spatial light modulators (SLMs) capable of phase modulation are electronically reprogrammable but cost in excess of USD 10,000 per unit, placing them out of reach of most research groups. Commercial liquid-crystal displays (LCDs) offer a programmable, low-cost alternative: a monochrome 200 × 200 pixel LCD costs under USD 50 and, being a transmissive electro-optic device, can be updated between exposures without mechanical realignment. This project investigated whether such commodity hardware can implement a functional ONN for digit classification, establishing a low-cost entry point to experimental optical computing research.

## 1.2 Project Objectives

This project aimed to design, simulate, train, and numerically validate an LCD-based ONN for MNIST handwritten digit classification. The specific objectives were as follows:

1. Develop a differentiable simulation framework implementing the Angular Spectrum Method (ASM) for accurate optical field propagation between LCD layers. The framework was required to support automatic differentiation through the full wave propagation pipeline so that mask patterns could be optimised by standard gradient-based methods.
2. Model realistic LCD characteristics, including pixel fill factor, contrast ratio limits, and binary quantisation constraints, to ensure the simulation results are physically meaningful and transferable to real hardware.
3. Train amplitude masks for two diffractive layers via gradient-based optimisation and compare multiple output detection schemes to identify the architecture best suited to physical hardware read-out, with minimal post-optical computation at the detector.
4. Validate the trained simulation independently using a second numerical implementation to confirm that results are reproducible and not artefacts of the PyTorch training framework, and to identify any numerical precision issues that would affect a hardware deployment.

Objectives 1–4 were fully achieved in simulation. Physical hardware assembly, originally envisaged as a fifth objective, was not completed within the project period and is identified as the primary direction for future work in Section 6.

## 1.3 Report Organisation

The remainder of this report is structured as follows. Section 2 surveys related work in diffractive neural networks and optical computing hardware. Section 3 describes the system architecture and training methodology, including justifications for key engineering choices. Section 4 presents results from the five-method detection comparison and analyses the max-zone training dynamics that achieved 90.85% test accuracy. Section 5 describes the independent numerical validation in NumPy and MATLAB and discusses the float32 precision dependency identified during that process. Section 6 concludes the report with a discussion of significance, limitations, and future directions. The findings from each section build cumulatively toward the overall assessment presented in Section 6.

## 1.4 Project Planning

Table 1 summarises the project timeline against the planned schedule. All simulation and software objectives were completed on schedule. Physical hardware assembly was scoped as an aspirational final phase, contingent on completing the simulation validation; with validation concluding in March 2026, assembly was deferred to future work.

**Table 1: Project milestone schedule.**

| Phase | Planned period | Key deliverables | Status |
|---|---|---|---|
| Literature review | Sep–Oct 2025 | Summary of D²NN, ONN, ASM literature | Complete |
| Framework development | Nov–Dec 2025 | Differentiable ASM in PyTorch; `AmplitudeLayer`; initial training loop | Complete |
| Midterm report | Jan 2026 | Midterm report submission (Jan 23) | Complete |
| Multi-method comparison | Feb 2026 | Five detection architectures trained; zone-collapse analysis | Complete |
| Independent validation | Mar 2026 | NumPy and MATLAB reproductions; float32 precision investigation | Complete |
| Report writing | Apr 2026 | Final report, figures, technical paper (due Apr 20) | In progress |
| Hardware assembly | Post-submission | Optical breadboard; LCD characterisation; physical MNIST evaluation | Future work |

The overall schedule was met for all completed phases. The hardware phase was not started within the project period and is acknowledged as the primary remaining objective in Section 6.
