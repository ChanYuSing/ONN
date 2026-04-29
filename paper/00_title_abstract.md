 An LCD-Based Optical Neural Network for Handwritten Digit Classification

Chan Yu Sing

Department of Electrical and Electronic Engineering, The University of Hong Kong, ysing@connect.hku.hk

---

***Abstract*** *—* **This paper presents the design, simulation, training, and independent numerical validation of a low-cost diffractive optical neural network (ONN) implemented with two amplitude-modulating liquid-crystal display (LCD) layers for handwritten digit classification on MNIST. Each layer is a 200 × 200 pixel monochrome LCD with 138.3 µm pixel pitch, illuminated by a 532 nm laser with 50 mm inter-layer spacing. Field propagation is computed using the Angular Spectrum Method in a differentiable PyTorch framework. Five output detection schemes were trained and evaluated under identical conditions; a max-zone 5 × 5 detector trained with Gumbel-Softmax assignment, round-robin initialisation, and adaptive plateau scheduling achieved 90.85% test accuracy — within 0.9 percentage points of the 91.75% benchmark of Lin et al. obtained with five phase-modulating layers, at over two orders of magnitude lower hardware cost. The trained model was independently reproduced in NumPy and MATLAB, both yielding 90.85% when the float32 angular spectrum transfer function from the trained checkpoint was reused. Recomputing the transfer function in double precision collapses accuracy to approximately 43%, isolating a precision-consistency requirement that, to the author's knowledge, has not previously been quantified in the diffractive ONN literature. The findings establish commodity LCDs as a viable low-cost platform for diffractive ONN prototyping and identify physical hardware assembly as the principal remaining objective.**

***Index Terms*** *—* Angular Spectrum Method, Diffractive Neural Network, Liquid-Crystal Display, MNIST Classification, Optical Computing.
