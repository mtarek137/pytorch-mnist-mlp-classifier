# PyTorch MNIST Handwritten Digit Classifier (MLP)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~98%25-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A modular, end-to-end Multi-Layer Perceptron (MLP) built from scratch using raw **PyTorch** to classify handwritten digits from the classic **MNIST** dataset. 

This repository demonstrates complete PyTorch fundamentals: dynamic GPU acceleration, custom `nn.Module` architecture design, automated mini-batch pipelines via `DataLoader`, gradient-based optimization loops, real-time validation tracking, and model weight serialization.

---

## 📌 Key Highlights

- **Custom Feedforward Architecture:** Fully connected 2-layer MLP with non-linear ReLU activation.
- **Dynamic Hardware Acceleration:** Automatic CUDA GPU / Apple Silicon MPS / CPU device mapping.
- **Iterative Validation Tracking:** Tracks and logs loss and validation accuracy per epoch.
- **Visual Performance Diagnostics:** Generates side-by-side training loss curves, test accuracy trajectories, and sample prediction grids with visual error highlighting.
- **Model Serialization:** Exports model parameters (`state_dict`) for lightweight deployment and instant inference.

---

## 🏗️ Model Architecture

The neural network takes a flattened $28 \times 28$ grayscale image (784 features) and outputs unnormalized log-probabilities (logits) across 10 output classes ($0\text{--}9$):
