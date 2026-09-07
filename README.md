# inverse-double-slit

A Deep Learning prototype for inverse reconstruction of physical parameters from two-slit quantum interference intensity patterns.

## Overview

This project explores an inverse modeling problem using a simplified two-slit interference model.

A parameterized forward model generates an interference intensity distribution from a set of hidden physical parameters. A neural network then learns the inverse mapping from the observed intensity pattern back to the underlying parameters.

The project is intended as a small-scale Scientific Machine Learning (SciML) prototype connecting physical-system simulation, synthetic data generation, and deep learning-based inverse reconstruction.

## Physical Model

The two slit contributions are represented as complex-valued Gaussian amplitudes:

\[
\psi_L(x) =
A_L G(x;x_L,\sigma_L)
\]

\[
\psi_R(x) =
A_R G(x;x_R,\sigma_R)e^{i\Delta\phi}
\]

where:

- \(x_L, x_R\) are the slit positions.
- \(\sigma_L, \sigma_R\) control the spatial widths.
- \(A_L, A_R\) are amplitude parameters.
- \(\Delta\phi\) is the relative phase.

The observed interference intensity is calculated as:

\[
I(x)=|\psi_L(x)+\psi_R(x)|^2
\]

The model uses a canonical left/right ordering:

\[
x_L < x_R
\]

to remove the permutation ambiguity between the two slit components.

## Inverse Problem

The neural network receives a sampled intensity distribution \(I(x)\) and learns to reconstruct the underlying model parameters.

The current target representation is based on:

\[
[x_L,\sigma_L,A_L,x_R,\sigma_R,A_R,\cos(\Delta\phi)]
\]

The phase is represented through its cosine because, for this simplified real-valued Gaussian interference model, the intensity depends on the relative phase through:

\[
\cos(\Delta\phi)
\]

rather than uniquely determining the sign of the phase.

## Data Generation

The dataset is generated synthetically from the forward model.

Randomized parameters include:

- slit positions
- Gaussian widths
- amplitudes
- relative phase

The generated intensity patterns are used as inputs to the inverse neural model.

Target parameters are scaled using their known parameter ranges during training and transformed back to their original physical ranges for evaluation.

## Model

The current implementation uses a multilayer fully connected neural network with:

- Dense layers
- ReLU activations
- Batch Normalization
- Dropout
- Adam optimization
- Mean Squared Error loss

The implementation currently uses TensorFlow/Keras.

## Evaluation

Model performance is evaluated in the original parameter units using per-parameter Mean Absolute Error (MAE).

The reconstructed parameters can also be passed through the forward model to regenerate an estimated interference pattern, allowing direct comparison between the original and reconstructed intensity distributions.

## Identifiability Analysis

Before training the neural network, numerical inverse fitting experiments were performed from multiple initial conditions to examine whether the underlying parameters can be recovered from the intensity distribution.

These experiments revealed an important symmetry in the simplified model:

\[
\cos(\phi)=\cos(-\phi)
\]

Therefore, the sign of the relative phase cannot be uniquely recovered from intensity alone in this formulation.

This motivates using identifiable parameter representations rather than forcing the model to predict quantities that are not uniquely determined by the measurement.

## Current Status

This repository contains an initial research and engineering prototype.

The project is being developed toward a more complete Scientific Machine Learning pipeline, including:

- PyTorch implementation
- improved inverse-model experiments
- expanded evaluation and error analysis
- inference API
- interactive visualization interface

## Project Motivation

This project serves as an initial exploration of inverse neural modeling for physical systems.

The broader motivation is to investigate how Deep Learning can be used to construct computational models that learn inverse mappings or surrogate representations of complex physical processes.

## Future Directions

Planned extensions include:

- migration to PyTorch
- improved parameter-recovery experiments
- uncertainty and robustness analysis
- physics-based reconstruction diagnostics
- interactive inference interface
- deployment as an end-to-end application

## Technologies

- Python
- NumPy
- TensorFlow / Keras
- SciPy
- Matplotlib
- scikit-learn

## Author

Duaa Shaheen
