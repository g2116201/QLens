# QLens
This repository contains the code for the developing quantum-inspired interpretability framework: "QLens: Towards A Quantum Perspective of Transformers."

**Abstract:** Current methods for understanding Transformers are successful at extracting intermediate output probability distributions during inference. However, these approaches function as limited diagnostic checkpoints, lacking a mathematical framework for modeling how each layer facilitates transitions between these distributions. This gap inspires us to turn to quantum mechanics, a field possessing a pre-built mathematical toolkit for describing the evolution of probability distributions from its study of stochastic particle measurements. We propose \framework a novel attempt to develop a physics-based perspective on the Transformer generation process. Under \framework, these neural networks are studied by converting their latent activations into a state vector in a Hilbert space derived from the model's output units. The evolution of this state through subsequent hidden layers is modeled with unitary operators and analogously defined Hamiltonians. To demonstrate \framework's potential, we conduct a proof-of-concept by probing three one-layer Transformers on common deployment tasks to investigate the influence of individual layers in model prediction trajectories. We present our work as a foundation for interdisciplinary methods to be leveraged towards a broader understanding of Transformers.

## Repository Layout
The contents of this repositiory are divided into three overarching folders, each of which contains code for one of the three datasets used to test QLens:
- `Sentihood`
- `Amazon_Books`
- `Tiny_Stories`
