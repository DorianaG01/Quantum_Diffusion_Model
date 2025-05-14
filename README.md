# Quantum Diffusion Model Implementation

This code implements a Quantum Diffusion Model (QDM) based on the paper by Cacioppo et al. for the Quantum Diffusion Models course exam.

The model uses parameterized quantum circuits (PQCs) in place of traditional artificial neural networks to generate quantum states. The implementation presents a hybrid approach, with a classical forward process and a quantum backward process, using the Reverse-Bottleneck architecture. A latent version of the model has been created, with conditioning capabilities. The model was implemented using PennyLane and tested on the MNIST dataset.

Paper: [https://arxiv.org/abs/2311.15444](https://arxiv.org/abs/2311.15444)

## The quantum diffusion model implementation includes:

- **Forward Process**: A classical autoencoder with complex Gaussian noise
- **Backward Process**: A parameterized quantum circuit (PQC) using PennyLane
- **Reverse-Bottleneck Architecture**: A three-block quantum circuit design with data qubits and ancilla qubits
- **Latent Model**: Dimensionality reduction with a classical autoencoder followed by quantum diffusion modeling
- **Conditioning Support**: Ability to generate class-conditioned samples
- **Evaluation Metrics**: Including FID, Wasserstein distance, and ROC-AUC
- **MNIST Processing**: Functionality to preprocess and use MNIST data

## Code Structure:

The code structure is organized into logical sections that follow the paper's methodology:
- Data input preparation
- Forward diffusion process (classical)
- Backward diffusion process (quantum)
- Training and sampling procedures
- Evaluation
- Visualization

## Usage

To run the model:
```bash
python quantum_diffusion_model.py
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
```


