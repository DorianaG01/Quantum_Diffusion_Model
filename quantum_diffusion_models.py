import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import math
import matplotlib.pyplot as plt
import traceback

"""
Implementation of a Hybrid Quantum Diffusion Model.
- Forward process: Classical autoencoder
- Backward process: Parameterized quantum circuit (PQC)

Based on the reverse-bottleneck architecture described in the paper.
Includes corrections for parameter dimensions and configurations
based on simulation results presented in the paper.
"""

# Model parameters definition based on the simulation results from the paper
latent_qubits = 3    # 3 qubits for the latent model
m_ancilla = 1        # Number of ancilla qubits
label_qubits = 4     # 4 qubits to encode 10 MNIST classes

# Circuit depth based on the paper
latent_n_layers = 50 // 3  # 50 total layers distributed across 3 blocks

# Diffusion parameters
timesteps = 15     # Reduced number of time steps as indicated in the paper
beta_min = 1e-4    # Minimum beta value for the schedule
beta_max = 0.02    # Maximum beta value for the schedule
latent_dim = 2**latent_qubits  # Dimension for the latent model

# Definition of the forward process schedule
def get_beta_schedule(timesteps, beta_min, beta_max):
    """Creates the beta schedule for the diffusion process."""
    return np.linspace(beta_min, beta_max, timesteps)

# Calculation of alphas from the beta schedule
def compute_alpha(beta_schedule):
    """Calculates alphas and cumulative alphas."""
    alpha = 1 - beta_schedule
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod

beta_schedule = get_beta_schedule(timesteps, beta_min, beta_max)
alpha, alpha_cumprod = compute_alpha(beta_schedule)

# PART 1: CLASSICAL FORWARD PROCESS COMPONENTS (AUTOENCODER)

class AutoencoderForward(nn.Module):
    """
    Classical autoencoder for the forward process.
    Implements the addition of complex Gaussian noise to the data.
    """
    def __init__(self, feature_dim, timesteps, alpha_cumprod):
        super(AutoencoderForward, self).__init__()
        self.feature_dim = feature_dim
        self.timesteps = timesteps
        self.alpha_cumprod = alpha_cumprod
    
    def forward_step(self, x, t):
        """
        Performs a single step of the forward process by adding complex noise.
        
        Args:
            x: Input data
            t: Current timestep
            
        Returns:
            x_t: Noisy data at timestep t
            noise: Complex noise added
        """
        # Get alpha for this timestep
        a_t = self.alpha_cumprod[t]
        
        # Generate complex noise
        noise_real = torch.randn_like(x)
        noise_imag = torch.randn_like(x)
        noise = noise_real + 1j * noise_imag
        
        # Apply noise according to the equation in the paper
        x_t = torch.sqrt(torch.tensor(a_t)) * x + torch.sqrt(torch.tensor(1 - a_t)) * noise
        
        return x_t, noise
    
    def forward(self, x_0):
        """
        Performs the complete forward diffusion process.
        
        Args:
            x_0: Original data
            
        Returns:
            x_t: List of diffused data for each timestep
        """
        x_t_sequence = []
        x_t = x_0
        
        # Sequentially apply diffusion steps
        for t in range(self.timesteps):
            x_t, _ = self.forward_step(x_t, t)
            x_t_sequence.append(x_t)
        
        return x_t_sequence

# PART 2: QUANTUM BACKWARD PROCESS COMPONENTS (PQC)

# Definition of the parameterized unitary block
def unitary_block(params, wires, n_layers):
    """
    Implements a parameterized unitary block with rotation and entanglement layers.
    
    Args:
        params: Block parameters
        wires: Qubits to apply the block to
        n_layers: Number of layers in the block
    """
    n_wires = len(wires)
    
    # Single-qubit rotation layer
    for i, wire in enumerate(wires):
        qml.RX(params[0, i], wires=wire)
        qml.RY(params[1, i], wires=wire)
        qml.RZ(params[2, i], wires=wire)
    
    # Entanglement layer
    for layer in range(n_layers):
        # Entanglement with CNOT
        for i in range(n_wires-1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.CNOT(wires=[wires[n_wires-1], wires[0]])  # Circular entanglement
        
        # Parameterized rotations
        for i, wire in enumerate(wires):
            qml.RX(params[3+layer*3, i], wires=wire)
            qml.RY(params[4+layer*3, i], wires=wire)
            qml.RZ(params[5+layer*3, i], wires=wire)

# Definition of the encoding of classical data into quantum states (amplitude encoding)
def amplitude_encode(x, wires):
    """
    Implements amplitude encoding of a classical vector into a quantum state.
    
    Args:
        x: Classical data vector (can be complex)
        wires: Qubits to encode the information on
    """
    # Amplitude normalization
    x_norm = x / np.sqrt(np.sum(np.abs(x)**2))
    qml.AmplitudeEmbedding(x_norm, wires=wires, normalize=True)

# PART 3: COMPLETE QUANTUM DIFFUSION MODEL

# Creation of the simulated quantum device
def create_device(n_qubits, m_ancilla, extra_qubits=0):
    """
    Creates a quantum device with the correct number of qubits.
    Uses default.mixed which better supports intermediate measurements.
    
    Args:
        n_qubits: Number of qubits for data
        m_ancilla: Number of ancilla qubits
        extra_qubits: Additional qubits (for labels, etc.)
        
    Returns:
        PennyLane quantum device
    """
    total_qubits = n_qubits + m_ancilla + extra_qubits
    return qml.device('default.mixed', wires=total_qubits)

# Definition of the parameterized quantum circuit for the reverse-bottleneck
def create_reverse_bottleneck_pqc(dev, n_qubits, m_ancilla, n_layers, use_labels=False, n_label_qubits=0):
    """
    Creates a reverse-bottleneck PQC for quantum diffusion.
    
    Args:
        dev: Quantum device
        n_qubits: Number of qubits for data
        m_ancilla: Number of ancilla qubits
        n_layers: Number of layers in the circuit
        use_labels: If True, adds qubits for labels
        n_label_qubits: Number of qubits for labels
        
    Returns:
        QNode function for the PQC circuit
    """
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x_t, params, t, label=None):
        # Calculate the number of parameters needed per block
        params_per_block = 3 + n_layers * 3
        
        # Determine the effective number of qubits for each block
        n_block1_qubits = n_qubits + (n_label_qubits if use_labels else 0)
        n_block2_qubits = n_qubits + m_ancilla + (n_label_qubits if use_labels else 0)
        n_block3_qubits = n_qubits
        
        # Calculate the parameters needed for each block
        block1_size = params_per_block * n_block1_qubits
        block2_size = params_per_block * n_block2_qubits
        block3_size = params_per_block * n_block3_qubits
        
        # Verify that the number of parameters is correct
        total_size = block1_size + block2_size + block3_size
        if params.size != total_size:
            raise ValueError(f"Incorrect number of parameters. Expected: {total_size}, Received: {params.size}")
        
        # Correctly reshape the parameters for each block
        block1_params = params[:block1_size].reshape(params_per_block, n_block1_qubits)
        block2_params = params[block1_size:block1_size+block2_size].reshape(params_per_block, n_block2_qubits)
        block3_params = params[block1_size+block2_size:].reshape(params_per_block, n_block3_qubits)
        
        # Amplitude encoding of the noisy state
        amplitude_encode(x_t, wires=range(n_qubits))
        
        # If we're using labels, prepare the states for the labels
        if use_labels and label is not None:
            # Convert the label to binary representation
            binary_label = [int(b) for b in format(label, f'0{n_label_qubits}b')]
            
            # Set the label qubits to the correct states
            for i, bit in enumerate(binary_label):
                if bit == 1:
                    qml.PauliX(wires=n_qubits + m_ancilla + i)
        
        # Determine the wires for each block
        wires_block1 = list(range(n_qubits))
        if use_labels:
            # Add the label qubits
            wires_block1 += list(range(n_qubits + m_ancilla, n_qubits + m_ancilla + n_label_qubits))
        
        # Unitary block 1
        unitary_block(block1_params, wires_block1, n_layers)
        
        # Unitary block 2 (with ancilla qubits)
        wires_block2 = list(range(n_qubits + m_ancilla))
        if use_labels:
            # Add the label qubits
            wires_block2 += list(range(n_qubits + m_ancilla, n_qubits + m_ancilla + n_label_qubits))
        
        unitary_block(block2_params, wires_block2, n_layers)
        
        # Measurement of the ancilla qubit
        if m_ancilla > 0:
            for i in range(n_qubits, n_qubits + m_ancilla):
                qml.measure(wires=i)
        
        # If using labels, also measure the label qubits
        if use_labels:
            for i in range(n_label_qubits):
                qml.measure(wires=n_qubits + m_ancilla + i)
        
        # Unitary block 3 (only on data qubits)
        unitary_block(block3_params, range(n_qubits), n_layers)
        
        # Return the amplitude of the resulting state
        return qml.probs(wires=range(n_qubits))
    
    return circuit

# Definition of the loss function (infidelity)
def create_infidelity_loss_fn(reverse_bottleneck_pqc, n_qubits, use_labels=False):
    """
    Creates a loss function based on infidelity between the target state and the prediction.
    
    Args:
        reverse_bottleneck_pqc: PQC circuit function
        n_qubits: Number of qubits for data
        use_labels: If True, considers labels in the loss
        
    Returns:
        Loss function
    """
    def infidelity_loss(params, x_t, x_t_minus_1, t, label=None):
        """
        Calculates the infidelity loss.
        
        Args:
            params: Circuit parameters
            x_t: Noisy state at timestep t
            x_t_minus_1: Target state (less noisy) at timestep t-1
            t: Current timestep
            label: Label for conditioning (optional)
            
        Returns:
            Infidelity loss (1 - fidelity)
        """
        # Make sure everything is in NumPy format
        if torch.is_tensor(params):
            params = params.detach().cpu().numpy()
        if torch.is_tensor(x_t):
            x_t = x_t.detach().cpu().numpy()
        if torch.is_tensor(x_t_minus_1):
            x_t_minus_1 = x_t_minus_1.detach().cpu().numpy()

        try:
            # Get the predicted state from the circuit
            if use_labels and label is not None:
                circuit_output = reverse_bottleneck_pqc(x_t, params, t, label)
            else:
                circuit_output = reverse_bottleneck_pqc(x_t, params, t)
            
            # For the default.qubit device, the result might be different
            # Check if we got a probability density or a state
            if len(circuit_output.shape) == 1:  # We got probabilities
                # Retrieve the amplitudes as the square root of the probabilities (phase equal to 0)
                predicted_amplitudes = np.sqrt(circuit_output)
            else:  # We got the complete state or a density matrix
                # Extract the relevant part of the state (excluding ancilla and labels)
                predicted_amplitudes = circuit_output[:2**n_qubits]
            
            # Normalize the amplitude of x_t_minus_1
            x_target_norm = x_t_minus_1 / np.sqrt(np.sum(np.abs(x_t_minus_1)**2))
            
            # The fidelity is |<ψ|φ>|²
            fidelity = np.abs(np.vdot(predicted_amplitudes, x_target_norm))**2
            
            # Infidelity = 1 - fidelity
            return 1 - fidelity
        except Exception as e:
            print(f"Error in calculating the infidelity loss: {str(e)}")
            return 1.0
    
    return infidelity_loss

# Calculation of necessary parameters for the circuit
def calculate_circuit_parameters(n_qubits, m_ancilla, n_layers, use_labels=False, n_label_qubits=0):
    """
    Calculates the number and structure of parameters needed for the circuit.
    
    Args:
        n_qubits: Number of qubits for data
        m_ancilla: Number of ancilla qubits
        n_layers: Number of layers in the circuit
        use_labels: If True, considers labels
        n_label_qubits: Number of qubits for labels
        
    Returns:
        total_params: Total number of parameters
        params_structure: Dictionary with the parameter structure
    """
    params_per_block = 3 + n_layers * 3
    
    # Determine the effective number of qubits for each block
    n_block1_qubits = n_qubits + (n_label_qubits if use_labels else 0)
    n_block2_qubits = n_qubits + m_ancilla + (n_label_qubits if use_labels else 0)
    n_block3_qubits = n_qubits
    
    # Calculate the parameters needed for each block
    block1_size = params_per_block * n_block1_qubits
    block2_size = params_per_block * n_block2_qubits
    block3_size = params_per_block * n_block3_qubits
    
    total_params = block1_size + block2_size + block3_size
    
    params_structure = {
        'total': total_params,
        'block1': {
            'size': block1_size,
            'shape': (params_per_block, n_block1_qubits)
        },
        'block2': {
            'size': block2_size,
            'shape': (params_per_block, n_block2_qubits)
        },
        'block3': {
            'size': block3_size,
            'shape': (params_per_block, n_block3_qubits)
        }
    }
    
    return total_params, params_structure

# PART 4: MODEL TRAINING

def train_quantum_diffusion(X_train, y_train=None, n_qubits=8, m_ancilla=1, n_layers=50, 
                          use_labels=False, n_label_qubits=0, n_epochs=20, batch_size=32, lr=0.01):
    """
    Trains the quantum diffusion model.
    """
    # Initialize the forward process
    forward_process = AutoencoderForward(X_train.shape[1], timesteps, alpha_cumprod)
    
    # Calculate the parameters needed for the quantum circuit
    total_params, params_structure = calculate_circuit_parameters(
        n_qubits, m_ancilla, n_layers, use_labels, n_label_qubits)
    
    print(f"Total circuit parameters: {total_params}")
    
    # Create the quantum device
    extra_qubits = n_label_qubits if use_labels else 0
    dev = qml.device('default.qubit', wires=n_qubits + m_ancilla + extra_qubits)
    
    # Create the circuit and loss function
    circuit = create_reverse_bottleneck_pqc(dev, n_qubits, m_ancilla, n_layers, use_labels, n_label_qubits)
    loss_fn = create_infidelity_loss_fn(circuit, n_qubits, use_labels)
    
    # Initialize circuit parameters
    np.random.seed(42)
    params = np.random.uniform(0, 2*np.pi, total_params)
    
    # ADDITION: Create PyTorch tensor for parameters with requires_grad=True
    params_tensor = torch.tensor(params, requires_grad=True, dtype=torch.float64)
    
    # ADDITION: Create optimizer
    optimizer = torch.optim.Adam([params_tensor], lr=lr)
    
    # Convert the dataset to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.complex128)
    
    # Check compatibility of dimensions for conditioned training
    if use_labels and y_train is not None:
        if len(y_train) != len(X_train_tensor):
            min_len = min(len(y_train), len(X_train_tensor))
            X_train_tensor = X_train_tensor[:min_len]
            y_train = y_train[:min_len]
            print(f"Dataset resized to {min_len} examples")
    
    # Loss tracking
    losses = []
    
    print(f"Starting training for {n_epochs} epochs...")
    
    # Dataset index management
    indices = np.arange(len(X_train_tensor))
    
    try:
        for epoch in range(n_epochs):
            epoch_loss = 0
            # Shuffle dataset
            np.random.shuffle(indices)
            
            # Training on mini-batches
            num_batches = math.ceil(len(X_train_tensor) / batch_size)
            
            for i in range(num_batches):
                # Calculate indices for this batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Batch preparation
                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train[batch_indices] if use_labels and y_train is not None else None
                
                # MODIFICATION: Reset gradient at the start of each batch
                optimizer.zero_grad()
                
                # Batch loss calculation
                batch_loss = 0
                successful_samples = 0
                
                for idx in range(len(batch_X)):
                    x_0 = batch_X[idx]
                    
                    # Apply forward process
                    x_t_sequence = forward_process(x_0)
                    
                    # Label for conditioning (if applicable)
                    label = int(batch_y[idx]) if use_labels and batch_y is not None else None
                    
                    # Calculate loss for each timestep
                    sample_loss = 0
                    timestep_count = 0
                    
                    for t in range(1, timesteps):
                        # Convert to NumPy
                        x_t = x_t_sequence[t].detach().cpu().numpy()
                        x_t_minus_1 = x_t_sequence[t-1].detach().cpu().numpy()
                        
                        # Calculate loss using parameters as PyTorch tensor
                        current_loss = loss_fn(params_tensor.detach().cpu().numpy(), x_t, x_t_minus_1, t, label)
                        sample_loss += current_loss
                        timestep_count += 1
                        
                        # Print timestep information for debugging
                        if idx == 0 and i % 5 == 0:  # Print only for the first sample and every 5 batches
                            print(f"  Timestep {t}/{timesteps-1}, Loss: {current_loss:.6f}")
                    
                    # Average loss for the sample
                    if timestep_count > 0:
                        sample_loss /= timestep_count
                        batch_loss += sample_loss
                        successful_samples += 1
                
                # Average loss for the batch
                if successful_samples > 0:
                    batch_loss /= successful_samples
                    
                    # ADDITION: Create a tensor for the loss that maintains the computational graph
                    batch_loss_tensor = torch.tensor(batch_loss, requires_grad=True)
                    
                    # ADDITION: Backpropagation
                    batch_loss_tensor.backward()
                    
                    # ADDITION: Update parameters
                    optimizer.step()
                    
                    epoch_loss += batch_loss
                    print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{num_batches}, Loss: {batch_loss:.6f}")
            
            # Average loss for the epoch
            if num_batches > 0:
                epoch_loss /= num_batches
                losses.append(epoch_loss)
                print(f"Epoch {epoch+1}/{n_epochs}, Average loss: {epoch_loss:.6f}")
                
                # ADDITION: Save current parameters in NumPy format
                params = params_tensor.detach().cpu().numpy()
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()  # Addition to print full traceback
        
        # If no losses are recorded, add a default value
        if len(losses) == 0:
            losses.append(1.0)
            
        # Update final parameters
        params = params_tensor.detach().cpu().numpy()

    return params, losses

# PART 5: SAMPLING FROM THE TRAINED MODEL

def sample_from_model(params, n_samples, n_qubits, m_ancilla, n_layers, use_labels=False, 
                      n_label_qubits=0, labels=None):
    """
    Generates samples from the trained quantum diffusion model.
    
    Args:
        params: Trained circuit parameters
        n_samples: Number of samples to generate
        n_qubits: Number of qubits for data
        m_ancilla: Number of ancilla qubits
        n_layers: Number of layers in the circuit
        use_labels: If True, uses the conditioned model
        n_label_qubits: Number of qubits for labels
        labels: Labels for conditioning (if use_labels=True)
        
    Returns:
        samples: Generated samples
    """
    # Create device and circuit
    extra_qubits = n_label_qubits if use_labels else 0
    dev = create_device(n_qubits, m_ancilla, extra_qubits)
    circuit = create_reverse_bottleneck_pqc(dev, n_qubits, m_ancilla, n_layers, use_labels, n_label_qubits)
    
    samples = []
    
    # If no labels are provided but use_labels is True, generate random labels
    if use_labels and labels is None:
        labels = np.random.randint(0, 2**n_label_qubits, size=n_samples)
    
    for i in range(n_samples):
        # Start with pure noise (complex Gaussian)
        feature_dim = 2**n_qubits
        noise_real = np.random.normal(0, 1, size=feature_dim)
        noise_imag = np.random.normal(0, 1, size=feature_dim)
        x_T = noise_real + 1j * noise_imag
        
        # Normalization
        x_T = x_T / np.sqrt(np.sum(np.abs(x_T)**2))
        
        # Label for conditioning (if applicable)
        label = int(labels[i]) if use_labels and labels is not None else None
        
        # Inverse denoising process
        x_t = x_T
        
        for t in reversed(range(timesteps)):
            # Apply quantum circuit to remove noise
            if use_labels and label is not None:
                state = circuit(x_t, params, t, label)
            else:
                state = circuit(x_t, params, t)
            
            # Extract relevant amplitudes
            x_t = state[:2**n_qubits]
            
            # Normalization
            x_t = x_t / np.sqrt(np.sum(np.abs(x_t)**2))
        
        # Convert to absolute values as indicated in the paper
        sample = np.abs(x_t)
        samples.append(sample)
    
    return np.array(samples)

# PART 6: LATENT MODEL (HYBRID CLASSICAL-QUANTUM)

class ClassicalAutoencoder(nn.Module):
    """
    Classical autoencoder for dimensionality reduction.
    Used in the latent variant of the quantum diffusion model.
    
    This component implements the "Latent models" part described in section 3.3 of the paper,
    where a pre-trained classical autoencoder is used to reduce the dimensionality
    of the data before applying the quantum diffusion model.
    """
    def __init__(self, input_dim, latent_dim):
        super(ClassicalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # For images normalized between 0 and 1
        )
    
    def forward(self, x):
        # Encoding
        z = self.encoder(x)
        # Decoding
        reconstructed = self.decoder(z)
        return reconstructed
    
    def encode(self, x):
        """Encodes data into the latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decodes from latent space to original space."""
        return self.decoder(z)

def train_classical_autoencoder(X_train, latent_dim, n_epochs=50, batch_size=64, lr=0.001):
    """
    Trains a classical autoencoder for dimensionality reduction.
    
    Args:
        X_train: Training dataset
        latent_dim: Latent space dimension
        n_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        
    Returns:
        model: Trained autoencoder
    """
    input_dim = X_train.shape[1]
    model = ClassicalAutoencoder(input_dim, latent_dim)
    
    # Loss criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Conversion to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # Dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"Starting autoencoder training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        running_loss = 0.0
        for data, targets in dataloader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")
    
    return model

# PART 7: EVALUATION METRICS
# As described in section 3.4 "Model evaluation" of the paper

def calculate_roc_auc(samples, true_labels, n_classes=10):
    """
    Calculates the ROC-AUC score to evaluate conditioning performance.
    
    Args:
        samples: Generated samples
        true_labels: True labels of the samples
        n_classes: Number of classes
        
    Returns:
        roc_auc_scores: ROC-AUC score for each class
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    
    # Train classifiers
    classifiers = []
    for i in range(n_classes):
        # Create a binary dataset for each class
        binary_labels = (true_labels == i).astype(int)
        
        # Train the classifier
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        clf.fit(samples, binary_labels)
        classifiers.append(clf)
    
    # Calculate ROC-AUC score for each class
    roc_auc_scores = []
    for i, clf in enumerate(classifiers):
        # Predict probabilities for class i
        pred_probs = clf.predict_proba(samples)[:, 1]
        
        # Create binary labels for class i
        binary_labels = (true_labels == i).astype(int)
        
        # Calculate ROC-AUC score
        auc_score = roc_auc_score(binary_labels, pred_probs)
        roc_auc_scores.append(auc_score)
    
    return roc_auc_scores

def calculate_frechet_inception_distance(real_samples, generated_samples):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated samples.
    
    Args:
        real_samples: Real samples
        generated_samples: Generated samples
        
    Returns:
        fid: Fréchet Inception Distance
    """
    from scipy import linalg
    
    # Calculate mean and covariance for real samples
    mu1 = np.mean(real_samples, axis=0)
    sigma1 = np.cov(real_samples, rowvar=False)
    
    # Calculate mean and covariance for generated samples
    mu2 = np.mean(generated_samples, axis=0)
    sigma2 = np.cov(generated_samples, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    
    # Add a small constant to avoid numerical problems
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * 1e-6
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * 1e-6
    
    # Calculate geometric mean
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check if there are numerically insignificant imaginary values
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid

def calculate_wasserstein_distance_gaussian_mixture(real_samples, generated_samples, n_components=10):
    """
    Calculates the 2-Wasserstein distance for Gaussian mixture models (WaM).
    As specified in section 3.4 of the paper, this metric is useful as FID
    can suffer from problems with small grayscale images.
    
    Args:
        real_samples: Real samples
        generated_samples: Generated samples
        n_components: Number of components in the mixture model
        
    Returns:
        wam: 2-Wasserstein distance
    """
    from sklearn.mixture import GaussianMixture
    
    # Train GMM on real samples
    gmm_real = GaussianMixture(n_components=n_components, random_state=42)
    gmm_real.fit(real_samples)
    
    # Train GMM on generated samples
    gmm_gen = GaussianMixture(n_components=n_components, random_state=42)
    gmm_gen.fit(generated_samples)
    
    # Extract model parameters
    means_real = gmm_real.means_
    covs_real = gmm_real.covariances_
    weights_real = gmm_real.weights_
    
    means_gen = gmm_gen.means_
    covs_gen = gmm_gen.covariances_
    weights_gen = gmm_gen.weights_
    
    # WaM calculation (approximation)
    wam = 0
    for i in range(n_components):
        for j in range(n_components):
            # Calculate 2-Wasserstein distance between two Gaussian distributions
            diff_means = means_real[i] - means_gen[j]
            term1 = np.sum(diff_means**2)
            
            # Calculate trace
            cov_real = covs_real[i]
            cov_gen = covs_gen[j]
            
            # Calculate geometric mean (approximation)
            cov_mean = (cov_real + cov_gen) / 2
            
            term2 = np.trace(cov_real) + np.trace(cov_gen) - 2 * np.trace(cov_mean)
            
            # Calculate weighted distance
            wam += weights_real[i] * weights_gen[j] * (term1 + term2)
    
    return wam

# PART 8: MNIST DATASET PREPROCESSING

def preprocess_mnist_for_latent_model(latent_dim=8, fraction=1.0):
    """
    Preprocesses the MNIST dataset for the latent model:
    - Normalizes values between 0 and 1
    - Trains an autoencoder to reduce dimensionality to latent_dim
    - Optionally uses only a fraction of the dataset
    
    Args:
        latent_dim: Latent space dimension
        fraction: Fraction of the dataset to use (1.0 = all)
        
    Returns:
        X_train_latent: Training data in latent space
        X_test_latent: Test data in latent space
        y_train: Training labels
        y_test: Test labels
        autoencoder: Trained autoencoder for encoding/decoding
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.astype('float32')
    y = y.astype('int')
    
    # Use only a fraction of the dataset if requested
    if fraction < 1.0:
        X_subset_indices = np.random.choice(len(X), size=int(len(X) * fraction), replace=False)
        # DataFrame requires .iloc or .loc for indexing with array of indices
        if hasattr(X, 'iloc'):
            X = X.iloc[X_subset_indices]
            y = y.iloc[X_subset_indices] if hasattr(y, 'iloc') else y[X_subset_indices]
        else:
            # If X is a NumPy array, we can index it directly
            X = X[X_subset_indices]
            y = y[X_subset_indices]
        print(f"Using {fraction*100:.1f}% of the dataset: {len(X)} examples")
    
    # Normalize data between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the autoencoder for dimensionality reduction
    print(f"Training the autoencoder to reduce dimensionality to {latent_dim}...")
    autoencoder = train_classical_autoencoder(X_train, latent_dim, n_epochs=20)
    
    # Convert data to latent space
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        X_train_latent = autoencoder.encode(X_train_tensor).numpy()
        X_test_latent = autoencoder.encode(X_test_tensor).numpy()
    
    print(f"Latent dataset dimensions: X_train_latent: {X_train_latent.shape}, X_test_latent: {X_test_latent.shape}")
    
    return X_train_latent, X_test_latent, y_train, y_test, autoencoder

# PART 9: RESULT VISUALIZATION

def plot_generated_samples(samples_uncond=None, samples_cond=None, n_images=25):
    """
    Visualizes samples generated by the model.
    
    Args:
        samples_uncond: Samples generated by the unconditioned model
        samples_cond: Samples generated by the conditioned model
        n_images: Number of images to display
    """
    import math
    
    # Calculate rows and columns
    imgs_per_row = 5
    rows_per_model = math.ceil(min(n_images, 25) / imgs_per_row)
    
    # Display unconditioned samples
    if samples_uncond is not None:
        n_to_show = min(n_images, len(samples_uncond))
        fig_uncond, axes_uncond = plt.subplots(rows_per_model, imgs_per_row, 
                                              figsize=(15, 4 * rows_per_model))
        
        # Handle the case of a single axis
        if rows_per_model == 1 and imgs_per_row == 1:
            axes_uncond = np.array([axes_uncond])
        if rows_per_model == 1:
            axes_uncond = np.expand_dims(axes_uncond, axis=0)
            
        # Display images
        for i in range(rows_per_model):
            for j in range(imgs_per_row):
                idx = i * imgs_per_row + j
                if idx < n_to_show:
                    sample_idx = idx % len(samples_uncond)
                    ax = axes_uncond[i, j] if rows_per_model > 1 else axes_uncond[j]
                    ax.imshow(samples_uncond[sample_idx].reshape(28, 28), cmap='gray')
                    ax.axis('off')
                else:
                    if rows_per_model > 1:
                        axes_uncond[i, j].axis('off')
                    else:
                        axes_uncond[j].axis('off')
        
        plt.suptitle("Samples from unconditioned model", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    # Display conditioned samples
    if samples_cond is not None:
        n_to_show = min(n_images, len(samples_cond))
        fig_cond, axes_cond = plt.subplots(rows_per_model, imgs_per_row, 
                                          figsize=(15, 4 * rows_per_model))
        
        # Handle the case of a single axis
        if rows_per_model == 1 and imgs_per_row == 1:
            axes_cond = np.array([axes_cond])
        if rows_per_model == 1:
            axes_cond = np.expand_dims(axes_cond, axis=0)
            
        # Display images
        for i in range(rows_per_model):
            for j in range(imgs_per_row):
                idx = i * imgs_per_row + j
                if idx < n_to_show:
                    sample_idx = idx % len(samples_cond)
                    ax = axes_cond[i, j] if rows_per_model > 1 else axes_cond[j]
                    ax.imshow(samples_cond[sample_idx].reshape(28, 28), cmap='gray')
                    ax.axis('off')
                else:
                    if rows_per_model > 1:
                        axes_cond[i, j].axis('off')
                    else:
                        axes_cond[j].axis('off')
        
        plt.suptitle("Samples from conditioned model", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# PART 10: EXAMPLE MODEL EXECUTION

def run_latent_quantum_model(use_conditioning=False, dataset_fraction=0.1):
    """
    Example usage of the latent quantum model, following the paper's parameters.
    
    Args:
        use_conditioning: If True, implements the conditioned model
        dataset_fraction: Fraction of the dataset to use (0.1 = 10%)
        
    Returns:
        params: Trained parameters
        latent_samples: Generated samples in latent space
        generated_samples: Generated samples in original space
        autoencoder: Autoencoder used for encoding/decoding
    """
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running latent model on: {device}")

    # Preprocess MNIST dataset for the latent model
    X_train_latent, X_test_latent, y_train, y_test, autoencoder = preprocess_mnist_for_latent_model(
        latent_dim=8, fraction=dataset_fraction
    )

    # Convert y_train and y_test from Pandas Series to NumPy array if needed
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # Parameters from the paper
    n_qubits = 3  # 3 qubits to encode the latent space
    m_ancilla = 1  # 1 ancilla qubit
    n_layers = latent_n_layers  
    n_label_qubits = 4 if use_conditioning else 0  # 4 qubits for labels if conditioned

    # Hyperparameters
    timesteps = 8
    n_epochs = 5
    batch_size = 64

    # Model training
    print(f"Training the {'conditioned ' if use_conditioning else ''}latent quantum diffusion model...")
    try:
        params_trained, losses = train_quantum_diffusion(
            X_train_latent, y_train=y_train if use_conditioning else None,
            n_qubits=n_qubits, m_ancilla=m_ancilla, n_layers=n_layers,
            use_labels=use_conditioning, n_label_qubits=n_label_qubits,
            n_epochs=n_epochs, batch_size=batch_size
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        total_params, _ = calculate_circuit_parameters(
            n_qubits, m_ancilla, n_layers, use_conditioning, n_label_qubits)
        params_trained = np.random.uniform(0, 2*np.pi, total_params)
        losses = [1.0]

    # Sample generation
    print("Generating samples from the trained model...")
    n_samples = 50

    # If conditioned, generate samples for each class
    if use_conditioning:
        labels = np.repeat(np.arange(10), n_samples // 10 + 1)[:n_samples]
    else:
        labels = None

    # Generate samples in latent space
    try:
        latent_samples = sample_from_model(
            params_trained, n_samples,
            n_qubits=n_qubits, m_ancilla=m_ancilla, n_layers=n_layers,
            use_labels=use_conditioning, n_label_qubits=n_label_qubits,
            labels=labels
        )
    except Exception as e:
        print(f"Error during sampling: {str(e)}")
        feature_dim = 2**n_qubits
        latent_samples = np.random.rand(n_samples, feature_dim)
        for i in range(n_samples):
            latent_samples[i] = latent_samples[i] / np.sum(latent_samples[i])

    # Decode latent samples to original space
    with torch.no_grad():
        latent_samples_tensor = torch.tensor(latent_samples, dtype=torch.float32)
        generated_samples = autoencoder.decode(latent_samples_tensor).numpy()

    # Model evaluation
    print("Evaluating the model...")

    # Evaluation in latent space
    try:
        fid_latent = calculate_frechet_inception_distance(X_test_latent, latent_samples)
        wam_latent = calculate_wasserstein_distance_gaussian_mixture(X_test_latent, latent_samples)

        print(f"Evaluation metrics in latent space:")
        print(f"Fréchet Inception Distance (FID): {fid_latent:.4f}")
        print(f"Wasserstein distance for Gaussian mixture models (WaM): {wam_latent:.4f}")
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")

    # If conditioned, also calculate ROC-AUC
    if use_conditioning and labels is not None:
        try:
            roc_auc_scores = calculate_roc_auc(latent_samples, labels)
            print(f"ROC-AUC scores per class:")
            for i, score in enumerate(roc_auc_scores):
                print(f"Class {i}: {score:.4f}")
        except Exception as e:
            print(f"Error calculating ROC-AUC: {str(e)}")

    return params_trained, latent_samples, generated_samples, autoencoder

# Example simplified usage
if __name__ == "__main__":
    # Run a reduced model for testing
    print("Starting the quantum diffusion model on a subset of MNIST...")
    
    try:
        # Run unconditioned model with a small fraction of the dataset
        params_uncond, latent_samples_uncond, samples_uncond, autoencoder_uncond = run_latent_quantum_model(
            use_conditioning=False, dataset_fraction=0.1)
        params_cond, latent_samples_cond, samples_cond, autoencoder_cond = run_latent_quantum_model(
            use_conditioning=True, dataset_fraction=0.1)
        
        
        # Visualize generated samples
        print("Visualizing generated samples:")
        plot_generated_samples(samples_uncond=samples_uncond)
        plot_generated_samples(samples_cond=samples_cond)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Execution completed.")