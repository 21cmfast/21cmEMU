#!/usr/bin/env python3
"""Convert TensorFlow/Keras 21cmEMU v1 model to PyTorch.

This script loads a Keras SavedModel or .keras/.h5 file and creates
an equivalent PyTorch module with transferred weights.

Usage
-----
The conversion can be run either as a script:

    python convert_tf_to_pytorch.py path/to/keras_model output_model.pt

Or via the API:

    >>> from convert_tf_to_pytorch import convert_keras_to_pytorch
    >>> pytorch_model = convert_keras_to_pytorch("path/to/keras_model")
    >>> torch.save(pytorch_model.state_dict(), "output.pt")

Architecture
------------
The v1 21cmEMU model (Breitman+23) is a multi-output MLP:

    Input: 9 astrophysical parameters (normalised to [0,1])
    
    Shared layers:
        Dense(512, activation='relu')
        Dense(512, activation='relu')
        Dense(512, activation='relu')
    
    Output heads (each a Dense layer):
        - Tb: (n_z,)        global brightness temperature
        - xHI: (n_z,)       neutral fraction
        - Ts: (n_z,)        spin temperature  
        - tau: (1,)         optical depth
        - PS: (n_z, n_k)    power spectrum (flattened to n_z*n_k)
        - UVLFs: (n_z, n_mag) UV luminosity functions (flattened)

The model predicts all 6 outputs in a single forward pass.

Requirements
------------
- tensorflow >= 2.x (only for loading source model)
- torch >= 2.x

After conversion, only PyTorch is needed for inference.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


log = logging.getLogger(__name__)


# =============================================================================
# PyTorch Model Definition
# =============================================================================

class DefaultEmulatorPyTorch(nn.Module):
    """PyTorch equivalent of the v1 TensorFlow 21cmEMU model.
    
    A multi-output MLP with shared hidden layers and separate output heads
    for each summary statistic.
    
    Parameters
    ----------
    n_params : int
        Number of input parameters (default 9 for v1).
    hidden_sizes : list of int
        Sizes of shared hidden layers.
    output_sizes : dict
        Mapping from output name to output dimension.
    activation : str
        Activation function for hidden layers ('relu', 'elu', 'gelu').
    """
    
    def __init__(
        self,
        n_params: int = 9,
        hidden_sizes: List[int] = None,
        output_sizes: Dict[str, int] = None,
        activation: str = 'relu',
    ):
        super().__init__()
        
        if hidden_sizes is None:
            # Default v1 architecture
            hidden_sizes = [512, 512, 512]
            
        if output_sizes is None:
            # Default v1 outputs
            # These are the flattened dimensions from the original model
            output_sizes = {
                'Tb': 93,      # n_z redshift bins
                'xHI': 93,     # n_z redshift bins  
                'Ts': 93,      # n_z redshift bins
                'tau': 1,      # scalar
                'PS': 93 * 14, # n_z × n_k (flattened)
                'UVLFs': 7 * 45, # n_lf_z × n_mag (flattened)
            }
        
        self.n_params = n_params
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.output_names = list(output_sizes.keys())
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build shared layers
        shared_layers = []
        in_dim = n_params
        for h_size in hidden_sizes:
            shared_layers.append(nn.Linear(in_dim, h_size))
            shared_layers.append(act_fn())
            in_dim = h_size
        self.shared = nn.Sequential(*shared_layers)
        
        # Build output heads
        self.heads = nn.ModuleDict()
        for name, out_dim in output_sizes.items():
            self.heads[name] = nn.Linear(hidden_sizes[-1], out_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_params)
            Normalised input parameters.
            
        Returns
        -------
        dict of torch.Tensor
            Dictionary mapping output name to prediction tensor.
        """
        # Shared layers
        h = self.shared(x)
        
        # Output heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(h)
        
        return outputs
    
    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning concatenated outputs (for compatibility).
        
        Returns all outputs concatenated along dim=-1 in the order
        specified by self.output_names.
        """
        outputs = self.forward(x)
        return torch.cat([outputs[name] for name in self.output_names], dim=-1)


# =============================================================================
# Conversion Functions
# =============================================================================

def inspect_keras_model(model_path: str) -> Dict[str, Any]:
    """Inspect a Keras model and extract architecture info.
    
    Parameters
    ----------
    model_path : str
        Path to saved Keras model (SavedModel dir, .keras, or .h5).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'layers': list of layer info dicts
        - 'input_shape': input shape
        - 'output_shape': output shape(s)
        - 'weights': list of (name, shape) tuples
    """
    import tensorflow as tf
    
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    info = {
        'layers': [],
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'weights': [],
    }
    
    # Extract layer info
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
            'config': layer.get_config(),
            'weights': [(w.name, w.shape.as_list()) for w in layer.weights],
        }
        info['layers'].append(layer_info)
        
        for w in layer.weights:
            info['weights'].append((w.name, w.shape.as_list()))
    
    return info


def extract_weights_from_keras(model_path: str) -> Dict[str, np.ndarray]:
    """Extract weights from a Keras model.
    
    Parameters
    ----------
    model_path : str
        Path to saved Keras model.
        
    Returns
    -------
    dict
        Dictionary mapping weight name to numpy array.
    """
    import tensorflow as tf
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    weights = {}
    for layer in model.layers:
        for w in layer.weights:
            # Clean up the name (remove ':0' suffix etc.)
            name = w.name.replace(':0', '')
            weights[name] = w.numpy()
    
    return weights


def convert_keras_to_pytorch(
    model_path: str,
    verbose: bool = True,
) -> DefaultEmulatorPyTorch:
    """Convert a Keras 21cmEMU model to PyTorch.
    
    Parameters
    ----------
    model_path : str
        Path to saved Keras model.
    verbose : bool
        Whether to print conversion progress.
        
    Returns
    -------
    DefaultEmulatorPyTorch
        PyTorch model with transferred weights.
    """
    import tensorflow as tf
    
    if verbose:
        print(f"Loading Keras model from {model_path}...")
    
    # Load and inspect
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    
    # Analyze architecture
    dense_layers = []
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layers.append({
                'name': layer.name,
                'units': layer.units,
                'activation': layer.activation.__name__,
                'weights': layer.get_weights(),
            })
    
    if verbose:
        print(f"Found {len(dense_layers)} Dense layers:")
        for i, info in enumerate(dense_layers):
            w_shape = info['weights'][0].shape if info['weights'] else "N/A"
            print(f"  [{i}] {info['name']}: units={info['units']}, "
                  f"activation={info['activation']}, weights={w_shape}")
    
    # Determine architecture from layer analysis
    # Typically: shared layers have activations, output layers don't
    shared_dims = []
    output_heads = {}
    
    for info in dense_layers:
        if info['activation'] not in ('linear', 'softmax', 'sigmoid'):
            shared_dims.append(info['units'])
        else:
            # This is an output layer
            output_heads[info['name']] = info['units']
    
    # Get input dimension from first layer's weights
    n_params = dense_layers[0]['weights'][0].shape[0]
    
    if verbose:
        print(f"\nDetected architecture:")
        print(f"  Input dim: {n_params}")
        print(f"  Shared hidden: {shared_dims}")
        print(f"  Output heads: {output_heads}")
    
    # Create PyTorch model
    pytorch_model = DefaultEmulatorPyTorch(
        n_params=n_params,
        hidden_sizes=shared_dims,
        output_sizes=output_heads,
        activation='relu',  # v1 uses ReLU
    )
    
    # Transfer weights
    if verbose:
        print("\nTransferring weights...")
    
    # Shared layers
    keras_layer_idx = 0
    for i, (torch_linear, torch_act) in enumerate(zip(
        pytorch_model.shared[0::2],  # Linear layers
        pytorch_model.shared[1::2],  # Activation layers
    )):
        keras_layer = dense_layers[keras_layer_idx]
        weights, biases = keras_layer['weights']
        
        # Keras Dense: weight is (in, out), PyTorch Linear: weight is (out, in)
        torch_linear.weight.data = torch.from_numpy(weights.T.copy())
        torch_linear.bias.data = torch.from_numpy(biases.copy())
        
        if verbose:
            print(f"  Shared layer {i}: {keras_layer['name']} -> "
                  f"shape {torch_linear.weight.shape}")
        
        keras_layer_idx += 1
    
    # Output heads
    for name in pytorch_model.output_names:
        keras_layer = dense_layers[keras_layer_idx]
        weights, biases = keras_layer['weights']
        
        torch_head = pytorch_model.heads[name]
        torch_head.weight.data = torch.from_numpy(weights.T.copy())
        torch_head.bias.data = torch.from_numpy(biases.copy())
        
        if verbose:
            print(f"  Output head '{name}': {keras_layer['name']} -> "
                  f"shape {torch_head.weight.shape}")
        
        keras_layer_idx += 1
    
    pytorch_model.eval()
    
    if verbose:
        print(f"\nConversion complete! Total parameters: "
              f"{sum(p.numel() for p in pytorch_model.parameters()):,}")
    
    return pytorch_model


def verify_conversion(
    keras_model_path: str,
    pytorch_model: DefaultEmulatorPyTorch,
    n_test: int = 100,
    atol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """Verify that PyTorch model produces same outputs as Keras.
    
    Parameters
    ----------
    keras_model_path : str
        Path to original Keras model.
    pytorch_model : DefaultEmulatorPyTorch
        Converted PyTorch model.
    n_test : int
        Number of random test samples.
    atol : float
        Absolute tolerance for comparison.
    verbose : bool
        Whether to print verification results.
        
    Returns
    -------
    bool
        True if outputs match within tolerance.
    """
    import tensorflow as tf
    
    keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
    
    # Generate random test inputs
    np.random.seed(42)
    test_inputs = np.random.rand(n_test, pytorch_model.n_params).astype(np.float32)
    
    # Keras predictions
    keras_preds = keras_model.predict(test_inputs, verbose=0)
    if not isinstance(keras_preds, (list, tuple)):
        keras_preds = [keras_preds]
    keras_concat = np.concatenate([p.reshape(n_test, -1) for p in keras_preds], axis=-1)
    
    # PyTorch predictions
    pytorch_model.eval()
    with torch.no_grad():
        torch_inputs = torch.from_numpy(test_inputs)
        torch_preds = pytorch_model.forward_stacked(torch_inputs)
        torch_concat = torch_preds.numpy()
    
    # Compare
    max_diff = np.abs(keras_concat - torch_concat).max()
    mean_diff = np.abs(keras_concat - torch_concat).mean()
    
    passed = max_diff < atol
    
    if verbose:
        print(f"\nVerification results ({n_test} samples):")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Tolerance: {atol:.2e}")
        print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    
    return passed


# =============================================================================
# Alternative: Convert from scratch if model structure is known
# =============================================================================

def create_v1_model_from_scratch(
    hidden_sizes: List[int] = None,
    n_z: int = 93,
    n_k: int = 14,
    n_lf_z: int = 7,
    n_mag: int = 45,
) -> DefaultEmulatorPyTorch:
    """Create a v1-style model from scratch (for retraining).
    
    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes. Default [512, 512, 512].
    n_z : int
        Number of redshift bins for global signals.
    n_k : int
        Number of k bins for PS.
    n_lf_z : int  
        Number of redshifts for UVLFs.
    n_mag : int
        Number of magnitude bins for UVLFs.
        
    Returns
    -------
    DefaultEmulatorPyTorch
        Untrained PyTorch model with v1 architecture.
    """
    if hidden_sizes is None:
        hidden_sizes = [512, 512, 512]
    
    output_sizes = {
        'Tb': n_z,
        'xHI': n_z,
        'Ts': n_z,
        'tau': 1,
        'PS': n_z * n_k,
        'UVLFs': n_lf_z * n_mag,
    }
    
    return DefaultEmulatorPyTorch(
        n_params=9,
        hidden_sizes=hidden_sizes,
        output_sizes=output_sizes,
        activation='relu',
    )


# =============================================================================
# Integration with 21cmEMUv3 package
# =============================================================================

def load_default_model_pytorch(
    model_path: str,
    device: str = None,
) -> DefaultEmulatorPyTorch:
    """Load converted v1 model for inference.
    
    Parameters
    ----------
    model_path : str
        Path to .pt file with saved state dict.
    device : str, optional
        Device to load model on.
        
    Returns
    -------
    DefaultEmulatorPyTorch
        Loaded model in eval mode.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load state dict to inspect architecture
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Infer architecture from state dict
    # Count shared layers by looking for 'shared.N.weight' keys
    shared_dims = []
    output_dims = {}
    
    for key, tensor in state_dict.items():
        if key.startswith('shared.') and key.endswith('.weight'):
            layer_idx = int(key.split('.')[1])
            if layer_idx % 2 == 0:  # Linear layers at even indices
                shared_dims.append(tensor.shape[0])
        elif key.startswith('heads.') and key.endswith('.weight'):
            head_name = key.split('.')[1]
            output_dims[head_name] = tensor.shape[0]
    
    # Get input dim from first layer
    n_params = state_dict['shared.0.weight'].shape[1]
    
    # Create model
    model = DefaultEmulatorPyTorch(
        n_params=n_params,
        hidden_sizes=shared_dims,
        output_sizes=output_dims,
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow 21cmEMU model to PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "keras_model",
        help="Path to Keras model (SavedModel dir, .keras, or .h5)",
    )
    parser.add_argument(
        "output",
        help="Output path for PyTorch state dict (.pt)",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify conversion by comparing outputs",
    )
    parser.add_argument(
        "--inspect", "-i",
        action="store_true", 
        help="Just inspect the Keras model without converting",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.inspect:
        info = inspect_keras_model(args.keras_model)
        print("\n=== Keras Model Inspection ===")
        print(f"Input shape: {info['input_shape']}")
        print(f"Output shape: {info['output_shape']}")
        print(f"\nLayers ({len(info['layers'])}):")
        for layer in info['layers']:
            print(f"  {layer['name']} ({layer['type']})")
            for wname, wshape in layer['weights']:
                print(f"    {wname}: {wshape}")
        return
    
    # Convert
    pytorch_model = convert_keras_to_pytorch(args.keras_model, verbose=verbose)
    
    # Optionally verify
    if args.verify:
        passed = verify_conversion(args.keras_model, pytorch_model, verbose=verbose)
        if not passed:
            print("\nWARNING: Verification failed! Check conversion.")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_model.state_dict(), output_path)
    
    if verbose:
        print(f"\nSaved PyTorch model to {output_path}")


if __name__ == "__main__":
    main()
