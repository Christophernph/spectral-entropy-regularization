import torch
import torch.nn as nn
import numpy as np
from skopt import gp_minimize
from skopt.space import Real

class CompressionOptimizer:
    def __init__(self, model, epsilon=1.0):
        """
        Compression Optimizer for Quantization and Low-Rank Approximation.
        
        Args:
            model (nn.Module): The neural network model to compress.
            epsilon (float): Distortion tolerance for compression.
        """
        self.model = model
        self.epsilon = epsilon

    def quantize(self, params, delta):
        """
        Quantize the model parameters.
        
        Args:
            params (torch.Tensor): Model parameters.
            delta (float): Quantization bin size.
        
        Returns:
            torch.Tensor: Quantized parameters.
        """
        return torch.round(params / delta) * delta

    def low_rank_approximation(self, params, tau):
        """
        Perform low-rank approximation on the model parameters.
        
        Args:
            params (torch.Tensor): Model parameters (2D matrix).
            tau (float): Threshold for singular value truncation.
        
        Returns:
            torch.Tensor: Low-rank approximated parameters.
        """
        # Compute Singular Value Decomposition (SVD)
        U, S, V = torch.linalg.svd(params, full_matrices=False)
        
        # Normalize singular values
        normalized_singular_values = S / torch.sum(S)
        
        # Find the rank k that satisfies the threshold tau
        cumulative_sum = torch.cumsum(normalized_singular_values, dim=0)
        k = torch.sum(cumulative_sum < tau).item() + 1
        
        # Truncate singular values and reconstruct the matrix
        S_truncated = torch.zeros_like(S)
        S_truncated[:k] = S[:k]
        return U @ torch.diag(S_truncated) @ V

    def compress_model(self, tau, delta):
        """
        Compress the model using low-rank approximation and quantization.
        
        Args:
            tau (float): Threshold for low-rank approximation.
            delta (float): Quantization bin size.
        
        Returns:
            list: Compressed model parameters.
        """
        compressed_params = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:  # Only compress 2D weight matrices
                # Low-rank approximation
                low_rank_params = self.low_rank_approximation(param.data, tau)
                # Quantization
                quantized_params = self.quantize(low_rank_params, delta)
                compressed_params.append(quantized_params)
            else:
                compressed_params.append(param.data)  # Keep non-2D parameters unchanged
        return compressed_params

    def evaluate_compression(self, tau, delta, dataloader, criterion, original_params, original_loss):
        """
        Evaluate the compression by computing the loss difference.
        
        Args:
            tau (float): Threshold for low-rank approximation.
            delta (float): Quantization bin size.
            dataloader (DataLoader): Data loader for computing the loss.
            criterion (nn.Module): Loss function.
            original_params (dict): Original model parameters.
            original_loss (float): Original loss value.
        
        Returns:
            float: Loss difference between original and compressed models.
        """
        # Compress the model
        compressed_params = self.compress_model(tau, delta)
        
        # Compute the loss with the compressed model
        with torch.no_grad():
            # Replace model parameters with compressed parameters
            for (name, param), compressed_param in zip(self.model.named_parameters(), compressed_params):
                param.copy_(compressed_param)
            
            # Forward pass to compute loss
            inputs, targets = next(iter(dataloader))  # Example input and target
            outputs = self.model(inputs)
            compressed_loss = criterion(outputs, targets)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.copy_(original_params[name])
        
        # Return the absolute loss difference
        return abs(compressed_loss.item() - original_loss.item())

    def optimize_compression(self, dataloader, criterion, n_calls=10, tau_range=(0.1, 1.0), delta_range=(0.01, 0.1)):
        """
        Perform Bayesian optimization to find the best compression parameters.
        
        Args:
            dataloader (DataLoader): Data loader for computing the loss.
            criterion (nn.Module): Loss function.
            n_calls (int): Number of optimization iterations.
            tau_range (tuple): Range of tau values for optimization.
            delta_range (tuple): Range of delta values for optimization.
        
        Returns:
            dict: Best compression parameters (tau, delta) and compressed model.
        """
        # Save original parameters and compute original loss
        original_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        inputs, targets = next(iter(dataloader))  # Example input and target
        original_loss = criterion(self.model(inputs), targets)

        # Define the search space for tau and delta
        space = [
            Real(*tau_range, name='tau'),  # tau: threshold for low-rank approximation
            Real(*delta_range, name='delta')  # delta: quantization bin size
        ]

        # Perform Bayesian optimization
        result = gp_minimize(
            lambda x: self.evaluate_compression(x[0], x[1], dataloader, criterion, original_params, original_loss),
            space,  # Search space
            n_calls=n_calls,  # Number of optimization iterations
            random_state=42  # Random seed for reproducibility
        )

        # Extract the best parameters
        best_tau, best_delta = result.x
        best_compressed_params = self.compress_model(best_tau, best_delta)

        return {
            'tau': best_tau,
            'delta': best_delta,
            'compressed_params': best_compressed_params
        }