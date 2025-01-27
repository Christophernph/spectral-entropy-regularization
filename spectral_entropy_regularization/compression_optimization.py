from typing import Optional

import torch
import torch.nn as nn
from skopt import gp_minimize
from skopt.space import Real
from torch.utils.data import DataLoader


class CompressionOptimizer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 epsilon: float = 1.0):
        """
        Compression Optimizer for Quantization and Low-Rank Approximation.
        
        Args:
            model (nn.Module): The neural network model to compress.
            criterion (nn.Module): Loss function for evaluating the compression.
            epsilon (float): Distortion tolerance for compression.
        """
        self.model = model
        self.epsilon = epsilon
        self.criterion = criterion

    def quantize(self,
                 params: torch.Tensor,
                 delta: float) -> torch.Tensor:
        """
        Quantize the model parameters.
        
        Args:
            params (torch.Tensor): Model parameters.
            delta (float): Quantization bin size.
        
        Returns:
            torch.Tensor: Quantized parameters.
        """
        return torch.round(params / delta) * delta

    def low_rank_approximation(self,
                               params: torch.Tensor,
                               tau: float):
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
        return U @ torch.diag(S_truncated) @ V, k

    def compress_model(self,
                       tau: float,
                       delta: float) -> tuple[dict, int]:
        """
        Compress the model using low-rank approximation and quantization.
        
        Args:
            tau (float): Threshold for low-rank approximation.
            delta (float): Quantization bin size.
        
        Returns:
            tuple[dict, int]: Compressed model parameters and total compressed size.
        """
        compressed_params = dict()
        compressed_size = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:  # Only compress 2D weight matrices
                # Low-rank approximation
                low_rank_params, k = self.low_rank_approximation(param.data, tau)
                # Quantization
                quantized_params = self.quantize(low_rank_params, delta)
                compressed_params[name] = quantized_params
                compressed_size += k
        return compressed_params, compressed_size

    def evaluate_compression(self,
                             tau: float,
                             delta: float,
                             dataloader: DataLoader,
                             original_params: dict,
                             original_loss: torch.Tensor) -> float:
        """
        Evaluate the compression by computing the loss difference.
        
        Args:
            tau (float): Threshold for low-rank approximation.
            delta (float): Quantization bin size.
            dataloader (DataLoader): Data loader for computing the loss.
            original_params (dict): Original model parameters.
            original_loss (float): Original loss value.
        
        Returns:
            float: Compressed model size if the loss is within the distortion tolerance. Otherwise, a large value.
        """
        # Compress the model
        compressed_params, compression_size = self.compress_model(tau, delta)
        
        # Compute the loss with the compressed model
        with torch.no_grad():
            # Replace model parameters with compressed parameters
            for name, parameter in self.model.named_parameters():
                if name in compressed_params:
                    parameter.data = compressed_params[name]
            
            # Forward pass to compute loss
            compressed_loss = 0
            for inputs, targets in dataloader:
                compressed_loss += self.criterion(self.model(inputs), targets)
        
        # Restore original parameters
        for name, parameter in self.model.named_parameters():
            if name in original_params:
                parameter.data = original_params[name]
        
        if abs(original_loss.item() - compressed_loss.item()) < self.epsilon:
            return compression_size
        else:
            return 1e10 # Penalize invalid solutions

    def optimize_compression(self,
                             dataloader: DataLoader,
                             n_calls: int = 10,
                             tau_range: tuple[int, int] = (0.01, 0.1),
                             delta_range: tuple[int, int] = (0.01, 0.1),
                             seed: Optional[int] = None):
        """
        Perform Bayesian optimization to find the best compression parameters.
        
        Args:
            dataloader (DataLoader): Data loader for computing the loss.
            n_calls (int): Number of optimization iterations.
            tau_range (tuple): Range of tau values for optimization.
            delta_range (tuple): Range of delta values for optimization.
            seed (Optional[int]): Random seed for reproducibility.
        
        Returns:
            dict: Best compression parameters (tau, delta) and compressed model.
        """
        # Save original parameters and compute original loss
        params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        with torch.no_grad():
            original_loss = 0
            for inputs, targets in dataloader:
                original_loss += self.criterion(self.model(inputs), targets)

        # Define the search space for tau and delta
        space = [
            Real(*tau_range, name='tau'),  # tau: threshold for low-rank approximation
            Real(*delta_range, name='delta')  # delta: quantization bin size
        ]

        # Perform Bayesian optimization
        result = gp_minimize(
            lambda x: self.evaluate_compression(x[0], x[1], dataloader, params, original_loss),
            space,  # Search space
            n_calls=n_calls,  # Number of optimization iterations
            random_state=seed
        )

        # Extract the best parameters
        best_tau, best_delta = result.x
        best_compressed_params, _ = self.compress_model(best_tau, best_delta)

        # Restore original parameters
        for name, parameter in self.model.named_parameters():
            if name in params:
                parameter.data = params[name]

        return {
            'tau': best_tau,
            'delta': best_delta,
            'compressed_params': best_compressed_params
        }