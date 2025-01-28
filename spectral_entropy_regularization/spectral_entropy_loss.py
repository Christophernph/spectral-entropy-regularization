import torch
import torch.nn as nn

class SpectralEntropyLoss(nn.Module):
    def __init__(self):
        """
        Spectral Entropy Loss for regularization.
        """
        super(SpectralEntropyLoss, self).__init__()

    def spectral_entropy(self, weight_matrix) -> torch.Tensor:
        """
        Compute the spectral entropy of a weight matrix.
        
        Args:
            weight_matrix (torch.Tensor): A 2D weight matrix (e.g., from a linear layer).
        
        Returns:
            torch.Tensor: Spectral entropy of the weight matrix.
        """
        # Compute Singular Value Decomposition (SVD)
        singular_values = torch.linalg.svdvals(weight_matrix)
        
        # Normalize singular values to form a probability distribution
        normalized_singular_values = singular_values / torch.sum(singular_values)
        
        # Compute spectral entropy
        spectral_entropy = -torch.sum(normalized_singular_values * torch.log(normalized_singular_values + torch.finfo(torch.float32).eps))
        
        return spectral_entropy

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the total spectral entropy of all 2D weight matrices in the model.
        
        Args:
            model (nn.Module): The neural network model.
        
        Returns:
            torch.Tensor: Total spectral entropy of the model's weight matrices.
        """
        total_spectral_entropy = 0.0
        
        # Iterate over all parameters in the model
        for _, param in model.named_parameters():
            if param.ndim == 2:  # Only apply to 2D weight matrices
                total_spectral_entropy += self.spectral_entropy(param)
        
        return total_spectral_entropy