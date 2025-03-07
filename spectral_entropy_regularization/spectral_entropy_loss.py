import torch
import torch.nn as nn

class SpectralEntropyLoss(nn.Module):
    def __init__(
        self,
        maximum_dimension: int = 2,
        reduction: str = "mean"
    ):
        """
        Spectral Entropy Loss for regularization.
        
        Args:
            maximum_dimension (int): Maximum dimension of weight matrices for which to compute spectral entropy. Defaults to 2.
        """
        super(SpectralEntropyLoss, self).__init__()
        assert maximum_dimension >= 2, "The maximum dimension must be at least 2."
        assert reduction in ["mean", "sum"], "Reduction must be either 'mean' or 'sum'."
        self.maximum_dimension = maximum_dimension
        self.reduction = reduction

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
        normalized_singular_values = singular_values / torch.sum(singular_values, dim=1).unsqueeze(1)
        
        # Compute spectral entropy
        spectral_entropy = -torch.sum(normalized_singular_values * torch.log(normalized_singular_values + torch.finfo(torch.float32).eps))
        
        return spectral_entropy

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the total spectral entropy of all weight matrices with dimensions less than or equal to the maximum dimension.
        
        Args:
            model (nn.Module): The neural network model.
        
        Returns:
            torch.Tensor: Total spectral entropy of the model's weight matrices.
        """
        total_spectral_entropy = torch.tensor(0.0)
        count = 0
        
        # Iterate over all parameters in the model
        for _, param in model.named_parameters():
            if param.ndim > 1 and param.ndim <= self.maximum_dimension:
                *_, row, col = param.shape
                total_spectral_entropy += self.spectral_entropy(param.reshape(-1, row, col))
                count += 1
        
        if self.reduction == "mean":
            return total_spectral_entropy / count if count > 0 else total_spectral_entropy
        
        return total_spectral_entropy