import torch
import torch.nn as nn

class NoisyWeightWrapper(nn.Module):
    def __init__(self, model, noise_scale=0.01):
        """
        Wrapper for adding Gaussian noise to model weights during the forward pass.
        
        Args:
            model (nn.Module): The neural network model to wrap.
            noise_scale (float): Standard deviation of the Gaussian noise to add.
        """
        super(NoisyWeightWrapper, self).__init__()
        self.model = model
        self.noise_scale = noise_scale

    def forward(self, *args, **kwargs):
        """
        Forward pass with noisy weights.
        """
        # Add noise to the weights in-place (without tracking gradients)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:  # Only add noise to weight parameters
                    noise = torch.randn_like(param) * self.noise_scale  # Generate Gaussian noise
                    param.add_(noise)  # Add noise to the weights in-place

        # Perform the forward pass with noisy weights
        output = self.model(*args, **kwargs)

        # Remove the noise from the weights in-place (restore original weights)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    param.sub_(noise)  # Subtract the noise to restore original weights

        return output