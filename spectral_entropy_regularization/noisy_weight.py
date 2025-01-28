import torch
import torch.nn as nn

class NoisyWeightWrapper(nn.Module):
    def __init__(self, model: nn.Module, noise_scale: float = 0.01):
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
        # If not in training mode, return the model output without noise
        if not self.model.training:
            return self.model(*args, **kwargs)
        
        # Add noise to the weights (without tracking gradients)
        noise_dict = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad: # Only add noise to trainable weights
                    noise = torch.randn_like(param) * self.noise_scale  # Generate Gaussian noise
                    param.data = param.data + noise
                    noise_dict[name] = noise  # Save the noise for later removal

        # Perform the forward pass with noisy weights
        output = self.model(*args, **kwargs)

        # Remove the noise from the weights (restore original weights)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    noise = noise_dict[name]
                    param.data = param.data - noise

        return output