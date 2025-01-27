import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectral_entropy_regularization import NoisyWeightWrapper

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 128)
        self.targets = (self.data.sum(dim=1) > 0).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    
    model = SimpleModel()
    noisy_model = NoisyWeightWrapper(model, noise_scale=0.01)  # Adjust noise_scale as needed

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(noisy_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # Primary loss (e.g., cross-entropy for classification)

    for epoch in range(10):  # Example training loop
        for inputs, targets in dataloader: 
            optimizer.zero_grad()
            
            # Forward pass with noisy weights
            outputs = noisy_model(inputs)
            
            # Compute loss and perform backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Update weights
            optimizer.step()

if __name__ == "__main__":
    main()