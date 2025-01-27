import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectral_entropy_regularization import SpectralEntropyLoss

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
        self.targets = self.data.sum(dim=1) > 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    
    model = SimpleModel()

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion1 = nn.CrossEntropyLoss()  # Primary loss (e.g., cross-entropy for classification)
    criterion2 = SpectralEntropyLoss()  # Secondary loss (spectral entropy for regularization)
    beta = 0.1  # Weight for spectral entropy loss

    for epoch in range(10):  # Example training loop
        for inputs, targets in dataloader: 
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Compute loss and perform backward pass
            loss = criterion1(outputs, targets)
            loss += criterion2(model) * beta  # Add spectral entropy loss

            loss.backward()
            
            # Update weights
            optimizer.step()

if __name__ == "__main__":
    main()