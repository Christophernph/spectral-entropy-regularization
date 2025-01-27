import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectral_entropy_regularization import CompressionOptimizer

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
    
    model = SimpleModel()  # Replace with your model
    
    dataset = SimpleDataset() # Replace with your dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()  # Replace with your loss function

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # Primary loss (e.g., cross-entropy for classification)

    for epoch in range(100):  # Example training loop
        for inputs, targets in dataloader: 
            optimizer.zero_grad()
            
            # Forward pass with noisy weights
            outputs = model(inputs)
            
            # Compute loss and perform backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Update weights
            optimizer.step()


    # Optimize
    optimizer = CompressionOptimizer(model, criterion, epsilon=1.0) # adjust epsilon as needed
    result = optimizer.optimize_compression(dataloader, n_calls=10)
    print("Best tau:", result['tau'])
    print("Best delta:", result['delta'])
    
    # Compress the model (i.e. load the compressed parameters into the model)
    for name, parameter in model.named_parameters():
        if name in result['compressed_params']:
            parameter.data = result['compressed_params'][name]
    

if __name__ == '__main__':
    main()