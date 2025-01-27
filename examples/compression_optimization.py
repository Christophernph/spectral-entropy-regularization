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
        self.targets = self.data.sum(dim=1) > 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    
    model = SimpleModel()  # Replace with your model
    
    dataset = SimpleDataset() # Replace with your dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()  # Replace with your loss function

    optimizer = CompressionOptimizer(model, epsilon=1.0) # adjust epsilon as needed

    # Optimize
    result = optimizer.optimize_compression(dataloader, criterion, n_calls=10)
    print("Best tau:", result['tau'])
    print("Best delta:", result['delta'])

    # Compress the model
    for (name, param), compressed_param in zip(model.named_parameters(), result['compressed_params']):
        param.copy_(compressed_param)
    

if __name__ == '__main__':
    main()