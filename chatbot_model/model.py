# Import necessary libraries from PyTorch
import torch
import torch.nn as nn

# Define the neural network class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Define the first linear layer
        self.l1 = nn.Linear(input_size, hidden_size) 
        # Define the second linear layer
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        # Define the third linear layer
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Apply the first linear layer followed by ReLU activation
        out = self.l1(x)
        out = self.relu(out)
        # Apply the second linear layer followed by ReLU activation
        out = self.l2(out)
        out = self.relu(out)
        # Apply the third linear layer
        out = self.l3(out)
        # No activation and no softmax at the end
        return out
