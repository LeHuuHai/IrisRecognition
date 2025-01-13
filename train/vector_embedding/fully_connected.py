from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim=128, margin=1.0):
        # dense layer
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.margin = margin

    def forward(self, x):
        x = self.fc1(x)
        return x