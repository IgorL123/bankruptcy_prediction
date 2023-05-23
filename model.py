import torch
from torch import nn
import torch.nn.functional as f


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__(nn.Module)
        self.input = nn.Linear(input_dim, 20)
        self.hidden1 = nn.Linear(20, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, output_dim)

    def forward(self, x):
        x = f.relu(self.input(x))
        x = f.tanh(self.hidden1(x))
        x = f.tanh(self.hidden2(x))
        x = f.tanh(self.hidden3(x))

        return self.output(x)
