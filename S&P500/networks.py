from imports import *
from torch_imports import *

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

class OneLayerNeuralNetwork(nn.Module):
    def __init__(self, input_size=5, output_size=1):
        super().__init__()
        self.input = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.input(x)
        return x

    def learned_func(self):
        print(poly_desc(self.input.weight.view(-1), self.input.bias))

    def weight(self):
        return self.input.weight

class TwoLayerNeuralNetwork(nn.Module):
    def __init__(self, input_size=5, layer1_size=32, layer2_size=16, output_size=1):
        super().__init__()

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)

    def forward(self, x):
        x = F.dropout(self.layer1(x), p=0.9)
        x = F.relu(x)
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x
