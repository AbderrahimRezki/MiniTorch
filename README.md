# MiniTorch

**MiniTorch** is a minimalistic autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd), offering a PyTorch-like API for building and training neural networks.
It's designed as an educational tool to help understand the core mechanics of automatic differentiation and neural network training.

## Features

- **Autograd Engine**: Implements reverse-mode automatic differentiation to compute gradients.
- **Neural Network Modules**: Provides basic building blocks like layers and activation functions.
- **Training Loop**: Includes a simple training loop to train models on datasets.
- **PyTorch-like API**: Offers an interface similar to PyTorch for ease of use and familiarity.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AbderrahimRezki/MiniTorch.git
   cd MiniTorch
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**:

   ```bash
   pip install -e .
   ```

## Usage

Here's a simple example of how to define and train a neural network using MiniTorch:

```python
from MiniTorch import Tensor, Module, Linear, MSELoss
import numpy as np

# Define a simple neural network
class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1, 10)
        self.fc2 = Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

# Generate some dummy data
x = Tensor(np.random.randn(100, 1))
y = Tensor(3 * x.data + 2 + 0.1 * np.random.randn(100, 1))

# Initialize the model and loss function
model = SimpleNet()
criterion = MSELoss()

# Training loop
for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    model.step(lr=0.01)
    model.zero_grad()
    print(f"Epoch {epoch}: Loss = {loss.data}")
```

## Project Structure

```
MiniTorch/
├── MiniTorch/           # Core library code
│   ├── __init__.py
│   ├── autograd.py      # Autograd engine implementation
│   ├── tensor.py        # Tensor class with support for operations
│   ├── nn.py            # Neural network modules (e.g., Linear, ReLU)
│   └── optim.py         # Optimizers (e.g., SGD)
├── tests/               # Unit tests for the library
│   └── test_tensor.py
├── examples/            # Example scripts demonstrating usage
│   └── train.py
├── setup.py             # Setup script for installation
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
