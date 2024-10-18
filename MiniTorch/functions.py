from MiniTorch.tensor import Tensor
import numpy as np


def relu(tensor: Tensor):
    data = tensor.data.copy()
    data[data <= 0] = 0

    out = Tensor(data, _children=(tensor,), _op = "relu")
    def _backward():
        tensor.grad = out.grad * (data > 0)

    out._backward = _backward
    return out
