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

def sigmoid(tensor: Tensor):
    _sigmoid = lambda x: 1 / (1 + np.exp(-x))
    data = _sigmoid(tensor.data)
    out = Tensor(data, _children = (tensor, ), _op = "sigmoid")

    def _backward():
        tensor.grad += data * (1 - data)
    
    out._backward = _backward
    return out
