import numpy as np

class Tensor:
    def __init__(self, data, _children = (), _op = None):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._prev = _children
        self._op = op

        self._backward = lambda : None

    def __add__(self, other):
        out = Tensor(self.data + other.data, _children=(self, other), _op = "+")

        def _backward():
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, _children=(self, other), _op = "*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data = {self.data}, _children = ({self._prev}), _op = {self._op})"