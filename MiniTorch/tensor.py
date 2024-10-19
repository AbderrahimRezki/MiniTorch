import numpy as np
from collections import deque

class Tensor:
    def __init__(self, data, _children = (), _op = None):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._prev = _children
        self._op = _op

        self._backward = lambda : None

    def __add__(self, other):
        out = Tensor(self.data + other.data, _children=(self, other), _op = "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, _children=(self, other), _op = "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _children = (self, other), _op = "@")

        def _backward():
            self.grad += other.data.T * out.grad
            other.grad += self.data.T * out.grad

        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def backward(self):
        self.grad = 1

        visited = set()
        queue = deque([self])

        while queue:
            current = queue.pop()
            current._backward()

            for child in current._prev:
                if child not in visited:
                    queue.append(child)

    def __repr__(self):
        return f"Tensor(data = {self.data}, _children = {self._prev}, _op = {self._op})"

class Parameter(Tensor):
    def __init__(self, n_in, n_out):
        data = np.random.normal(1, 1 / n_in, size = (n_in, n_out))
        super().__init__(data)

    def __call__(self, x: Tensor):
        return self @ x

    def step(self, alpha = 0.1):
        self.data -= alpha * self.grad
