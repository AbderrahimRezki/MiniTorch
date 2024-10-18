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
        return f"Tensor(data = {self.data}, _children = ({self._prev}), _op = {self._op})"