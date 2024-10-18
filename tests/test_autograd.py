import unittest
from MiniTorch.tensor import Tensor
import numpy as np
from MiniTorch.functions import relu, sigmoid

class TestAutograd(unittest.TestCase):

    def test_add_scalars(self):
        a = Tensor(2)
        b = Tensor(3)
        c = a + b
        c.backward()

        self.assertEqual(c.data, np.array(5))
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_add_arrays(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b                     
        c.backward() 

        self.assertTrue((c.data == np.array([5.0, 7.0, 9.0])).all())
        self.assertTrue((a.grad == np.array([1.0, 1.0, 1.0])).all())
        self.assertTrue((b.grad == np.array([1.0, 1.0, 1.0])).all())

    def test_mul_scalars(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        c.backward()

        self.assertEqual(c.data, 6)
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

    def test_mul_arrays(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a * b
        c.backward()

        self.assertTrue((c.data == np.array([4.0, 10.0, 18.0])).all())
        self.assertTrue((a.grad == np.array([4.0, 5.0, 6.0])).all())
        self.assertTrue((b.grad == np.array([1.0, 2.0, 3.0])).all())

    def test_add_and_mul_mixed(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        d = Tensor([2.0, 3.0])
        c = (a * b) + d
        c.backward()

        self.assertTrue((c.data == np.array([5.0, 11.0])).all())
        self.assertTrue((a.grad == np.array([3.0, 4.0])).all())
        self.assertTrue((b.grad == np.array([1.0, 2.0])).all())
        self.assertTrue((d.grad == np.array([1.0, 1.0])).all())

    def test_chain_rule(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b    
        d = c + a  
        d.backward()
    
        self.assertEqual(d.data, 8.0)
        self.assertEqual(a.grad, 4.0)
        self.assertEqual(b.grad, 2.0)

    def test_chain_rule_arrays(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a * b
        d = c + a
        e = d * b
        e.backward()

        self.assertTrue((a.grad == np.array([20.0, 30.0, 42.0])).all())
        self.assertTrue((b.grad == np.array([9.0, 22.0, 39.0])).all())


    def test_relu_forward(self):
        a = Tensor([-1.0, 0.0, 1.0, 2.0])
        relu_a = relu(a)

        self.assertTrue((relu_a.data == np.array([0.0, 0.0, 1.0, 2.0])).all())

    def test_relu_backward(self):
        a = Tensor([-1.0, 0.0, 1.0, 2.0])
        relu_a = relu(a)
        relu_a.backward()

        self.assertTrue((a.grad == np.array([0.0, 0.0, 1.0, 1.0])).all())

    def test_complex_relu_backward(self):
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = Tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        
        relu_a = relu(a)       
        c = relu_a * b           
        
        c.backward()

        self.assertTrue((a.grad == np.array([0.0, 0.0, 0.0, 2.0, 2.0])).all()) 
        self.assertTrue((b.grad == np.array([0.0, 0.0, 0.0, 1.0, 2.0])).all())


    def test_multiple_operations(self):
        a = Tensor([1.0, -2.0, 3.0])
        b = Tensor([0.5, 0.5, 0.5])
        c = Tensor([2.0, 2.0, 2.0])
        d = Tensor([1.0, -1.0, 0.0])

        x = a + b                 
        relu_x = relu(x)   

        y = relu_x * c           
        z = y + d         
        relu_z = relu(z)   

        relu_z.backward()

        self.assertTrue((a.grad == np.array([2.0, 0.0, 2.0])).all())  # Gradients through the layers
        self.assertTrue((b.grad == np.array([2.0, 0.0, 2.0])).all())  # Gradient similar to `a` since a + b
        self.assertTrue((c.grad == np.array([1.5, 0.0, 3.5])).all())  # Gradient from relu_x * c
        self.assertTrue((d.grad == np.array([1.0, 0.0, 1.0])).all())  # Gradient from the final addition (relu_z + d)

    def test_sigmoid_forward(self):
        a = Tensor([-2.0, 0.0, 2.0])
        sigmoid_a = sigmoid(a)

        expected = 1 / (1 + np.exp(-np.array([-2.0, 0.0, 2.0])))
        self.assertTrue((sigmoid_a.data == expected).all())

    def test_sigmoid_backward(self):
        a = Tensor([1.0, -1.0, 2.0])
        sigmoid_a = sigmoid(a)
        sigmoid_a.backward()

        sigmoid_vals = 1 / (1 + np.exp(-np.array([1.0, -1.0, 2.0])))
        expected_grad = sigmoid_vals * (1 - sigmoid_vals)
        
        self.assertTrue((a.grad == expected_grad).all())


if __name__ == '__main__':
    unittest.main()