    a = Tensor([1.0, -2.0, 3.0])
        b = Tensor([0.5, 0.5, 0.5])
        c = Tensor([2.0, 2.0, 2.0])
        d = Tensor([1.0, -1.0, 0.0])

        # Layer 1: Element-wise addition and ReLU
        x = a + b                 # x = [1.5, -1.5, 3.5]
        relu_x = x.relu()         # relu_x = [1.5, 0.0, 3.5]

        # Layer 2: Element-wise multiplication
        y = relu_x * c            # y = [3.0, 0.0, 7.0]

        # Layer 3: Addition and ReLU
        z = y + d                 # z = [4.0, -1.0, 7.0]
        relu_z = z.relu()         # relu_z = [4.0, 0.0, 7.0]

        # Backpropagation
        relu_z.backward()

        # Expected gradients
        self.assertTrue((a.grad == np.array([2.0, 0.0, 2.0])).all())  # Gradients through the layers
        self.assertTrue((b.grad == np.array([2.0, 0.0, 2.0])).all())  # Gradient similar to `a` since a + b
        self.assertTrue((c.grad == np.array([1.5, 0.0, 3.5])).all())  # Gradient from relu_x * c
        self.assertTrue((d.grad == np.array([1.0, 0.0, 1.0])).all())  # Gradient from the final addition (relu_z + d)
