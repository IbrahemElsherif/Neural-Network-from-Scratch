# Neural Network from Scratch
Implementing a Neural Network from Scratch using only NumPy.

This simple two-layer neural network is implemented using NumPy and trained on the famous MNIST dataset. It's a good practice for anyone wanting to deepen their understanding of the intuition behind neural networks, gradient descent, and backward and forward propagation.

We implement the following four steps:

1. Implement forward propagation.
2. Implement backward propagation.
3. Update parameters using gradient descent.  
4. Repeat until convergence.

An accuracy of 84% was achieved on both the training and test sets.

Here is the math behind the full network:

### **Forward propagation**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$  
$$A^{[1]} = g_{\text{ReLU}}(Z^{[1]})$$  
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$  
$$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$  

### **Backward propagation**

$$dZ^{[2]} = A^{[2]} - Y$$  
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$  
$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$  
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g^{[1]\prime} (Z^{[1]})$$  
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$  
$$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$  

### **Parameter updates**

$$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$  
$$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$  
$$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$  
$$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$  

---

Credit to Samson Zhang for his amazing tutorial: [YouTube link](https://youtu.be/w8yWXqWQYmU)
