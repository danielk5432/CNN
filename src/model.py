import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return x > 0

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    res = np.exp(x - max)
    return res / np.sum(res, axis=1, keepdims=True)

class NN:
    def __init__(self, layers):
        self.layer = layers
        self.weight = []
        self.bias = []
        for i in range(len(self.layer) - 1):
            self.weight.append(np.random.randn(self.layer[i], self.layer[i+1]) * 0.01)
            self.bias.append(np.zeros((1, self.layer[i+1])))


    def forward(self, x):
        self.activations = [x]
        self.z_values = []

        for w, b in zip(self.weight, self.bias):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            if w is self.weight[-1]:
                self.activations.append(softmax(z))
            else:
                self.activations.append(relu(z))

        return self.activations[-1]

    def backward(self, x, y, learning_rate):
        m = y.shape[0]
        dz = self.activations[-1] - y

        for i in reversed(range(len(self.weight))):
            dw = (self.activations[i].T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = (dz @ self.weight[i].T) * relu_d(self.z_values[i-1])

            self.weight[i] -= learning_rate * dw
            self.bias[i] -= learning_rate * db

    def train(self, x, y, epochs, b_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for i in range(0, x.shape[0], b_size):
                x_batch = x[i:i+b_size]
                y_batch = y[i:i+b_size]
                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)

            
            loss = -np.mean(np.sum(y * np.log(self.forward(x)), axis=1))
            print(f"Epoch {epoch}, Loss: {loss}")

    def evaluate(self, x, y):
        predictions = np.argmax(self.forward(x), axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
