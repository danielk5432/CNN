import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return x > 0

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    res = np.exp(x - max)
    return res / np.sum(res, axis=1, keepdims=True)

def conv2d(x, kernel):
    """Optimized 2D Convolution with Numpy."""
    batch_size, h, w, c = x.shape
    kh, kw, in_channels, out_channels = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1  # Output height and width

    # Create sliding windows for the input tensor
    windows = np.lib.stride_tricks.sliding_window_view(x, (1, kh, kw, c))  # Add batch dimension
    windows = windows[:, :oh, :ow, :, :, :]  # Trim to valid convolution size
    windows = windows.reshape(batch_size, oh, ow, -1)  # Flatten spatial dims

    # Perform convolution using tensordot
    kernel_flat = kernel.reshape(-1, out_channels)  # Flatten kernel
    output = np.tensordot(windows, kernel_flat, axes=([3], [0]))

    return output




def max_pool(x, pool_size):
    """Performs max pooling."""
    batch_size, h, w, c = x.shape
    ph, pw = pool_size
    h_out, w_out = h // ph, w // pw
    out = np.zeros((batch_size, h_out, w_out, c))

    for b in range(batch_size):
        for i in range(h_out):
            for j in range(w_out):
                region = x[b, i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
                out[b, i, j, :] = np.max(region, axis=(0, 1))

    return out


class CNN:
    def __init__(self, conv_layers, fc_layers):
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernels = []
        self.weight = []
        self.bias = []

        # Initialize convolutional layers
        for kernel_shape in self.conv_layers:
            self.kernels.append(np.random.randn(*kernel_shape) * 0.01)

        # Initialize fully connected layers
        for i in range(len(self.fc_layers) - 1):
            self.weight.append(np.random.randn(self.fc_layers[i], self.fc_layers[i+1]) * 0.01)
            self.bias.append(np.zeros((1, self.fc_layers[i+1])))

    def forward(self, x):
        self.conv_activations = []
        self.pool_activations = []
        self.z_values = []
        self.activations = []

        # Convolutional layers
        for kernel in self.kernels:
            x = relu(conv2d(x, kernel))
            self.conv_activations.append(x)
            x = max_pool(x, (2, 2))  # Assuming 2x2 pooling
            self.pool_activations.append(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        self.activations = [x]
        for w, b in zip(self.weight, self.bias):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            if w is self.weight[-1]:
                self.activations.append(softmax(z))  # Softmax for final layer
            else:
                self.activations.append(relu(z))

        return self.activations[-1]


    def backward(self, x, y, learning_rate):
        m = y.shape[0]  # Batch size
        dz = self.activations[-1] - y  # Derivative of softmax-crossentropy

        for i in reversed(range(len(self.weight))):
            dw = (self.activations[i].T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = (dz @ self.weight[i].T) * relu_d(self.z_values[i-1])

            # Update weights and biases
            self.weight[i] -= learning_rate * dw
            self.bias[i] -= learning_rate * db

        # Gradients for convolutional layers can be added similarly

    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]
            print(x.shape, batch_size)
            for i in range(0, x.shape[0], batch_size):
                
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)

            # Calculate and print loss after each epoch
            logits = self.forward(x)
            loss = -np.mean(np.sum(y * np.log(logits + 1e-9), axis=1))
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")



    def evaluate(self, x, y):
        logits = self.forward(x)
        predictions = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
