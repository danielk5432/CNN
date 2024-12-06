import numpy as np

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # Stability improvement
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Convolution operation
def conv2d(x, kernel, padding='same'):
    batch_size, h, w, c = x.shape
    kh, kw, in_channels, out_channels = kernel.shape

    if padding == 'same':
        pad_h = (kh - 1) // 2 if kh % 2 == 1 else kh // 2
        pad_w = (kw - 1) // 2 if kw % 2 == 1 else kw // 2
        x = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)

    h, w = x.shape[1:3]
    oh, ow = h - kh + 1, w - kw + 1
    output = np.zeros((batch_size, oh, ow, out_channels))

    for b in range(batch_size):
        for i in range(oh):
            for j in range(ow):
                region = x[b, i:i+kh, j:j+kw, :]
                for k in range(out_channels):
                    output[b, i, j, k] = np.sum(region * kernel[:, :, :, k])
    
    return output

# Max Pooling operation
def max_pool(x, pool_size):
    batch_size, h, w, c = x.shape
    ph, pw = pool_size
    h_out, w_out = h // ph, w // pw
    out = np.zeros((batch_size, h_out, w_out, c))

    for b in range(batch_size):
        for i in range(h_out):
            for j in range(w_out):
                region = x[b, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw, :]
                out[b, i, j, :] = np.max(region, axis=(0, 1))

    return out

# Backpropagation for Max Pooling
def max_pool_backward(dout, original, pool_size):
    batch_size, h, w, c = original.shape
    ph, pw = pool_size
    h_out, w_out = dout.shape[1:3]
    dinput = np.zeros_like(original)

    for b in range(batch_size):
        for i in range(h_out):
            for j in range(w_out):
                for k in range(c):
                    region = original[b, i*ph:(i+1)*ph, j*pw:(j+1)*pw, k]
                    max_idx = np.unravel_index(np.argmax(region), region.shape)
                    dinput[b, i*ph + max_idx[0], j*pw + max_idx[1], k] += dout[b, i, j, k]

    return dinput

# Backpropagation for Convolution
def conv2d_backward(dout, x, kernel):
    batch_size, h, w, c = x.shape
    kh, kw, in_channels, out_channels = kernel.shape
    oh, ow = dout.shape[1:3]

    dkernel = np.zeros_like(kernel)
    dx = np.zeros_like(x)

    for b in range(batch_size):
        for i in range(oh):
            for j in range(ow):
                region = x[b, i:i+kh, j:j+kw, :]
                for k in range(out_channels):
                    dkernel[:, :, :, k] += region * dout[b, i, j, k]
                for ch in range(in_channels):
                    dx[b, i:i+kh, j:j+kw, ch] += np.sum(kernel[:, :, ch, :] * dout[b, i, j, :], axis=-1)
    
    return dx, dkernel

# CNN Class
class CNN:
    def __init__(self, conv_layers, fc_layers):
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernels = []
        self.weight = []
        self.bias = []
        self.flatten_size = None  # Flatten size 초기화

        # Initialize convolutional layers
        for kernel_shape in self.conv_layers:
            self.kernels.append(np.random.randn(*kernel_shape) * 0.01)

    def initialize_fc_layers(self):
        """Fully Connected Layer 초기화 (Forward Pass 후 호출)"""
        if self.flatten_size is None:
            raise ValueError("Flatten size is not set. Run forward pass first.")

        self.fc_layers = [self.flatten_size] + self.fc_layers
        for i in range(len(self.fc_layers) - 1):
            self.weight.append(np.random.randn(self.fc_layers[i], self.fc_layers[i + 1]) * 0.01)
            self.bias.append(np.zeros((1, self.fc_layers[i + 1])))

    def forward(self, x):
        self.conv_activations = []
        self.pool_activations = []
        self.flatten_shape = None
        self.z_values = []
        self.activations = []

        # Convolutional layers
        for kernel in self.kernels:
            x = relu(conv2d(x, kernel))
            self.conv_activations.append(x)
            x = max_pool(x, (2, 2))  # Assuming 2x2 pooling
            self.pool_activations.append(x)

        # Save shape before flatten
        self.flatten_shape = x.shape
        self.flatten_size = np.prod(self.flatten_shape[1:])  # Set Flatten size

        # Fully connected layer initialization (if not already initialized)
        if not self.weight:
            self.initialize_fc_layers()

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

        # Fully connected layers
        for i in reversed(range(len(self.weight))):
            dw = (self.activations[i].T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = (dz @ self.weight[i].T) * relu_d(self.z_values[i - 1])

            # Update weights and biases
            self.weight[i] -= learning_rate * dw
            self.bias[i] -= learning_rate * db

        # Reshape dz to match the shape of the last pooling output
        print(f"Flatten Shape: {self.flatten_shape}")  # Debugging
        print(f"dz Shape (before reshape): {dz.shape}")
        expected_flatten_size = np.prod(self.flatten_shape[1:])  # Total size of a single Flatten output
        print(dz.size, expected_flatten_size * dz.shape[0])
        if dz.size != expected_flatten_size * dz.shape[0]:
            raise ValueError(f"Mismatch in flatten size. Expected: {expected_flatten_size * dz.shape[0]}, Got: {dz.size}")

        # Reshape dz to Pooling output shape
        dout = dz.reshape(self.flatten_shape[0], *self.flatten_shape[1:])
        print(f"dout Shape (after reshape): {dout.shape}")

        # Convolutional layers backward
        for i in reversed(range(len(self.kernels))):
            dout = max_pool_backward(dout, self.conv_activations[i], (2, 2))  # Backprop MaxPooling
            dout, dkernel = conv2d_backward(dout, x if i == 0 else self.pool_activations[i - 1], self.kernels[i])

            # Update kernels
            self.kernels[i] -= learning_rate * dkernel


    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)

            # Calculate and print loss after each epoch
            logits = self.forward(x)
            loss = -np.mean(np.sum(y * np.log(logits + 1e-9), axis=1))
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def evaluate(self, x, y):
        logits = self.forward(x)
        predictions = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
