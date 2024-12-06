import numpy as np


class Convolution:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(output_channels)

    def forward(self, x):
        n, h, w, c = x.shape
        print(f"Conv Input shape: {x.shape}")

        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # Sliding window
        shape = (n, h_out, w_out, c, self.kernel_size, self.kernel_size)
        strides = (x.strides[0], self.stride * x.strides[1], self.stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # Convolution
        conv_out = np.tensordot(windows, self.kernels, axes=([3, 4, 5], [1, 2, 3])) + self.biases.reshape(1, 1, 1, -1)

        print(f"Conv Output shape: {conv_out.shape}")
        return conv_out

    def backward(self, x, grad):
        """
        Backpropagation for the convolutional layer.
        Args:
            x: Input to the convolutional layer.
            grad: Gradient from the next layer.
        Returns:
            grad_input: Gradient to pass to the previous layer.
            dw: Gradient w.r.t kernels.
            db: Gradient w.r.t biases.
        """
        n, h, w, c = x.shape
        out_channels, in_channels, kh, kw = self.kernels.shape

        # Initialize gradients
        grad_input = np.zeros_like(x)
        dw = np.zeros_like(self.kernels)
        db = np.sum(grad, axis=(0, 1, 2))

        # Padding for gradient computation
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            grad_input = np.pad(grad_input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # Compute gradients using sliding window
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                window = x[:, i * self.stride:i * self.stride + kh, j * self.stride:j * self.stride + kw, :]
                dw += np.tensordot(grad[:, i, j, :], window, axes=([0], [0]))
                grad_input[:, i * self.stride:i * self.stride + kh, j * self.stride:j * self.stride + kw, :] += np.tensordot(grad[:, i, j, :], self.kernels, axes=([3], [0]))

        # Remove padding from grad_input
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return grad_input, dw, db


class MaxPooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        n, h, w, c = x.shape
        print(f"MaxPool Input shape: {x.shape}")

        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        shape = (n, h_out, w_out, c, self.pool_size, self.pool_size)
        strides = (x.strides[0], self.stride * x.strides[1], self.stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        max_out = np.max(windows, axis=(4, 5))

        print(f"MaxPool Output shape: {max_out.shape}")
        return max_out


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, x):
        print(f"FullyConnected Input shape: {x.shape}")
        output = x @ self.weights + self.biases
        print(f"FullyConnected Output shape: {output.shape}")
        return output
    
    def backward(self, x, grad):
        """
        Backpropagation for max pooling.
        Args:
            x: Input to the pooling layer.
            grad: Gradient from the next layer.
        Returns:
            grad_input: Gradient to pass to the previous layer.
        """
        n, h, w, c = x.shape
        grad_input = np.zeros_like(x)

        pool_size = self.pool_size
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                window = x[:, i * self.stride:i * self.stride + pool_size, j * self.stride:j * self.stride + pool_size, :]
                max_mask = (window == np.max(window, axis=(1, 2), keepdims=True))
                grad_input[:, i * self.stride:i * self.stride + pool_size, j * self.stride:j * self.stride + pool_size, :] += grad[:, i, j, :][:, None, None, :] * max_mask
        return grad_input

class CNN:
    def __init__(self):
        # Define layers
        self.conv1 = Convolution(input_channels=1, output_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPooling(pool_size=2, stride=2)
        self.conv2 = Convolution(input_channels=8, output_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPooling(pool_size=2, stride=2)
        self.fc1 = FullyConnected(input_size=7 * 7 * 16, output_size=100)
        self.fc2 = FullyConnected(input_size=100, output_size=10)

    def forward(self, x):
        # Layer-wise forward pass
        self.cache = {}  # To store intermediate outputs for backpropagation
        x = self.conv1.forward(x)
        x = np.maximum(0, x)  # ReLU
        self.cache["relu1"] = x
        x = self.pool1.forward(x)
        self.cache["pool1"] = x
        x = self.conv2.forward(x)
        x = np.maximum(0, x)  # ReLU
        self.cache["relu2"] = x
        x = self.pool2.forward(x)
        self.cache["pool2"] = x
        x = x.reshape(x.shape[0], -1)  # Flatten
        self.cache["flatten"] = x
        x = self.fc1.forward(x)
        self.cache["fc1"] = x
        x = self.fc2.forward(x)
        self.cache["fc2"] = x
        return self.softmax(x)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation to update weights of all layers.
        Args:
            x: Input batch (N, 28, 28, 1)
            y: One-hot encoded labels (N, 10)
            learning_rate: Learning rate for gradient descent.
        """
        # Forward pass to get predictions and intermediate cache
        y_pred = self.forward(x)

        # Compute gradient of loss w.r.t softmax input
        grad = y_pred - y  # Cross-entropy loss with softmax

        # Backpropagate through FullyConnected Layer 2
        grad, dw_fc2, db_fc2 = self.compute_fc_grad(self.cache["fc1"], self.fc2, grad)
        self.fc2.weights -= learning_rate * dw_fc2
        self.fc2.biases -= learning_rate * db_fc2

        # Backpropagate through FullyConnected Layer 1
        grad, dw_fc1, db_fc1 = self.compute_fc_grad(self.cache["flatten"], self.fc1, grad)
        self.fc1.weights -= learning_rate * dw_fc1
        self.fc1.biases -= learning_rate * db_fc1

        # Backpropagate through Pooling Layer 2
        grad = self.pool2.backward(self.cache["relu2"], grad)

        # Backpropagate through Convolution Layer 2
        grad = np.where(self.cache["relu2"] > 0, grad, 0)  # ReLU grad
        grad, dw_conv2, db_conv2 = self.conv2.backward(self.cache["pool1"], grad)
        self.conv2.kernels -= learning_rate * dw_conv2
        self.conv2.biases -= learning_rate * db_conv2

        # Backpropagate through Pooling Layer 1
        grad = self.pool1.backward(self.cache["relu1"], grad)

        # Backpropagate through Convolution Layer 1
        grad = np.where(self.cache["relu1"] > 0, grad, 0)  # ReLU grad
        grad, dw_conv1, db_conv1 = self.conv1.backward(x, grad)
        self.conv1.kernels -= learning_rate * dw_conv1
        self.conv1.biases -= learning_rate * db_conv1

    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for i in range(0, x.shape[0], batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                y_pred = self.forward(batch_x)
                loss = self.compute_loss(y_pred, batch_y)
                print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss:.4f}")

                # Backpropagation not implemented in this snippet

    @staticmethod
    def compute_fc_grad(prev_activation, fc_layer, grad):
        """
        Computes gradients for a fully connected layer.
        Args:
            prev_activation: Input to the FC layer (N, D).
            fc_layer: FullyConnected object.
            grad: Gradient from the next layer (N, M).
        Returns:
            grad_input: Gradient to pass to the previous layer.
            dw: Gradient w.r.t weights.
            db: Gradient w.r.t biases.
        """
        dw = prev_activation.T @ grad
        db = np.sum(grad, axis=0)
        grad_input = grad @ fc_layer.weights.T
        return grad_input, dw, db
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss