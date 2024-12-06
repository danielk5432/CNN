import numpy as np

class CNN:
    def __init__(self, conv_layers, fc_layers):
        self.conv_layers = []
        for kh, kw, in_channels, out_channels in conv_layers:
            kernels = np.random.randn(out_channels, in_channels, kh, kw) * 0.1
            biases = np.zeros(out_channels)
            self.conv_layers.append((kernels, biases))

        self.fc_weights = []
        self.fc_biases = []
        for i in range(1, len(fc_layers)):
            if fc_layers[i - 1] is None:
                self.fc_weights.append(None)  # Placeholder for input size to be determined later
            else:
                weight = np.random.randn(fc_layers[i - 1], fc_layers[i]) * 0.1
                bias = np.zeros(fc_layers[i])
                self.fc_weights.append(weight)
                self.fc_biases.append(bias)

    def initialize_fc_weights(self, input_size):
        if self.fc_weights[0] is None:
            self.fc_weights[0] = np.random.randn(input_size, self.fc_weights[1].shape[0]) * 0.1
            self.fc_biases[0] = np.zeros(self.fc_weights[1].shape[0])

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(np.float32)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def conv2d(self, x, kernel, bias, stride=1, padding=0):
        n, h, w, c = x.shape
        out_channels, in_channels, kh, kw = kernel.shape
        h_out = (h + 2 * padding - kh) // stride + 1
        w_out = (w + 2 * padding - kw) // stride + 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

        shape = (n, h_out, w_out, in_channels, kh, kw)
        strides = (x.strides[0], stride * x.strides[1], stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
        sliding_windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        conv = np.tensordot(sliding_windows, kernel, axes=([3, 4, 5], [1, 2, 3]))
        return conv + bias.reshape((1, 1, 1, out_channels))

    def conv2d_grad(self, dout, x, kernel, stride=1, padding=0):
        n, h, w, c = x.shape
        out_channels, in_channels, kh, kw = kernel.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

        dx = np.zeros_like(x)
        dkernel = np.zeros_like(kernel)
        dbias = np.sum(dout, axis=(0, 1, 2))

        for i in range(h):
            for j in range(w):
                for b in range(n):
                    h_start, h_end = i * stride, i * stride + kh
                    w_start, w_end = j * stride, j * stride + kw
                    dx[b, h_start:h_end, w_start:w_end, :] += np.sum(dout[b, i, j, :, None, None, None] * kernel, axis=0)
                    dkernel += dout[b, i, j, :, None, None, None] * x[b, h_start:h_end, w_start:w_end, None, :, :, None]

        if padding > 0:
            dx = dx[:, padding:-padding, padding:-padding, :]
        return dx, dkernel, dbias

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def forward(self, x):
        self.cache = {"conv": [], "fc": []}
        for kernel, bias in self.conv_layers:
            x = self.conv2d(x, kernel, bias, stride=1, padding=1)
            self.cache["conv"].append((x, kernel, bias))
            x = self.relu(x)

        x = self.flatten(x)
        self.cache["flatten"] = x

        # Initialize fully connected weights dynamically
        self.initialize_fc_weights(x.shape[1])

        for i, (weight, bias) in enumerate(zip(self.fc_weights, self.fc_biases)):
            self.cache["fc"].append((x, weight, bias))
            x = x @ weight + bias
            if i < len(self.fc_weights) - 1:  # Apply ReLU to all layers except the last one
                x = self.relu(x)

        return self.softmax(x)



    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        y_pred = self.forward(x)
        loss_grad = (y_pred - y) / m

        # Fully connected layers
        for i in reversed(range(len(self.fc_weights))):
            cached_x, weight, bias = self.cache["fc"][i]
            dweight = cached_x.T @ loss_grad
            dbias = np.sum(loss_grad, axis=0)
            loss_grad = loss_grad @ weight.T * self.relu_grad(cached_x)

            self.fc_weights[i] -= learning_rate * dweight
            self.fc_biases[i] -= learning_rate * dbias

        # Flatten
        loss_grad = loss_grad.reshape(self.cache["flatten"].shape)

        # Convolutional layers
        for i in reversed(range(len(self.conv_layers))):
            cached_x, kernel, bias = self.cache["conv"][i]
            loss_grad, dkernel, dbias = self.conv2d_grad(loss_grad, cached_x, kernel, stride=1, padding=1)

            self.conv_layers[i] = (kernel - learning_rate * dkernel, bias - learning_rate * dbias)

    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for start in range(0, x.shape[0], batch_size):
                end = start + batch_size
                batch_x, batch_y = x[start:end], y[start:end]
                self.backward(batch_x, batch_y, learning_rate)
            print(f"Epoch {epoch + 1} completed.")

    def evaluate(self, x, y):
        predictions = self.forward(x)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        return accuracy
