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
            print(f"Initializing first FC layer with input size {input_size} and output size {self.fc_weights[1].shape[0]}")  # Debugging
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

        print(f"Input shape before padding: {x.shape}")  # Debugging: Input shape before padding
        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        print(f"Input shape after padding: {x.shape}")  # Debugging: Input shape after padding

        shape = (n, h_out, w_out, in_channels, kh, kw)
        strides = (x.strides[0], stride * x.strides[1], stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
        input_windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        print(f"Shape of input_windows after slicing: {input_windows.shape}")  # Debugging

        conv = np.tensordot(input_windows, kernel, axes=([3, 4, 5], [1, 2, 3]))
        conv += bias.reshape((1, 1, 1, out_channels))
        print(f"Output shape after convolution: {conv.shape}")  # Debugging
        return conv

    def flatten(self, x):
        print(f"Shape before flattening: {x.shape}")  # Debugging
        flattened = x.reshape(x.shape[0], -1)
        print(f"Shape after flattening: {flattened.shape}")  # Debugging
        return flattened

    def forward(self, x):
        self.cache = {"conv": [], "fc": []}
        for idx, (kernel, bias) in enumerate(self.conv_layers):
            x = self.conv2d(x, kernel, bias, stride=1, padding=1)
            print(f"After convolution {idx + 1}: {x.shape}")  # Debugging
            self.cache["conv"].append((x, kernel, bias))
            x = self.relu(x)
            print(f"After ReLU activation {idx + 1}: {x.shape}")  # Debugging

        x = self.flatten(x)
        print(f"After flattening: {x.shape}")  # Debugging
        self.cache["flatten"] = x

        # Initialize fully connected weights dynamically
        self.initialize_fc_weights(x.shape[1])

        for i, (weight, bias) in enumerate(zip(self.fc_weights, self.fc_biases)):
            print(f"Layer {i + 1}: Input size {x.shape}, Weight shape {weight.shape}, Bias shape {bias.shape}")  # Debugging
            x = x @ weight + bias
            print(f"Layer {i + 1}: Output size {x.shape}")  # Debugging
            if i < len(self.fc_weights) - 1:  # Apply ReLU to all layers except the last one
                x = self.relu(x)
                print(f"Layer {i + 1}: After ReLU {x.shape}")  # Debugging


        x = self.softmax(x)
        print(f"Output after softmax: {x.shape}")  # Debugging
        return x


    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        y_pred = self.forward(x)
        print(f"Shape of y_pred: {y_pred.shape}")  # Debugging
        print(f"Shape of y: {y.shape}")  # Debugging
        loss_grad = (y_pred - y) / m

        # Fully connected layers
        for i in reversed(range(len(self.fc_weights))):
            cached_x, weight, bias = self.cache["fc"][i]
            dweight = cached_x.T @ loss_grad
            dbias = np.sum(loss_grad, axis=0)
            loss_grad = loss_grad @ weight.T * self.relu_grad(cached_x)
            print(f"Gradient shapes - dweight: {dweight.shape}, dbias: {dbias.shape}")  # Debugging

            self.fc_weights[i] -= learning_rate * dweight
            self.fc_biases[i] -= learning_rate * dbias

        loss_grad = loss_grad.reshape(self.cache["flatten"].shape)

        # Convolutional layers
        for idx in reversed(range(len(self.conv_layers))):
            cached_x, kernel, bias = self.cache["conv"][idx]
            loss_grad, dkernel, dbias = self.conv2d_grad(loss_grad, cached_x, kernel, stride=1, padding=1)
            print(f"Conv Layer {idx + 1} Gradients - dkernel: {dkernel.shape}, dbias: {dbias.shape}")  # Debugging

            self.conv_layers[idx] = (kernel - learning_rate * dkernel, bias - learning_rate * dbias)

    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for start in range(0, x.shape[0], batch_size):
                end = start + batch_size
                batch_x, batch_y = x[start:end], y[start:end]
                self.backward(batch_x, batch_y, learning_rate)
            print(f"Epoch {epoch + 1} completed.")  # Debugging

    def evaluate(self, x, y):
        predictions = self.forward(x)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
        print(f"Evaluation accuracy: {accuracy:.4f}")  # Debugging
        return accuracy
