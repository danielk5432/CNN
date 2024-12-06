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
def conv2d_backward(dout, x, kernel, stride=1):
    """
    Backward pass for 2D convolution.
    Args:
        dout: Gradient of the output of the convolution (N, out_h, out_w, out_channels).
        x: Input to the convolutional layer (N, H, W, C).
        kernel: Convolution kernel (kh, kw, in_channels, out_channels).
        stride: Stride of the convolution.
    Returns:
        dx: Gradient of the input (N, H, W, C).
        dkernel: Gradient of the kernel (kh, kw, in_channels, out_channels).
    """
    batch_size, h, w, c = x.shape
    kh, kw, in_channels, out_channels = kernel.shape
    _, out_h, out_w, _ = dout.shape

    dx = np.zeros_like(x)
    dkernel = np.zeros_like(kernel)

    # Pad x and dx to account for stride and kernel application
    padded_x = np.pad(x, [(0, 0), (kh-1, kh-1), (kw-1, kw-1), (0, 0)], mode='constant')
    padded_dx = np.pad(dx, [(0, 0), (kh-1, kh-1), (kw-1, kw-1), (0, 0)], mode='constant')

    # Compute dkernel
    for i in range(out_h):
        for j in range(out_w):
            region = padded_x[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :]
            for n in range(batch_size):
                dkernel += np.tensordot(region[n], dout[n, i, j, :], axes=([0], [0]))

    # Compute dx
    flipped_kernel = np.flip(kernel, axis=(0, 1))
    for i in range(h):
        for j in range(w):
            region = dout[:, i:i+kh, j:j+kw, :]
            for n in range(batch_size):
                dx[n, i, j, :] += np.tensordot(region[n], flipped_kernel, axes=([3], [3]))

    return dx, dkernel

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

        # Debugging: Input size
        print(f"Input shape: {x.shape}")

        # Convolutional layers
        for idx, kernel in enumerate(self.kernels):
            x = relu(conv2d(x, kernel))
            self.conv_activations.append(x)
            print(f"After convolution {idx + 1}: {x.shape}")  # Debugging: Shape after convolution
            
            x = max_pool(x, (2, 2))  # Assuming 2x2 pooling
            self.pool_activations.append(x)
            print(f"After pooling {idx + 1}: {x.shape}")  # Debugging: Shape after pooling

        
        self.pool_activations.append(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)
        print(f"After flattening: {x.shape}")  # Debugging: Shape after flattening
        flatten_input = self.pool_activations[-1]
        self.activations.append(flatten_input.reshape(flatten_input.shape[0], -1))

        # Fully connected layers
        self.activations = [x]
        for idx, (w, b) in enumerate(zip(self.weight, self.bias)):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            print(f"After fully connected layer {idx + 1}: {z.shape}")  # Debugging: Shape after FC layer
            
            if w is self.weight[-1]:
                self.activations.append(softmax(z))  # Softmax for final layer
            else:
                self.activations.append(relu(z))

        print(f"Output shape: {self.activations[-1].shape}")  # Debugging: Final output shape
        return self.activations[-1]



    def backward(self, x, y, learning_rate):
        m = y.shape[0]  # Batch size
        print(f"Initial input shape: {x.shape}")  # Debug: Input shape
        
        dz = self.activations[-1] - y  # Derivative of softmax-crossentropy
        print(f"Shape after loss derivative (dz): {dz.shape}")  # Debug: dz shape

        # Backprop through fully connected layers
        for i in reversed(range(len(self.weight))):
            print(f"\n[Fully Connected Layer {i}]")
            print(f"Weight shape: {self.weight[i].shape}")
            print(f"Activation shape: {self.activations[i].shape}")
            
            dw = (self.activations[i].T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            print(f"Gradient dw shape: {dw.shape}, db shape: {db.shape}")
            
            if i > 0:  # 중간층 및 입력층
                dz = (dz @ self.weight[i].T) * relu_d(self.z_values[i - 1])
                print(f"dz shape after backprop through layer {i}: {dz.shape}")
            else:  # 최종 FC 레이어 0 (활성화 미분 포함!)
                dz = dz @ self.weight[i].T  # 이 경우 relu_d 반드시 곱해야 함
                print(f"dz shape after backprop through layer {i} (relu_d applied): {dz.shape}")

            # Update weights and biases
            self.weight[i] -= learning_rate * dw
            self.bias[i] -= learning_rate * db
        

        # Backprop through convolutional layers
        for i in reversed(range(len(self.kernels))):
            print(f"\n[Convolutional Layer {i}]")
            print(f"Kernel shape: {self.kernels[i].shape}")
            print(f"Pooling activation shape: {self.pool_activations[i].shape}")
            
            dz = dz.reshape(self.pool_activations[i].shape)
            print(f"Shape after reshaping for un-pooling (dz): {dz.shape}")
            
            dz = np.repeat(np.repeat(dz, 2, axis=1), 2, axis=2)  # Un-pooling
            print(f"Shape after un-pooling (dz): {dz.shape}")

            dz *= relu_d(self.conv_activations[i])  # Derivative of ReLU
            print(f"Shape after applying ReLU derivative (dz): {dz.shape}")
            
            dx, dkernel = conv2d_backward(dz, x, self.kernels[i])
            print(f"dx shape: {dx.shape}, dkernel shape: {dkernel.shape}")
            
            self.kernels[i] -= learning_rate * dkernel


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