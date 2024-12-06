import numpy as np
import pickle
import sys

def im2col(x, kernel_size, stride, padding):
    n, h, w, c = x.shape
    h_out = (h + 2 * padding - kernel_size) // stride + 1
    w_out = (w + 2 * padding - kernel_size) // stride + 1

    # Padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Extract sliding windows
    shape = (n, h_out, w_out, c, kernel_size, kernel_size)
    strides = (x.strides[0], stride * x.strides[1], stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return windows.reshape(n * h_out * w_out, -1), h_out, w_out  # Flatten windows


def col2im(cols, input_shape, kernel_size, stride, padding, h_out, w_out):
    """
    Converts column representation back to the original image tensor.

    Args:
        cols (numpy.ndarray): Columns matrix (flattened patches).
        input_shape (tuple): Shape of the original input image (n, h, w, c).
        kernel_size (int or tuple): Size of the convolution kernel (h_kernel, w_kernel).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        h_out (int): Height of the output feature map.
        w_out (int): Width of the output feature map.

    Returns:
        numpy.ndarray: Reconstructed image tensor of shape `input_shape`.
    """
    n, h, w, c = input_shape
    h_kernel, w_kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    h_padded, w_padded = h + 2 * padding, w + 2 * padding

    # Initialize padded image tensor
    x_padded = np.zeros((n, h_padded, w_padded, c))

    # Reshape cols to match sliding window dimensions
    cols_reshaped = cols.reshape(n, h_out, w_out, h_kernel, w_kernel, c)

    # Loop over output dimensions to add contributions back to the padded image
    for i in range(h_out):
        for j in range(w_out):
            x_padded[
                :, 
                i * stride:i * stride + h_kernel, 
                j * stride:j * stride + w_kernel, 
                :
            ] += cols_reshaped[:, i, j, :, :, :]

    # Remove padding if applicable
    if padding > 0:
        x_padded = x_padded[:, padding:-padding, padding:-padding, :]

    return x_padded



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
        self.input_shape = x.shape

        self.x_cols, self.h_out, self.w_out = im2col(
            x, self.kernel_size, self.stride, self.padding
        )

        #print(f"forward Conv Input shape: {x.shape}, im2col shape: {self.x_cols.shape}")

        kernels_reshaped = self.kernels.reshape(self.output_channels, -1)
        conv_out = self.x_cols @ kernels_reshaped.T + self.biases  # Matrix multiplication

        #print(f"forward Conv output shape: {conv_out.shape}")

        return conv_out.reshape(self.input_shape[0], self.h_out, self.w_out, self.output_channels)

    def backward(self, grad):
        # x_cols는 Forward에서 계산한 결과를 재활용
        
        # Gradients for biases
        db = np.sum(grad, axis=(0, 1, 2))
        
        #print(f"backward convolution input shape: {grad.shape}, x_cols shape: {self.x_cols.shape}")

        # Gradients for kernels
        grad_reshaped = grad.transpose(0, 3, 1, 2).reshape(-1, self.output_channels)  # (N, H_out, W_out, C_out) -> (N*H_out*W_out, C_out)

        dw = grad_reshaped.T @ self.x_cols
        dw = dw.reshape(self.kernels.shape)

        # Gradients for input
        kernels_reshaped = self.kernels.reshape(self.output_channels, -1)
        grad_input_cols = grad_reshaped @ kernels_reshaped
        #print(f"grad_input_cols shape: {grad_input_cols.shape}, x_cols shape: {self.x_cols.shape}")
        grad_input = col2im(
            grad_input_cols, self.input_shape, self.kernel_size, self.stride, self.padding, self.h_out, self.w_out
        )

        return grad_input, dw, db




class MaxPooling:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        n, h, w, c = x.shape
        #print(f"MaxPool Input shape: {x.shape}")

        # Calculate output dimensions
        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        # Add padding if necessary
        pad_h = max(0, (h_out - 1) * self.stride + self.pool_size - h)
        pad_w = max(0, (w_out - 1) * self.stride + self.pool_size - w)

        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        h_out = (x.shape[1] - self.pool_size) // self.stride + 1
        w_out = (x.shape[2] - self.pool_size) // self.stride + 1

        # Create sliding windows
        shape = (n, h_out, w_out, c, self.pool_size, self.pool_size)
        strides = (x.strides[0], self.stride * x.strides[1], self.stride * x.strides[2], x.strides[3], x.strides[1], x.strides[2])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        self.cache = windows
        self.x_shape = x.shape

        max_out = np.max(windows, axis=(4, 5))
        #print(f"MaxPool Output shape: {max_out.shape}")
        return max_out


    def backward(self, grad):
        windows = self.cache
        n, h_out, w_out, c, pool_h, pool_w = windows.shape
        
        # grad의 형상을 (n, h_out, w_out, c)로 변환
        grad = grad.reshape(n, h_out, w_out, c)
        
        grad_input = np.zeros(self.x_shape)  # 원래 입력 크기와 같은 배열 생성
        #print(f"backward MaxPooling input shape: {grad.shape}, window shape {windows.shape}, x shape: {self.x_shape}")

        # 각 윈도우 내 최대값 위치 마스크
        max_mask = (windows == np.max(windows, axis=(4, 5), keepdims=True))
        
        # grad를 pool window에 맞게 확장
        grad_expanded = grad[:, :, :, :, None, None]  # 마지막 두 차원을 확장
        grad_distributed = max_mask * grad_expanded  # grad를 각 윈도우에 분배

        # 스트라이드를 고려해 grad를 입력 크기에 분배
        for i in range(pool_h):
            for j in range(pool_w):
                grad_input[:, 
                        i * self.stride:i * self.stride + h_out, 
                        j * self.stride:j * self.stride + w_out, :] += grad_distributed[:, :, :, :, i, j]

        #print(f"backward MaxPooling output shape: {grad_input.shape}")

        return grad_input


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, x):
        #print(f"FullyConnected Input shape: {x.shape}")
        output = x @ self.weights + self.biases
        #print(f"FullyConnected Output shape: {output.shape}")
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
        #print(f"backward FullyConnected Input shape: {x.shape}")

        pool_size = self.pool_size
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                window = x[:, i * self.stride:i * self.stride + pool_size, j * self.stride:j * self.stride + pool_size, :]
                max_mask = (window == np.max(window, axis=(1, 2), keepdims=True))
                grad_input[:, i * self.stride:i * self.stride + pool_size, j * self.stride:j * self.stride + pool_size, :] += grad[:, i, j, :][:, None, None, :] * max_mask
        #print(f"backward FullyConnected Input shape: {grad_input.shape}")

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

    def backward(self, x, y, learning_rate, y_pred):
        # Forward pass to get predictions and intermediate cache

        # Compute gradient of loss w.r.t softmax input
        grad = y_pred - y  # Cross-entropy loss with softmax

        # Backpropagate through FullyConnected Layer 2
        #print(f"backward FullyConnected2 Input shape: {grad.shape}")
        grad, dw_fc2, db_fc2 = self.compute_fc_grad(self.cache["fc1"], self.fc2, grad)
        self.fc2.weights -= learning_rate * dw_fc2
        self.fc2.biases -= learning_rate * db_fc2

        # Backpropagate through FullyConnected Layer 1
        #print(f"backward FullyConnected1 Input shape: {grad.shape}")
        grad, dw_fc1, db_fc1 = self.compute_fc_grad(self.cache["flatten"], self.fc1, grad)
        self.fc1.weights -= learning_rate * dw_fc1
        self.fc1.biases -= learning_rate * db_fc1

        # Backpropagate through Pooling Layer 2
        grad = self.pool2.backward(grad)  # Only pass the gradient

        # Backpropagate through Convolution Layer 2
        grad = np.where(self.cache["relu2"] > 0, grad, 0)  # ReLU grad
        grad, dw_conv2, db_conv2 = self.conv2.backward(grad)
        self.conv2.kernels -= learning_rate * dw_conv2
        self.conv2.biases -= learning_rate * db_conv2

        # Backpropagate through Pooling Layer 1
        grad = self.pool1.backward(grad)  # Only pass the gradient

        # Backpropagate through Convolution Layer 1
        grad = np.where(self.cache["relu1"] > 0, grad, 0)  # ReLU grad
        grad, dw_conv1, db_conv1 = self.conv1.backward(grad)
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

                # Backpropagation
                self.backward(batch_x, batch_y, learning_rate, y_pred)
            print(f"Epoch {epoch + 1} , Loss: {loss:.4f}")

    def evaluate(self, x, y):
        y_pred = self.forward(x)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_classes == y_true_classes)
        return accuracy

    def save_model(self, filepath):
        model_data = {
            "conv1_kernels": self.conv1.kernels,
            "conv1_biases": self.conv1.biases,
            "conv2_kernels": self.conv2.kernels,
            "conv2_biases": self.conv2.biases,
            "fc1_weights": self.fc1.weights,
            "fc1_biases": self.fc1.biases,
            "fc2_weights": self.fc2.weights,
            "fc2_biases": self.fc2.biases
        }
        with open(filepath, 'wb') as f:
            np.save(f, model_data)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = np.load(f, allow_pickle=True).item()
        self.conv1.kernels = model_data["conv1_kernels"]
        self.conv1.biases = model_data["conv1_biases"]
        self.conv2.kernels = model_data["conv2_kernels"]
        self.conv2.biases = model_data["conv2_biases"]
        self.fc1.weights = model_data["fc1_weights"]
        self.fc1.biases = model_data["fc1_biases"]
        self.fc2.weights = model_data["fc2_weights"]
        self.fc2.biases = model_data["fc2_biases"]
        print(f"Model loaded from {filepath}")

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
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
