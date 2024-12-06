import numpy as np
import pickle

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    print(input_data.shape, filter_h, filter_w, stride, pad)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Invalid output dimensions: out_h={out_h}, out_w={out_w}. Check input size, filter size, stride, and padding."
        )

    # Padding
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    # Create sliding windows
    shape = (N, C, filter_h, filter_w, out_h, out_w)
    strides = (
        img.strides[0], 
        img.strides[1], 
        img.strides[2], 
        img.strides[3], 
        img.strides[2] * stride, 
        img.strides[3] * stride
    )
    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    col = col.reshape(N * out_h * out_w, -1)

    return col




def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class ConvLayer:
    def __init__(self, filters, kernel_size, stride=1, pad=0, initializer='he', reg=0):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.pad = pad
        self.reg = reg
        self.param = None  # Weights will be initialized during forward pass
        self.bias = None
        self.col = None
        self.col_param = None

    def forward(self, x):
        n, c, h, w = x.shape
        if self.param is None:
            # Initialize weights and biases based on input channel size
            self.param = np.random.randn(self.filters, c, *self.kernel_size) * np.sqrt(2. / (c * self.kernel_size[0] * self.kernel_size[1]))
            self.bias = np.zeros(self.filters)
        
        fn, fc, fh, fw = self.param.shape
        out_h = int(1 + (h + 2 * self.pad - fh) / self.stride)
        out_w = int(1 + (w + 2 * self.pad - fw) / self.stride)

        col = im2col(x, fh, fw, self.stride, self.pad)
        col_param = self.param.reshape(fn, -1).T

        self.col = col
        self.col_param = col_param
        out = np.dot(col, col_param) + self.bias
        out = out.reshape(n, out_h, out_w, fn).transpose(0, 3, 1, 2)
        return out


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1

        col = im2col(x, ph, pw, self.stride, 0)
        col = col.reshape(-1, ph * pw)
        out = np.max(col, axis=1)
        self.arg_max = np.argmax(col, axis=1)

        out = out.reshape(n, c, out_h, out_w)
        return out

    def backward(self, dout):
        n, c, h, w = dout.shape
        ph, pw = self.pool_size
        dmax = np.zeros_like(self.arg_max)
        flat_dout = dout.transpose(0, 2, 3, 1).flatten()
        dmax[self.arg_max] = flat_dout

        dmax = dmax.reshape(-1, ph * pw)
        dx = col2im(dmax, self.input_shape, ph, pw, self.stride, 0)
        return dx

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, reg=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)
        self.reg = reg

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        self.grad_weights = np.dot(self.x.T, dout) + self.reg * self.weights
        self.grad_bias = np.sum(dout, axis=0)
        return np.dot(dout, self.weights.T)

class SimpleCNN:
    def __init__(self):
        self.conv1 = ConvLayer(8, 3, stride=1, pad=1)
        self.pool1 = MaxPoolingLayer(pool_size=(2, 2), stride=2)
        self.conv2 = ConvLayer(16, 3, stride=1, pad=1)
        self.pool2 = MaxPoolingLayer(pool_size=(2, 2), stride=2)
        self.fc = FullyConnectedLayer(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc.forward(x)
        return softmax(x)

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = dout.reshape(-1, 16, 7, 7)
        dout = self.pool2.backward(dout)
        dout = relu_derivative(self.conv2.forward(self.conv1.x)) * dout
        dout = self.pool1.backward(dout)
        dout = relu_derivative(self.conv1.forward(self.conv1.x)) * dout
        return dout

    def train(self, x, y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                logits = self.forward(x_batch)
                loss_grad = logits - y_batch
                self.backward(loss_grad)

                self.fc.weights -= learning_rate * self.fc.grad_weights
                self.fc.bias -= learning_rate * self.fc.grad_bias

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
