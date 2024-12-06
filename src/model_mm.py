import numpy as np

# im2col and col2im function are used for improved computational efficiency avoiding nested for loops
def im2col(input, filter_size, stride=1): # Convert input into a column matrix for efficient matrix multiplication
    batch_size, channels, height, width = input.shape
    f = filter_size
    out_height = (height - f) // stride + 1
    out_width = (width - f) // stride + 1
    print(batch_size, channels, f, f, out_height, out_width)
    cols = np.zeros((batch_size, channels, f, f, out_height, out_width)) # Create an empty array for columns

    for i in range(f): # Extract patches using slicing
        for j in range(f):
            cols[:, :, i, j, :, :] = input[:, :, i:i + stride * out_height:stride, j:j + stride * out_width:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1) # Reshape and reorder the patches for matrix multiplication
    return cols, out_height, out_width # Return a reshaped matrix


def col2im(cols, input_shape, filter_size, out_height, out_width, stride=1): # Converts column matrix back into original image format
    batch_size, channels, height, width = input_shape
    f = filter_size

    cols = cols.reshape(batch_size, out_height, out_width, channels, f, f).transpose(0, 3, 4, 5, 1, 2) # Reshape cols back into the original image shape

    input_gradient = np.zeros((batch_size, channels, height, width)) # Initialize gradient matrix

    for i in range(f): # Reassemble the patches back into the input tensor.
        for j in range(f):
            input_gradient[:, :, i:i + stride * out_height:stride, j:j + stride * out_width:stride] += cols[:, :, i, j, :, :]

    return input_gradient


class Conv2D: # Conv2D class implements convolutional layer
    def __init__(self, num_filters, filter_size, in_channels=1, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.stride = stride
        self.filters = np.random.randn(num_filters, in_channels, filter_size, filter_size) * 0.1


    def forward(self, input): # Perform forward pass using im2col for efficient convolution
        self.last_input = input
        self.cols, self.out_height, self.out_width = im2col(input, self.filter_size, self.stride)
        reshaped_filters = self.filters.reshape(self.num_filters, -1)
        out = self.cols @ reshaped_filters.T # Matrix multiplication with reshaped input and filters
        out = out.reshape(input.shape[0], self.out_height, self.out_width, self.num_filters) # Reshape the result back into the tensor format
        return out.transpose(0, 3, 1, 2) # Return the reshaped output

    def backward(self, d_out, learning_rate):
        d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)
        d_filters = d_out_flat.T @ self.cols # Compute gradients for the filters.
        d_filters = d_filters.reshape(self.filters.shape)

        reshaped_filters = self.filters.reshape(self.num_filters, -1)
        d_cols = d_out_flat @ reshaped_filters # Compute the gradient for the input tensor.
        d_input = col2im(d_cols, self.last_input.shape, self.filter_size, self.out_height, self.out_width, self.stride)

        self.filters -= learning_rate * d_filters # Update filters using the learning rate
        return d_input


class ReLU: # ReLU activation function. ReLU prevents gradient vanishing and introduces non linearity.
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_out): # Propagate the gradient only for positive input values
        return d_out * (self.last_input > 0)


class MaxPooling: # Max pooling reduces computational complexity while retaining important features
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        self.input = input

        self.out_height = (height - self.pool_size) // self.stride + 1
        self.out_width = (width - self.pool_size) // self.stride + 1

        cols, _, _ = im2col(input, self.pool_size, self.stride) # Reshape input into columns using im2col function
        cols = cols.reshape(batch_size, channels, self.pool_size * self.pool_size, self.out_height, self.out_width) # Reshape cols to apply max pooling

        self.max_idx = np.argmax(cols, axis=2)  # Indices of max values for backward pass
        out = np.max(cols, axis=2) # Compute the max-pooled output.
        return out

    def backward(self, d_out):
        batch_size, channels, height, width = self.input.shape

        d_cols = np.zeros((batch_size, channels, self.pool_size * self.pool_size, self.out_height, self.out_width)) # Create gradient template

        flat_idx = np.arange(self.max_idx.size) # Place gradients into the position of maximum values
        d_cols.reshape(-1)[flat_idx * self.pool_size * self.pool_size + self.max_idx.ravel()] = d_out.ravel()

        d_cols = d_cols.reshape(batch_size, channels, self.pool_size, self.pool_size, self.out_height, self.out_width)
        d_cols = d_cols.transpose(0, 1, 4, 5, 2, 3).reshape(batch_size, channels, -1) # Reshape and reassemble the gradient back into the original input shape.

        return col2im(d_cols, self.input.shape, self.pool_size, self.out_height, self.out_width, self.stride) # Return the output reshaped back to image



class Dense: # Fully connected layer
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input): # Compute the linear transformation y = Wx + b
        self.last_input_shape = input.shape
        self.last_input = input.reshape(input.shape[0], -1)
        self.last_output = self.last_input @ self.weights + self.biases
        return self.last_output

    def backward(self, d_out, learning_rate): # Compute the gradients for the weights, biases, and inputs
        d_weights = self.last_input.T @ d_out
        d_biases = np.sum(d_out, axis=0)
        d_input = d_out @ self.weights.T

        self.weights -= learning_rate * d_weights # Update weights using learning rate
        self.biases -= learning_rate * d_biases # Update biases using learning rate
        return d_input.reshape(self.last_input_shape)


class Softmax: # Softmax activation function for multi class classification
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.last_output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.last_output # Return a probability distribution

    def backward(self, d_out):
        return d_out


