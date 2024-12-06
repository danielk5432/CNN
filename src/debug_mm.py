import numpy as np
import os
import pickle
from model_mm import Conv2D, ReLU, MaxPooling, Dense, Softmax
from data import load_data

def train():
    dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset')  # Dataset path

    train_images, train_labels, test_images, test_labels = load_data()  # Load the MNIST dataset from binary files
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    train_images = train_images[:, np.newaxis, :, :]  # Add channel dimension to input data

    train_labels_one_hot = np.eye(10)[train_labels] # One-hot encode labels

    # Model definitions
    conv1 = Conv2D(num_filters=8, filter_size=3, in_channels=1)  # Conv layer 1
    relu1 = ReLU()  # ReLU activation
    pool1 = MaxPooling(pool_size=2, stride=2)  # Max pooling layer 1

    conv2 = Conv2D(num_filters=16, filter_size=3, in_channels=8)  # Conv layer 2
    relu2 = ReLU()  # ReLU activation
    pool2 = MaxPooling(pool_size=2, stride=2)  # Max pooling layer 2

    dense = Dense(input_size=16 * 5 * 5, output_size=10)  # Fully connected layer
    softmax = Softmax()  # Softmax for classification

    # Training parameters
    epochs = 10  # Number of epochs
    batch_size = 64  # Batch size
    learning_rate = 0.03  # Learning rate

    for epoch in range(epochs):  # Training loop
        epoch_loss = 0

        # Shuffle the data
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        train_images = train_images[indices]
        train_labels_one_hot = train_labels_one_hot[indices]

        # Process data in batches
        for i in range(0, len(train_images), batch_size):
            images = train_images[i:i + batch_size]
            labels = train_labels_one_hot[i:i + batch_size]

            # Forward pass
            out = conv1.forward(images)
            out = relu1.forward(out)
            out = pool1.forward(out)

            out = conv2.forward(out)
            out = relu2.forward(out)
            out = pool2.forward(out)

            out = out.reshape(out.shape[0], -1)
            predictions = dense.forward(out)
            probs = softmax.forward(predictions)

            # Compute loss (softmax cross-entropy)
            loss = -np.mean(np.sum(labels * np.log(probs + 1e-8), axis=1))
            epoch_loss += loss

            # Backward pass
            d_preds = (probs - labels) / labels.shape[0]
            d_out = softmax.backward(d_preds)
            d_out = dense.backward(d_out, learning_rate)

            d_out = d_out.reshape(-1, 16, 5, 5)
            d_out = pool2.backward(d_out)
            d_out = relu2.backward(d_out)
            d_out = conv2.backward(d_out, learning_rate)

            d_out = pool1.backward(d_out)
            d_out = relu1.backward(d_out)
            conv1.backward(d_out, learning_rate)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_images):.4f}")

    # Save trained model
    checkpoint = {
        "conv1_filters": conv1.filters,
        "conv2_filters": conv2.filters,
        "dense_weights": dense.weights,
        "dense_biases": dense.biases
    }
    with open("../ckpt/ckpt.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    print("Model saved to ckpt/ckpt.pkl")

train()
