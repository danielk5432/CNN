import numpy as np
import sys
from model import CNN
from data import load_data

if __name__ == "__main__":
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_data()

    # Define CNN structure
    conv_layers = [(3, 3, 1, 8), (3, 3, 8, 16)]  # 2 Conv layers: 1 input channel -> 8 filters -> 16 filters
    fc_layers = [400, 128, 10]  # Fully connected layers: 400 (flattened) -> 128 -> 10 classes
    model = CNN(conv_layers, fc_layers)

    # Train the model
    model.train(train_images, train_labels, epochs=50, batch_size=64, learning_rate=0.1)

    # Save model parameters
    save_path = sys.path[0] + "/../ckpt/ckpt.pkl"
    with open(save_path, 'wb') as f:
        np.save(f, {"kernels": model.kernels, "weights": model.weight, "biases": model.bias})
    print(f"Model saved to {save_path}")
