import numpy as np
import sys
from model import CNN
from data import load_data

if __name__ == "__main__":
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_data()

    # Load saved model data
    path = sys.path[0] + '/../ckpt/ckpt.pkl'
    with open(path, 'rb') as f:
        model_data = np.load(f, allow_pickle=True).item()

    # Define CNN structure
    conv_layers = [(3, 3, 1, 8), (3, 3, 8, 16)]  # 2 Conv layers: 1 input channel -> 8 filters -> 16 filters
    fc_layers = [400, 128, 10]  # Fully connected layers: 400 (flattened) -> 128 -> 10 classes
    model = CNN(conv_layers, fc_layers)

    # Load weights and biases into the model
    model.kernels = model_data["kernels"]
    model.weight = model_data["weights"]
    model.bias = model_data["biases"]

    # Evaluate model
    train_acc = model.evaluate(train_images, train_labels)
    test_acc = model.evaluate(test_images, test_labels)

    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
