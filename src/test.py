import numpy as np
import sys
from model import *
from data import load_data

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data()

    path = sys.path[0] + '/../ckpt/ckpt.pkl'
    with open(path, 'rb') as f:
        model_data = np.load(f, allow_pickle=True).item()

    layers = [784, 128, 10]
    model = NN(layers)
    model.weights = model_data["weights"]
    model.biases = model_data["biases"]

    train_acc = model.evaluate(train_images, train_labels)
    test_acc = model.evaluate(test_images, test_labels)

    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
