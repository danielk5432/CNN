import numpy as np
import sys
from model import *
from data import load_data

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data()

    layers = [784, 128, 10]  # input 784, one hidden layer 128, output 10
    model = NN(layers)

    model.train(train_images, train_labels, epochs=50, b_size=64, learning_rate=0.1)

    save_path = sys.path[0] + "/../ckpt/ckpt.pkl" 
    with open(save_path, 'wb') as f:
        np.save(f, {"weights": model.weight, "biases": model.bias})
    print(f"Model saved to {save_path}")
