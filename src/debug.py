import numpy as np
import sys
from model import CNN
from data import load_data


# Load and preprocess data
train_images, train_labels, test_images, test_labels = load_data()

model = CNN()

model.train(train_images, train_labels, epochs=10, batch_size=64, learning_rate=0.01)

# Evaluate model
save_path = sys.path[0] + "/../ckpt/ckpt.pkl"
with open(save_path, 'wb') as f:
    np.save(f, {"kernels": model.kernels, "weights": model.weight, "biases": model.bias})
print(f"Model saved to {save_path}")

train_acc = model.evaluate(train_images, train_labels)
test_acc = model.evaluate(test_images, test_labels)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")