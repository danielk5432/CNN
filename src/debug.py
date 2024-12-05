import numpy as np
import sys
from model import CNN
from data import load_data


# Load and preprocess data
train_images, train_labels, test_images, test_labels = load_data()

# Define CNN structure
#conv_layers = [(3, 3, 1, 8), (3, 3, 8, 16)]  # 2 Conv layers: 1 input channel -> 8 filters -> 16 filters
#fc_layers = [400, 128, 10]  # Fully connected layers: 400 (flattened) -> 128 -> 10 classes
#model = CNN(conv_layers, fc_layers)

test_model = CNN(conv_layers=[], fc_layers=[784, 128, 10])

# Train the model
test_model.train(train_images, train_labels, epochs=50, batch_size=64, learning_rate=0.1)

# Evaluate model
train_acc = test_model.evaluate(train_images, train_labels)
test_acc = test_model.evaluate(test_images, test_labels)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")