import numpy as np
import sys

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        f.read(16)  # 16 byte header
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(-1, 28 * 28)  # unpack 28*28 image
        return images / 255.0  # Normalize to [0, 1]

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        f.read(8)  # 8 byte header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def load_data():
    train_images = load_mnist_images(sys.path[0] + '/../dataset/train-images.idx3-ubyte')
    train_labels = load_mnist_labels(sys.path[0] + '/../dataset/train-labels.idx1-ubyte')
    test_images = load_mnist_images(sys.path[0] + '/../dataset/t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels(sys.path[0] + '/../dataset/t10k-labels.idx1-ubyte')
    
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    return train_images, train_labels, test_images, test_labels
