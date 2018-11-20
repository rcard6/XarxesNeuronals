import torch  # Import main library
from torch.utils.data import DataLoader  # Main class for threaded data loading
import matplotlib.pyplot as plt
import numpy as np

# Optimizaiton config
target_class = 3  # Train a classifier for this class
batch_size = 100  # Number of samples used to estimate the gradient (bigger = stable trai
learning_rate = 0.05  # Optimizer learning rate
epochs = 100  # Number of iterations over the whole dataset.
# Prepare data
train_data = np.load('../db/train.npy')
val_data = np.load('../db/val.npy')


def select_class(data, clss):
    images = np.array(data.item()["images"])
    labels = np.array(data.item()["labels"])
    labels = (labels == target_class).astype(int)
    return images, labels


train_images, train_labels = select_class(train_data, target_class)
val_images, val_labels = select_class(val_data, target_class)
train_size = train_labels.shape[0]
val_size = val_labels.shape[0]
print(train_size, "training images.")


indices = np.arange(train_size)
positive_indices = indices[train_labels == 1]
negative_indices = indices[train_labels == 0]
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Positive")
plt.imshow(train_images[positive_indices[0], :, :], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Negative")
plt.imshow(train_images[negative_indices[0], :, :], cmap="gray")
plt.show()
