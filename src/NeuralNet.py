import torch
import numpy as np
from IPython import display  # Remove this line out of jupyter notebooks
import matplotlib.pyplot as plt


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()  # Necessary for torch to detect this class as trainable
        # Here define network architecture
        self.layer1 = torch.nn.Linear(28 ** 2, 4)  # Linear layer with 32 neurons
        self.layer2 = torch.nn.Linear(4, 4)  # Linear layer with 64 neurons
        self.output = torch.nn.Linear(4, 1)  # Linear layer with 1 neuron (binary output)

    def forward(self, x):
        # Here define architecture behavior
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return torch.sigmoid(self.output(x))  # Binary output


# Instantiate network
model = NeuralNet()

# Optimizaiton config
target_class = 3  # Train a classifier for this class
batch_size = 100  # Number of samples used to estimate the gradient (bigger = stable trai
learning_rate = 0.05  # Optimizer learning rate
epochs = 100  # Number of iterations over the whole dataset.

# Create optimizer for the network parameters
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
# Instantiate loss function
criterion = torch.nn.BCELoss()  # Binary logistic regression

# Prepare data
train_data = np.load('../db/train.npy')
val_data = np.load('../db/val.npy')
test_data = np.load('../db/test_alumnes.npy')


def select_class(data, clss, train=False):
    images = np.array(data.item()["images"])
    labels = np.array(data.item()["labels"])
    indices = np.arange(labels.shape[0])

    if train:
        np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]
    labels = (labels == target_class).astype(int)
    return images, labels


train_images, train_labels = select_class(train_data, target_class, train=True)
val_images, val_labels = select_class(val_data, target_class, train=False)

train_size = train_labels.shape[0]
val_size = val_labels.shape[0]


# Function to iterate the training set and update network weights with batches of images
def train(model, optimizer, criterion):
    model.train()  # training mode
    running_loss = 0
    running_corrects = 0
    total = 0
    for idx in range(0, train_size, batch_size):
        optimizer.zero_grad()  # make the gradients 0
        x = torch.from_numpy(train_images[idx:(idx + batch_size), ...]).float()
        y = torch.from_numpy(train_labels[idx:(idx + batch_size), ...]).float()
        output = model(x.view(-1, 28 ** 2))  # forward pass
        preds = (output > 0.5).float()
        loss = criterion(output.view_as(y), y)  # calculate the loss value
        loss.backward()  # compute the gradients
        optimizer.step()  # uptade network parameters
        # statistics
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y).item()  # .item() converts type from torc
        total += float(y.size(0))
    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy
    return epoch_loss, epoch_acc


# Function to iterate the validation set and update network weights with batches of imag
def val(model, criterion):
    model.eval()  # validation mode
    running_loss = 0
    running_corrects = 0
    total = 0
    with torch.no_grad():  # We are not backpropagating trhough the validation set, so we
        for idx in range(0, val_size, batch_size):
            x = torch.from_numpy(val_images[idx:(idx + batch_size), ...]).float()
            y = torch.from_numpy(val_labels[idx:(idx + batch_size), ...]).float()
            output = model(x.view(-1, 28 ** 2))  # forward pass
            preds = (output > 0.5).float()
            loss = criterion(output.view_as(y), y)  # calculate the loss value
            # statistics
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds == y).item()  # .item() converts type from
            total += float(y.size(0))
    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy
    return epoch_loss, epoch_acc


# Main training loop
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

for epoch in range(epochs):
    t_loss, t_acc = train(model, optimizer, criterion)
    v_loss, v_acc = val(model, criterion)
    train_loss.append(t_loss)
    train_accuracy.append(t_acc)
    val_loss.append(v_loss)
    val_accuracy.append(v_acc)
    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.plot(train_loss, 'b-')
    plt.plot(val_loss, 'r-')
    plt.legend(["train", "val"])
    plt.subplot(1, 2, 2)
    plt.title("accuracy")
    plt.plot(train_accuracy, 'b-')
    plt.plot(val_accuracy, 'r-')
    plt.legend(["train", "val"])
    display.clear_output(wait=True)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.show(
