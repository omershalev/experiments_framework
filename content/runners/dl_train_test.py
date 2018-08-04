from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from experiments_framework.content.runners import dl_dataset

# Hyper parameters
num_epochs = 1500
batch_size = 600
learning_rate = 0.00001

# MNIST constants
input_size = 784
num_classes = 10

# Override hyper parameters with the ones from the tutorial - remove it!
num_epochs = 10
batch_size = 100
learning_rate = 0.001


train_dataset = dl_dataset.SyntheticScanDataset(r'/home/omer/Downloads/temp')

train_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle=True)


# Model definition
class ScanClassifier(nn.Module):
    def __init__(self, input_size=360):
        super(ScanClassifier, self).__init__()
        self.fnn = nn.Sequential(
                    nn.Linear(input_size, 72),
                    nn.ReLU(),
                    nn.Linear(72, 58),
                    nn.ReLU(),
                    nn.Linear(58, 39),
                    nn.ReLU(),
                    nn.Linear(39, 22),
                    nn.ReLU(),
                    nn.Linear(22, 2))

    def forward(self, x):
        out = self.fnn(x)
        return out

model = ScanClassifier()


# Loss and optimizer definition
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Error and loss vectors definition
train_error = np.zeros((num_epochs))
train_loss = np.zeros((num_epochs))
test_error = np.zeros((num_epochs))
test_loss = np.zeros((num_epochs))


# Training
for epoch in range(num_epochs):

    # Iterate over test data for loss/error calculation
    correct = 0
    total = 10000
    for i, sample_batched in enumerate(train_loader):
        print (i)
        scans = sample_batched['scan']
        labels = sample_batched['location']
        images_var = Variable(scans)
        labels_var = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images_var)
        loss = criterion(outputs, labels_var)
        loss.backward()
        optimizer.step()

    print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))