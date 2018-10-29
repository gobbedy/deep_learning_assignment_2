#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_rescheduler


# Hyper Parameters

# input to the linear layer is total number of pixels, represented in grayscale as a number from 0-255
input_size = 28*28

# number of labels -- here labels are 0-9
num_classes = 10

# epoch = forward and backward pass over all training examples (more exactly: over the same number of samples
# as the training dataset since with SGD we may pick random samples and technically not use all training examples)
num_epochs = 50

# heuristically we pick a batch size with an adequate compromise between accurate gradient (larger sample size of
# datapoints is better) and speed of computation (smaller batch size is better)
batch_size = 100

# tradeoff: precision of
learning_rate = 0.1

# set of 60,000 28*28 images + 60,000 digit labels



try:

    # note: transforms.ToTensor() converts PIL format to tensor, and also
    # normalizes grayscale values in [0-255] range to [0-1]
    # this normalization allows faster learning as signmoid function more likely to be in roughly linear range
    # TODO: ask prof Mao that this is correct interpretation
    train_dataset = dsets.MNIST(root='.',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)

except RuntimeError:
    train_dataset = dsets.MNIST(root='.',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# set of 10,000 28*28 images + 10,000 digit labels
test_dataset = dsets.MNIST(root='.',
                           train=False,
                           transform=transforms.ToTensor())



# dataset loader: generator that yields input samples each based on rules: yields "batch_size" samples
# on each call and yields random samples or works its way sequentially thru training dataset depending on "shuffle"
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# batch size is irrelevant for test_loader, could set it to full test dataset
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()

        # crucial line:
        # calls register_parameters() (via __setattr__), which adds the linear model in the children modules list
        # the linear model registers its own parameters also via register_parameters()
        # later, LogisticRegression obj's self.parameters() will recursively find all its children modules' parameters
        # in our case, this will just be the parametesr of the linear layer

        # In short: this line is critical for keeping track of parameters for gradient descent

        # Note that the below forward function needs to call 'self.linear' and not its own instance of nn.Linear
        # as otherwise the model would lose track of parameters for gradient descent
        self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)

# softmax + cross entropy loss
criterion = nn.CrossEntropyLoss()

# Note that technically this is not just Gradient Descent, not Stochastic Gradient Descent
# What makes it 'stochastic' or 'not stochastic' is whether we use batch_size == dataset size or not
# In reality, SGD is just a generalized form of GD (where batch size = dateset size), so it is not actually
# incorrect to call it SGD when doing GD
# model.parameters() in this case is the linear layer's parameters, aka the 'theta' of our algorithm
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# adaptive learning rate policy -- "schedules" when to decrease learning rate
scheduler = lr_rescheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)


epoch=0

# Training the Model
while True:

    # train_loader keeps yielding batches until it reaches full epoch
    for i, (images, labels) in enumerate(train_loader):

        # 100 images x 784
        images = images.view(-1, 28 * 28)

        # 100 labels

        # prevent gradients from accumulating -- they should be computed "fresh" during each batch
        optimizer.zero_grad()

        # linear layer
        outputs = model(images)

        # softmax "layer" + cross entropy loss
        loss = criterion(outputs, labels)

        # computed loss gradients wrt to parameters
        loss.backward()

        # update parameters (here linear layer parameters) using learning rate + gradients
        optimizer.step()

        if (i + 1) % 600 == 0:

            lr=optimizer.param_groups[0]['lr']
            best_loss = scheduler.best
            print('Best end-of-epoch loss: %.7f, LR: %.4f, Epoch: [%d/%d], Step: [%d/%d], Loss: %.7f'
                  % (best_loss, lr, epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data.item()))


    epoch += 1

    # adapt learning rate
    scheduler.step(loss)
    lr = optimizer.param_groups[0]['lr']

    # stop when loss has stopped decreasing for a long time -- ie when, on 3 separate occasions, it didn't decrease
    # from start-to-end-of-epoch 3 epochs in a row
    # NOTE TO SELF: BETTER CONDITION WOULD BE CHECKING GRADIENT OF LOSS ITSELF
    if lr < 0.001:
        break


# Test the Model
correct = 0
total = 0

for i, (images, labels) in enumerate(test_loader):


    images = images.view(-1, 28 * 28)

    # linear layer
    outputs = model(images)


    # note that we do not need softmax layer for the decision process, as softmax does not change the
    # order the selection (merely amplifies the difference between them) -- picking index corresponding to
    # max value is sufficient

    _, predicted = torch.max(outputs.data, 1)
    total += len(labels)
    correct += (predicted == labels).sum()


print('Accuracy of the model on the 10000 test images: %.3f %%' % (100 * correct.item() / total))