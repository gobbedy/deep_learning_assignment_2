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
num_epochs = 20

# heuristically we pick a batch size with an adequate compromise between accurate gradient (larger sample size of
# datapoints is better) and speed of computation (smaller batch size is better)
batch_size = 100

# tradeoff: precision of
learning_rate = 0.1

# set of 60,000 28*28 images + 60,000 digit labels



# TODO: how are parameters initialized?

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
class Conv_Regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Conv_Regression, self).__init__()

        # crucial line:
        # calls register_parameters() (via __setattr__), which adds the linear model in the children modules list
        # the linear model registers its own parameters also via register_parameters()
        # later, LogisticRegression obj's self.parameters() will recursively find all its children modules' parameters
        # in our case, this will just be the parametesr of the linear layer

        # In short: this line is critical for keeping track of parameters for gradient descent

        # Note that the below forward function needs to call 'self.linear' and not its own instance of nn.Linear
        # as otherwise the model would lose track of parameters for gradient descent
        #self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

        ##self.linear1 = nn.Linear(in_features=input_size, out_features=50)
        ##self.linear2 = nn.Linear(in_features=50, out_features=num_classes)

        #self.conv1 = nn.Conv2d(1, 20, 5)
        #self.conv2 = nn.Conv2d(20, 20, 5)

        conv1_kernel_size = 5
        conv1_num_kernels = 6

        conv2_kernel_size = 5
        conv2_num_kernels = 16

        # input to conv1 is 2D image, so 1 channel, output has num_kernels channels ie 6 channels
        self.conv1 = nn.Conv2d(1, conv1_num_kernels, conv1_kernel_size)
        self.max_pooling1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(conv1_num_kernels, conv2_num_kernels, conv2_kernel_size)
        self.max_pooling2 = nn.MaxPool2d((2, 2))

        im_side = 28
        #c1_size = conv1_num_kernels * (im_side - conv1_kernel_size + 1) ** 2
        c1_side = im_side - conv1_kernel_size + 1 # 24
        p1_side = c1_side // 2 # 12
        c2_side = p1_side - conv2_kernel_size + 1 # 8
        p2_side = c2_side // 2 # 4
        #c2_size = conv2_num_kernels * (c1_side - conv2_kernel_size + 1) ** 2
        #c2_size = conv2_num_kernels * (p1_side - conv2_kernel_size + 1) ** 2 # 16 * 8^2 = 16 * 64 = 1024
        #print(c2_size)
        p2_size = conv2_num_kernels * p2_side ** 2  # 16 * 4^2 = 16^2 = 256

        linear1_output_len=120
        linear2_output_len=84
        linear3_output_len=num_classes

        self.linear1 = nn.Linear(in_features=p2_size, out_features=linear1_output_len)
        self.linear2 = nn.Linear(in_features=linear1_output_len, out_features=linear2_output_len)
        self.linear3 = nn.Linear(in_features=linear2_output_len, out_features=linear3_output_len)


    def forward(self, x):

        ##y1 = F.relu(self.linear1(x))
        ##out = F.relu(self.linear2(y1))



        c1 = self.conv1(x)
        p1 = F.relu(self.max_pooling1(c1))

        c2 = self.conv2(p1)
        p2 = F.relu(self.max_pooling2(c2))

        # p2_size should also work instead of -1
        p2 = p2.view(batch_size, -1)

        y1 = F.relu(self.linear1(p2))
        y2 = F.relu(self.linear2(y1))

        out = self.linear3(y2)


        return out


model = Conv_Regression(input_size, num_classes)

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
        #images = images.view(-1, 28 * 28)


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

        if (i + 1) % 100 == 0:

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
    if lr < 0.001:
        break

# Test the Model
correct = 0
total = 0


for i, (images, labels) in enumerate(test_loader):


    #images = images.view(-1, 28 * 28)

    # linear layer
    outputs = model(images)


    # note that we do not need softmax layer for the decision process, as softmax does not change the
    # order the selection (merely amplifies the difference between them) -- picking index corresponding to
    # max value is sufficient

    _, predicted = torch.max(outputs.data, 1)
    total += len(labels)
    correct += (predicted == labels).sum()


print('Accuracy of the model on the 10000 test images: %.3f %%' % (100 * correct.item() / total))