

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

import model_functions
import json
import argparse
parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments

parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Architecture')
parser.add_argument('--epochs', type = int, default = 5, help = 'Epochs')

# define the main function to include the global code logic
def main():

    arguments = parser.parse_args()
    device = torch.device("cuda:0")


    #checking the availibility of GPU
    value = torch.cuda.is_available()
    if value == True:
          print('The device',torch.cuda.get_device_name(0),"is available.")

    # Build and train the neural network (Transfer Learning)
    if arguments.arch == 'vgg16':
        input_size = 25088
        output_size = 4096
        model = models.vgg16(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        # Build custom classifier
        model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, output_size)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(output_size, 2)),
                                            ('output', nn.LogSoftmax(dim=1))]))
        param = model.classifier

    elif arguments.arch == 'alexnet':
        input_size = 9216
        output_size = 4096
        model = models.alexnet(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        # Build custom classifier
        model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, output_size)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(output_size, 2)),
                                            ('output', nn.LogSoftmax(dim=1))]))
        param = model.classifier

    else:
        model = models.resnet50(pretrained=True)
        # Parameters of newly constructed modules have requires_grad=True by default
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features , 2)
        param = model.fc


    print("The Model:",model)

    #Assigning the model to the device (GPU)
    model = model.to(device)

    #Optimizer
    optimizer = optim.SGD(param.parameters(), lr=0.001, momentum=0.9)

    #Loss
    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #Training the model
    print('Training the model')
    model = model_functions.train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs= arguments.epochs)

    #testing the model on test dataset
    print("Testing on the model")
    model_functions.test_accuracy(model)

    #solve the confusion matrix
    cm = model_functions.cal_confusion_matrix(model)

    #plot confusion matrix
    model_functions.plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

    #Saving the model
    model_functions.save_checkpoint(model,arguments.arch, arguments.epochs, 0.001)
    print("The model is saved!!")


if __name__ == '__main__':

    main()
