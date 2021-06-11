from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode



# Data augmentation and normalization for training
# Just normalization for validation
def data_transforms(x):
    if x == 'train':
        ret_transforms= transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
    elif x == 'val':    
        ret_transforms= transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else: 
        ret_transforms= transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
             
    return ret_transforms
# Image data directories

data_dir = 'dataset'

# Load the datasets with ImageFolder

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms(x))
                  
# Using the image datasets and the trainforms, define the dataloaders
                  
              for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
               
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

global device 
device = torch.device("cuda:0")

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Function for measuring network accuracy on test data
def test_accuracy(model):

    # Do validation on the test set
    model.eval()

    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in dataloaders['test']:
    
            images, labels = images.to(device), labels.to(device)
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(dataloaders['test'])))  

# Function for saving the model checkpoint
training_dataset = image_datasets['train']

def save_checkpoint(model, arch, epochs, lr):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'input_size': 224,
                  'output_size': 2,
                  'batch_size': 4,
                  'learning_rate': lr,
                  'model_name': arch,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                 }


    save_to_path1 = 'Model_weights_checkpoint/'

    
    if os.path.isdir(save_to_path1):
        print('the path exits')
    else:
        print("the path doesn't exists, so creating one!")
        os.mkdir(save_to_path1) 
    torch.save(checkpoint,save_to_path1+'checkpoint_'+arch+'.pth')
