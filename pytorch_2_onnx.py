from torch.autograd import Variable
import torch.onnx
import torchvision
from torchvision import models
import torch 
import torch.nn as nn
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser(description='Convert to onnx')

# Command line arguments

parser.add_argument('--path', type = str, default = 'Model_weights_checkpoint/checkpoint_vgg16.pth', help = 'Path to model weights')

arguments = parser.parse_args()

#change the path to the corresponding model_weights
input1 = arguments.path

device = torch.device("cuda:0")

# load the model checkpoint
checkpoint = torch.load(input1)
arch = checkpoint['model_name']


# create the model architecture
print('using model:  ' + arch)
if arch == 'vgg16':
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
        
elif arch == 'alexnet':
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
           

# load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

# add softmax layer
model = torch.nn.Sequential(model, torch.nn.Softmax(1))

model.to(device)
model.eval()

#print(model)
reso = checkpoint['input_size']
input = torch.ones((1, 3, reso, reso)).cuda()
output = arch + '.onnx'
# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, output, verbose=True, input_names=input_names, output_names=output_names)