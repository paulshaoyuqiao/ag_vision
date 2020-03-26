# Import All Necessary Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from apex import amp
from net_utils import *
from data_utils import *
from predict_utils import *

# define the leaves classes
# Type 0 and 1: Known Types
# Type 2: Other Type (Unknown: Including Other Types, Background, and Leaf Edges)
CLASSES=['0', '1', '2', '3']
# define the train/validation data directory
DATA_DIR = 'data-16'
MODEL_DIR = './models'
LOSS_DIR = './losses'
TARGET_WIDTH = 16
ARCHITECTURE = 'simple'
TEST_DIR = './test-data/'
LOSS_FILE = '{}_16x16_01_losses.csv'
LOSS_PLOT = '{}_16x16_01_losses.png'
MODEL_FILE = '{}_16x16_01.pt'.format(ARCHITECTURE)
LR = 0.005
MOMENTUM = 0.8

# Generate Data (DISABLED)
# for class_dir in CLASSES:
#     origin_dir = './{}'.format(class_dir)
#     generate_data_in_dir(origin_dir, '.png', class_dir, DATA_DIR, TARGET_WIDTH)

# Generate Data (for Old Dataset -- Other Type OR new bok choy Type - 3) (DISABLED)
# generate_data_in_dir('./3', '.png', '3', DATA_DIR, 16)

# Enable Running on GPU if Possible
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# initiate data loaders (train loader and validate loaders)
train_loader, validate_loader = generate_samplers('./{}'.format(DATA_DIR))
print('Finished Generating Train Data Loader and Validation Data Loader!')

# specify optimizer, model, criterion, and path to save best model
model = construct_net(ARCHITECTURE, device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.NLLLoss()
model_path = '{}/{}'.format(MODEL_DIR, MODEL_FILE)
print('Finished Initializing Model, Optimizer, Criterion, Loss Function, and Model Path!')

# use NVIDIA's mixed precision training module to speed up training
model, optimizer = amp.initialize(model, optimizer)
print('Finished optimizing Model and Optimizer to use Nvidia\'s distributed Training!')

print('----------------------------------------------------')
print('Begin Training...')
# begin training
train_losses, validate_losses = train_and_validate(
    model=model, optimizer=optimizer, device=device, criterion=criterion, 
    train_loader=train_loader, valid_loader=validate_loader, model_path=model_path
)
print('Finished Traning...')

# assert that training actually happens
assert len(train_losses) > 0 and len(validate_losses) > 0

# export and save losses to .csv file for later analysis
losses = pd.DataFrame(
    {
        'train': train_losses,
        'validate': validate_losses
    }
)
loss_file = LOSS_FILE.format(ARCHITECTURE)
losses.to_csv(path_or_buf='{}/{}'.format(LOSS_DIR, loss_file))
print('Generated CSV loss files at path {}'.format(loss_file))

# read from csv file path and generated loss graph
losses = pd.read_csv('./losses/' + loss_file)
plt.plot(losses['train'], label='train')
plt.plot(losses['validate'], label='validate')
plt.legend()
plt.savefig(LOSS_PLOT.format(ARCHITECTURE))

# testing model accuracy on training set
train_loader, validate_loader = generate_samplers('./{}'.format(DATA_DIR))
print('Finished Regenerating Train Data Loader and Validation Data Loader! Testing Classification Accuracy Now...')
test_classification_accuracy(device, model, model_path, train_loader)

test_path = './test-data/test_1.JPG'

mask = recover_shape_alt(device, test_path, model, model_path)
plt.imshow(mask)
plt.show()
print('Mask Recovery Finished!')
