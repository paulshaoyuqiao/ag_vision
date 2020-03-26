import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from resnet_utils import *
from functools import partial
from apex import amp

# defined architectures
NETS = ['simple', 'vgg19', 'res18']

# define the simple CNN architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # convolutional layer 1
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        # convolutional layer 2
        self.conv2 = nn.Conv2d(4, 32, 3, padding=1)
        # convolutional layer 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer (2x2 kernel)
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 -> 64)
        self.fc1 = nn.Linear(64 * 4, 64 * 2)
        # linear layer (64 -> 3)
        self.fc2 = nn.Linear(64 * 2, 3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add 2nd hidden layer, with relu activation function
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class VGG19Net(nn.Module):
    def __init__(self, features, num_classes=3, init_weights=True):
        super(VGG19, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    cfg = {
        'A': [4, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3 #Define the number of channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def vgg19(pretrained=False, **kwargs):
        """VGG 19-layer model (configuration "E")
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            kwargs['init_weights'] = False
        model = VGG19Net(make_layers(cfg['E']), **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
        return model


def construct_net(name, device, in_channels=3, n_classes=3):
    if name not in NETS:
        print('Error: {} not found in lists of available architectures, please choose from {}'.format(name, NETS))
    if not device:
        print('Error: Must provide an actual device (CPU or GPU) to train the model on')
    if name.lower().startswith('simple'):
        model = SimpleNet()
    elif name.lower().startswith('vgg19'):
        model = VGG19Net()
    else:
        model = resnet18(in_channels, n_classes)
    model = model.float()
    model.to(device)
    print(model)
    return model

def train_and_validate(
    model, optimizer, device, criterion, train_loader, valid_loader, model_path, n_epochs=32
):
    # number of epochs to train the model
    train_losses, validate_losses = [], []
    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        print('Training Currently In Epoch {}...'.format(epoch))
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        
        train_losses.append(train_loss)
        validate_losses.append(valid_loss)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model to path {} ...'.format(valid_loss_min,valid_loss, model_path))
            torch.save(model.state_dict(), model_path)
            valid_loss_min = valid_loss
        
    return train_losses, validate_losses
