## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # [(W−K+2P)/S]+1.

        # W is the input volume
        # K is the Kernel size 
        # P is the padding 
        # S is the stride 
        self.conv1 = nn.Conv2d(1, 32, 5) # 224 -5/1 +1 = 220
        
        self.maxpool = nn.MaxPool2d(2,2) # 220/2 = 110
        
        
        self.conv2 = nn.Conv2d(32,64, 3) # 110 -3/1 +1 = 108
        
        self.maxpool2 = nn.MaxPool2d(2,2) # 108/2 = 54
        
        
        self.conv3 = nn.Conv2d(64, 128, 3) # 54 -3/1 +1 = 52
        
        self.maxpool3 = nn.MaxPool2d(2,2) # 52/2 = 26
        
        
        self.conv4 = nn.Conv2d(128,256,3) # 26 -3/1 +1 = 24
        
        self.maxpool4 = nn.MaxPool2d(2,2) # 24/2 = 12
        
        
        self.conv5 = nn.Conv2d(256,512,1) # 12 -1/1 +1 = 12
        
        self.maxpool5 = nn.MaxPool2d(2,2) # 12/2 = 6
        
        self.dropout = nn.Dropout(p = 0.2)
        
        self.linear = nn.Linear(512*6*6 ,1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512,136)
        
        
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.dropout(self.maxpool(F.relu(self.conv1(x))))
        x = self.dropout(self.maxpool2(F.relu(self.conv2(x))))
        x = self.dropout(self.maxpool3(F.relu(self.conv3(x))))
        x = self.dropout(self.maxpool4(F.relu(self.conv4(x))))
        x = self.dropout(self.maxpool5(F.relu(self.conv5(x))))
                         
        # Linear layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
