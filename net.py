import torch.nn as nn
import torch.nn.functional as F

#Define my own ResNet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, down):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = conv1x1(inplanes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.down = down
    def forward(self, x):
        residual = x.clone()
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.down == True:
            residual = self.downsample(x)
            residual = self.bn3(residual)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1_1 = ResBlock(64, 64, 2, True)
        self.resblock1_2 = ResBlock(64, 64, 1, False)
        self.resblock2_1 = ResBlock(64, 128, 2, True)
        self.resblock2_2 = ResBlock(128, 128, 1, False)
        self.resblock3_1 = ResBlock(128, 256, 2, True)
        self.resblock3_2 = ResBlock(256, 256, 1, False)
        self.resblock4_1 = ResBlock(256, 512, 2, True)
        self.resblock4_2 = ResBlock(512, 512, 1, False)
        self.avgpooling = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc1 = nn.Linear(512, 10)
        
    def visualize(self, x):
        x = self.conv1(x)
        out1 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.resblock1_1(x)
        x = self.resblock1_2(x)
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.avgpooling(x)
        out2 = x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        
        return out1, out2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.resblock1_1(x)
        x = self.resblock1_2(x)
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.avgpooling(x)
 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        
        return x