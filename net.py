import torch
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
        self.resblock1_1 = ResBlock(64, 64, 1, True)
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
    
    
class ResNet_deconv(nn.Module):
    def __init__(self, demode=1):
        super(ResNet_deconv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.resblock1_1 = ResBlock(64, 64, 1, True)
        self.resblock1_2 = ResBlock(64, 64, 1, False)
        self.resblock2_1 = ResBlock(64, 128, 2, True)
        self.resblock2_2 = ResBlock(128, 128, 1, False)
        self.resblock3_1 = ResBlock(128, 256, 2, True)
        self.resblock3_2 = ResBlock(256, 256, 1, False)
        self.resblock4_1 = ResBlock(256, 512, 2, True)
        self.resblock4_2 = ResBlock(512, 512, 1, False)
        
        self.deconv1_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.deconv1_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(512)
        
        self.deconv1_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(512)
        self.deconv1_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(256)
        
        self.deconv2_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.deconv2_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(256)
        
        self.deconv2_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(256)
        self.deconv2_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(128)
        
        self.deconv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.deconv3_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        self.deconv3_3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.deconv3_4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(64)
        
        self.deconv4_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.deconv4_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(64)
        
        self.deconv4_3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(64)
        self.deconv4_4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(64)
        
        self.demaxpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        
        self.demode = demode
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        net = ResNet()
        net.eval()
        net = net.cuda()
        net.load_state_dict(torch.load('./cifar_net.pth'))
        params=net.state_dict() 
        if self.demode == 2:
                    
            for name, module in net._modules.items():
                if name == 'conv1':
                    self.conv1 = module
                if name == 'bn1':
                    self.bn1 = module
                if name == 'relu':
                    self.relu = module
#                 if name == 'maxpool':
#                     self.maxpool = module
                if name == 'resblock1_1':
                    self.resblock1_1 = module
                if name == 'resblock1_2':
                    self.resblock1_2 = module
                if name == 'resblock2_1':
                    self.resblock2_1 = module
                if name == 'resblock2_2':
                    self.resblock2_2 = module
                if name == 'resblock3_1':
                    self.esblock3_1 = module
                if name == 'resblock3_2':
                    self.resblock3_2 = module
                if name == 'resblock4_1':
                    self.resblock4_1 = module
                if name == 'resblock4_2':
                    self.resblock4_2 = module

            for k,v in params.items():

                if k == 'resblock1_1.conv1.weight':
                    self.deconv4_4.weight.data = v.data
                if k == 'resblock1_1.conv2.weight':
                    self.deconv4_3.weight.data = v.data
                if k == 'resblock1_2.conv1.weight':
                    self.deconv4_2.weight.data = v.data
                if k == 'resblock1_2.conv2.weight':
                    self.deconv4_1.weight.data = v.data
                if k == 'resblock2_1.conv1.weight':
                    self.deconv3_4.weight.data = v.data
                if k == 'resblock2_1.conv2.weight':
                    self.deconv3_3.weight.data = v.data
                if k == 'resblock2_2.conv1.weight':
                    self.deconv3_2.weight.data = v.data
                if k == 'resblock2_2.conv2.weight':
                    self.deconv3_1.weight.data = v.data
                if k == 'resblock3_1.conv1.weight':
                    self.deconv2_4.weight.data = v.data
                if k == 'resblock3_1.conv2.weight':
                    self.deconv2_3.weight.data = v.data
                if k == 'resblock3_2.conv1.weight':
                    self.deconv2_2.weight.data = v.data
                if k == 'resblock3_2.conv2.weight':
                    self.deconv2_1.weight.data = v.data
                if k == 'resblock4_1.conv1.weight':
                    self.deconv1_4.weight.data = v.data
                if k == 'resblock4_1.conv2.weight':
                    self.deconv1_3.weight.data = v.data
                if k == 'resblock4_2.conv1.weight':
                    self.deconv1_2.weight.data = v.data
                if k == 'resblock4_2.conv2.weight':
                    self.deconv1_1.weight.data = v.data
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data
        else:
            for name, module in net._modules.items():
                if name == 'conv1':
                    self.conv1 = module

            for k,v in params.items():
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data

        

    
    
    def forward(self, x):
        if self.demode ==2:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x, indices = self.maxpool(x)

            x = self.resblock1_1(x)
            x = self.resblock1_2(x)
            x = self.resblock2_1(x)
            x = self.resblock2_2(x)
            x = self.resblock3_1(x)
            x = self.resblock3_2(x)
            x = self.resblock4_1(x)
            x = self.resblock4_2(x)
            x = self.bn1_1(self.deconv1_1(self.relu(x)))
            x = self.bn1_2(self.deconv1_2(self.relu(x)))
            x = self.bn1_3(self.deconv1_3(self.relu(x)))
            x = self.bn1_4(self.deconv1_4(self.relu(x)))
            x = self.bn2_1(self.deconv2_1(self.relu(x)))
            x = self.bn2_2(self.deconv2_2(self.relu(x)))
            x = self.bn2_3(self.deconv2_3(self.relu(x)))
            x = self.bn2_4(self.deconv2_4(self.relu(x)))
            x = self.bn3_1(self.deconv3_1(self.relu(x)))
            x = self.bn3_2(self.deconv3_2(self.relu(x)))
            x = self.bn3_3(self.deconv3_3(self.relu(x)))
            x = self.bn3_4(self.deconv3_4(self.relu(x)))
            x = self.bn4_1(self.deconv4_1(self.relu(x)))
            x = self.bn4_2(self.deconv4_2(self.relu(x)))
            x = self.bn4_3(self.deconv4_3(self.relu(x)))
            x = self.bn4_4(self.deconv4_4(self.relu(x)))
            x = self.demaxpool(x, indices)
            x = self.deconv5(self.relu(x))
        else:
            x = self.conv1(x)
            x = self.deconv5(x)
        
        return x
    
class ResNet_encorder(nn.Module):
    def __init__(self, demode=1):
        super(ResNet_encorder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.resblock1_1 = ResBlock(64, 64, 1, True)
        self.resblock1_2 = ResBlock(64, 64, 1, False)
        self.resblock2_1 = ResBlock(64, 128, 2, True)
        self.resblock2_2 = ResBlock(128, 128, 1, False)
        self.resblock3_1 = ResBlock(128, 256, 2, True)
        self.resblock3_2 = ResBlock(256, 256, 1, False)
        self.resblock4_1 = ResBlock(256, 512, 2, True)
        self.resblock4_2 = ResBlock(512, 512, 1, False)

        self.demode = demode
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        net = ResNet()
        net.eval()
        net = net.cuda()
        net.load_state_dict(torch.load('./cifar_net.pth'))
        params=net.state_dict() 
        if self.demode == 2:
                    
            for name, module in net._modules.items():
                if name == 'conv1':
                    self.conv1 = module
                if name == 'bn1':
                    self.bn1 = module
                if name == 'relu':
                    self.relu = module
                if name == 'resblock1_1':
                    self.resblock1_1 = module
                if name == 'resblock1_2':
                    self.resblock1_2 = module
                if name == 'resblock2_1':
                    self.resblock2_1 = module
                if name == 'resblock2_2':
                    self.resblock2_2 = module
                if name == 'resblock3_1':
                    self.esblock3_1 = module
                if name == 'resblock3_2':
                    self.resblock3_2 = module
                if name == 'resblock4_1':
                    self.resblock4_1 = module
                if name == 'resblock4_2':
                    self.resblock4_2 = module

        else:
            for name, module in net._modules.items():
                if name == 'conv1':
                    self.conv1 = module
        
    def forward(self, x):
        if self.demode ==2:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x, indices = self.maxpool(x)

            x = self.resblock1_1(x)
            x = self.resblock1_2(x)
            x = self.resblock2_1(x)
            x = self.resblock2_2(x)
            x = self.resblock3_1(x)
            x = self.resblock3_2(x)
            x = self.resblock4_1(x)
            x = self.resblock4_2(x)
            
        else:
            x = self.conv1(x)
            indices = None
        return x, indices

class ResNet_decorder(nn.Module):
    def __init__(self, demode=1):
        super(ResNet_decorder, self).__init__()
        
        self.deconv1_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.deconv1_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(512)
        
        self.deconv1_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(512)
        self.deconv1_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(256)
        
        self.deconv2_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.deconv2_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(256)
        
        self.deconv2_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(256)
        self.deconv2_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(128)
        
        self.deconv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.deconv3_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        self.deconv3_3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.deconv3_4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(64)
        
        self.deconv4_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.deconv4_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(64)
        
        self.deconv4_3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(64)
        self.deconv4_4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(64)
        
        self.demaxpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.demode = demode
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        net = ResNet()
        net.eval()
        net = net.cuda()
        net.load_state_dict(torch.load('./cifar_net.pth'))
        params=net.state_dict() 
        if self.demode == 2:
            
            for k,v in params.items():

                if k == 'resblock1_1.conv1.weight':
                    self.deconv4_4.weight.data = v.data
                if k == 'resblock1_1.conv2.weight':
                    self.deconv4_3.weight.data = v.data
                if k == 'resblock1_2.conv1.weight':
                    self.deconv4_2.weight.data = v.data
                if k == 'resblock1_2.conv2.weight':
                    self.deconv4_1.weight.data = v.data
                if k == 'resblock2_1.conv1.weight':
                    self.deconv3_4.weight.data = v.data
                if k == 'resblock2_1.conv2.weight':
                    self.deconv3_3.weight.data = v.data
                if k == 'resblock2_2.conv1.weight':
                    self.deconv3_2.weight.data = v.data
                if k == 'resblock2_2.conv2.weight':
                    self.deconv3_1.weight.data = v.data
                if k == 'resblock3_1.conv1.weight':
                    self.deconv2_4.weight.data = v.data
                if k == 'resblock3_1.conv2.weight':
                    self.deconv2_3.weight.data = v.data
                if k == 'resblock3_2.conv1.weight':
                    self.deconv2_2.weight.data = v.data
                if k == 'resblock3_2.conv2.weight':
                    self.deconv2_1.weight.data = v.data
                if k == 'resblock4_1.conv1.weight':
                    self.deconv1_4.weight.data = v.data
                if k == 'resblock4_1.conv2.weight':
                    self.deconv1_3.weight.data = v.data
                if k == 'resblock4_2.conv1.weight':
                    self.deconv1_2.weight.data = v.data
                if k == 'resblock4_2.conv2.weight':
                    self.deconv1_1.weight.data = v.data
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data
        else:

            for k,v in params.items():
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data

    def forward(self, x, indices):
        
        if self.demode ==2:
            
            x = self.bn1_1(self.deconv1_1(self.relu(x)))
            x = self.bn1_2(self.deconv1_2(self.relu(x)))
            x = self.bn1_3(self.deconv1_3(self.relu(x)))
            x = self.bn1_4(self.deconv1_4(self.relu(x)))
            x = self.bn2_1(self.deconv2_1(self.relu(x)))
            x = self.bn2_2(self.deconv2_2(self.relu(x)))
            x = self.bn2_3(self.deconv2_3(self.relu(x)))
            x = self.bn2_4(self.deconv2_4(self.relu(x)))
            x = self.bn3_1(self.deconv3_1(self.relu(x)))
            x = self.bn3_2(self.deconv3_2(self.relu(x)))
            x = self.bn3_3(self.deconv3_3(self.relu(x)))
            x = self.bn3_4(self.deconv3_4(self.relu(x)))
            x = self.bn4_1(self.deconv4_1(self.relu(x)))
            x = self.bn4_2(self.deconv4_2(self.relu(x)))
            x = self.bn4_3(self.deconv4_3(self.relu(x)))
            x = self.bn4_4(self.deconv4_4(self.relu(x)))
            x = self.demaxpool(x, indices)
            x = self.deconv5(self.relu(x))
        else:
            
            x = self.deconv5(x)
        
        return x
    
class ResNet_decorder2(nn.Module):
    def __init__(self, demode=1):
        super(ResNet_decorder2, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                     padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.demaxpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.demode = demode
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        net = ResNet()
        net.eval()
        net = net.cuda()
        net.load_state_dict(torch.load('./cifar_net.pth'))
        params=net.state_dict() 
        if self.demode == 2:
            
            for k,v in params.items():

                if k == 'resblock1_1.downsample':
                    self.deconv4.weight.data = v.data
                if k == 'resblock2_1.downsample':
                    self.deconv3.weight.data = v.data
                if k == 'resblock3_1.downsample':
                    self.deconv2.weight.data = v.data
                if k == 'resblock4_1.downsample':
                    self.deconv1.weight.data = v.data
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data
        else:

            for k,v in params.items():
                if k == 'conv1.weight':
                    self.deconv5.weight.data = v.data

    def forward(self, x, indices):
        
        if self.demode ==2:
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.demaxpool(x, indices)
            x = self.deconv5(x)

        else:
            
            x = self.deconv5(x)
        
        return x