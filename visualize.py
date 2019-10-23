import os
import numpy as np
from net import ResNet
import torch
from torch.optim import Adam
from torchvision import models
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from misc_functions import preprocess_image, recreate_image, save_image
from matplotlib import pyplot as plt
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def data_prepare():
    # Loading and normalizing CIFAR10
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), 2), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, classes

def get_picture(trainloader):
    
    for i, data in enumerate(trainloader):
        pic = data[0].squeeze(0)
        if i == 0:
            break
    return pic

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, pic, lr, png_dir):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.pic = pic
        self.png_dir = png_dir
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.lr = lr
        # Create the folder to export images if not exists
        if not os.path.exists('./generated/'+ self.png_dir):
            os.makedirs('./generated/'+ self.png_dir)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)


    def visualise_layer_without_hooks(self):
        # Process image and return variable
       
        processed_image = self.pic[None,:,:,:]
        processed_image = processed_image.cuda()
        processed_image = Variable(processed_image, requires_grad=True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=self.lr, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for name, module in self.model._modules.items():
                # Forward pass layer by layer
                x = module(x)
                if name == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break

            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Save image
            if i % 200 == 0:
                # Recreate image
                processed_image = processed_image.cpu()
                self.created_image = recreate_image(processed_image)
                im_path = './generated/'+ self.png_dir+ '/layer_vis_' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

def layer_output_visualization(model, selected_layer, selected_filter, pic, png_dir):
    pic = pic[None,:,:,:]
    pic = pic.cuda()
    x = pic
    for name, module in model._modules.items():
        x = module(x)
        if name == selected_layer:
            break
    conv_output = x[0, selected_filter]
    x = conv_output.cpu().detach().numpy()
    if not os.path.exists('./output/'+ png_dir):
        os.makedirs('./output/'+ png_dir)
    im_path = './output/'+ png_dir+ '/layer_vis_' + str(selected_layer) + \
                    '_f' + str(selected_filter) + '_iter' + str(i) + '.jpg'
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    plt.savefig(im_path)
    
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data))*255 / _range
    
def filter_visualization(model, selected_layer, selected_filter, png_dir):
    for name, param in net.named_parameters():
        if name == selected_layer + '.weight':
            x = param
    x = x[selected_filter,:,:,:]
    x = x.cpu().detach().numpy()
    x = x.transpose(1,2,0)
    x = normalization(x)
    x = preprocess_image(x, resize_im=False)
    x = recreate_image(x)
    if not os.path.exists('./filter/'+ png_dir):
        os.makedirs('./filter/'+ png_dir)
    im_path = './filter/'+ png_dir+ '/layer_vis_' + str(selected_layer) + \
                    '_f' + str(selected_filter) + '_iter' + str(i) + '.jpg'
    save_image(x, im_path)
                
if __name__ == '__main__':

    #get model and data
    net = ResNet()
    net.eval()
    net = net.cuda()
    net.load_state_dict(torch.load('./cifar_net.pth'))
    trainloader, classes = data_prepare()
    pic = get_picture(trainloader)
    x = pic.cpu()[None, :, :, :]
    x = recreate_image(x)
    save_image(x, 'orginal.jpg')

    #define layer to visualize
    cnn_name = 'conv1'
    lr = 0.01
    for i in range(64):
        filter_pos = i
        layer_output_visualization(net, cnn_name, filter_pos, pic, cnn_name)
        filter_visualization(net, cnn_name, filter_pos, cnn_name)
        layer_vis = CNNLayerVisualization(net, cnn_name, filter_pos, pic, lr, cnn_name)
        layer_vis.visualise_layer_without_hooks()
        
    cnn_name = 'resblock4_2'
    lr = 0.1
    for i in range(512):
        filter_pos = i
        layer_output_visualization(net, cnn_name, filter_pos, pic, cnn_name)
        layer_vis = CNNLayerVisualization(net, cnn_name, filter_pos, pic, lr, cnn_name)
        layer_vis.visualise_layer_without_hooks()
        
"Reference: https://github.com/utkuozbulak/pytorch-cnn-visualizations"   
    