import os
import sys
import numpy as np
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from net import ResNet
from tensorboardX import SummaryWriter
writer = SummaryWriter('./res/')
parser = argparse.ArgumentParser(description='pytorch model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                            help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int,  metavar='N', default=40,
                            help='number of epochs to train')
parser.add_argument('--gpu', type=list, default=[0,1,2,3],
                            help='gpu device number')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--network', type=str, default = 'ResNet',
                            help='ResNet or Net')
parser.add_argument('--save_model', type=bool, default=True,
                            help='True or False')
parser.add_argument('--load_model', type=bool, default=False,
                            help='True or False')
parser.add_argument('--mode', type=str, default='train',
                            help='train or test')
parser.add_argument('--model_path', type=str, default='./cifar_net.pth',
                            help='model path')
parser.add_argument('--cuda_num', type=str, default="1",
                            help='cuda device num.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
torch.cuda.manual_seed(args.seed)



def data_prepare():
    # Loading and normalizing CIFAR10
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), 2), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224), 2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    correct = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        if i % 50 == 0:
            correct_i = 0.0
            total_i = 0
            running_loss_i = 0.0
            with torch.no_grad():
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total_i += labels.size(0)
                    correct_i += (predicted == labels).sum().item()
                    running_loss_i += loss.item()

            iteration_loss = running_loss_i / float(len(trainloader))
            iteration_acc = 100. * float(correct_i) / float(len(trainloader.dataset))
            iteration = len(trainloader)* epoch + 1 + i
            writer.add_scalar('scalar/Train_Loss_iteration', iteration_loss ,iteration)
            writer.add_scalar('scalar/Train_Acc_iteration', iteration_acc, iteration)
      
    epoch_loss = running_loss / float(len(trainloader))
    epoch_acc = 100. * float(correct) / float(len(trainloader.dataset))
    print('[%3d] train loss: %.3f' % (epoch + 1, epoch_loss))
    print('[%3d] train auc: %.3f' % (epoch + 1, epoch_acc))
    writer.add_scalar('scalar/Train_Loss_epoch', epoch_loss , epoch + 1)
    writer.add_scalar('scalar/Train_Acc_epoch', epoch_acc, epoch + 1)
        
        
def test(net, criterion, testloader, epoch):
    
    correct = 0.0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()
            
    epoch_loss = test_loss / float(len(testloader))
    epoch_acc = 100. * float(correct) / float(len(testloader.dataset))
    print('[%3d] test loss: %.3f' % (epoch + 1, epoch_loss))
    print('[%3d] test auc: %.3f' % (epoch + 1, epoch_acc))
    writer.add_scalar('scalar/Test_Loss', epoch_loss , epoch + 1)
    writer.add_scalar('scalar/Test_Acc', epoch_acc, epoch + 1)

def test_class(net, testloader, epoch):
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        

if __name__ == '__main__':
    # Loading and normalizing CIFAR10
    trainloader, testloader, classes = data_prepare()
    # Define a Convolutional Neural Network
    net = ResNet()
    net = net.cuda()
    x = torch.autograd.Variable(torch.rand(64, 3, 32, 32))
    writer.add_graph(net, x.cuda(), verbose=True)
    if args.load_model == True:
        net.load_state_dict(torch.load(args.model_path))
    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)
    # Train the network
    for epoch in range(args.epochs):
        train(net, criterion, optimizer, trainloader, epoch)
        test(net, criterion, testloader, epoch)
    model_path = './cifar_net.pth'
    torch.save(net.state_dict(), model_path)
    
"Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"