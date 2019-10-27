# CNN-model-and-visualization
A Pytorch version CNN model (RseNet) for image classification( CIFAR-10), including filter and output of layers visualization.


## Run
### Download Dataset, Train and Test Model
```
python main.py
```

### Visualization of CNN layers and filters
```
python visualize.py
```

### Concatenate the subimages
```
python mkjpg.py
```

## Results
### Accuracy and Loss
**Report the final accuracy (10,000 steps) of training and testing for the CIFAR-10 dataset.**
The batch size is set to 64, so we have 652 iterations per epoch. 10000 steps mean that we train our model for 13 epochs. We get a training accuracy of 96.724%, and a testing accuracy of 86.340%.
```
[ 13] train loss: 0.104
[ 13] train acc: 9.724
[ 13] test loss: 0.438
[ 13] test acc: 86.340
```
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/res/res.png)

### Example picture to test
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/orginal.jpg)

### Filter Visualization (the first convolutional layer)
conv1, kernel_size = 7x7, stride = 2
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/filter/layer_vis_conv1.jpg)

### Feature maps Visualization (the first and last convolutional layer)
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/output/layer_vis_conv1.jpg)
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/output/layer_vis_resblock4_2_0.jpg)

### The reconstructed patterns that cause high activations in feature maps of the last convolutional layer
***Method: Using optimizer***
Using a certain feature maps as label, we set a parameter with size [1,3,224,224], and use it as resnet's input. By this way, we can obtain an output with the same size as the label. Calculating the sum square error, and backprop it to optimize the parameter to convergence. Finally we can get a reconstruction. We can get the reconstructed patterns by one certain filter or all the filters.
**By certain filter**
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/generated/layer_vis_resblock4_2_1.jpg)
**By all the filters**
*original pictures*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/final.jpg)
*conv1*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/generated/final_optimizer_conv1.jpg)
*resblock4*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/generated/final_optimizer_res4.jpg)
***Method: Using deconvolution***
"Backward" means taking the a certain residue blocks' output and doing the reversing manipulation. For convolutional layer, the reversing manipulation is the deconvolutional layer with same kernel value; for maxpooling, the reversing manipulation is unpooling. So the backward is the reconstruction pattern of the activation of a certain residue block.
*original pictures*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/final.jpg)
*conv1*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/deconv/final_deconv1.jpg)
*resblock4*
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/deconv/final_deconv2.jpg)

