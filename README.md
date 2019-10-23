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
```
[ 16] train loss: 0.065
[ 16] train acc: 97.986
[ 16] test loss: 0.702
[ 16] test acc: 82.280
```
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/res/res.png)

### Example picture to test
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/orginal.jpg)

### Filter Visualization(first convolutional layer)
![cmd-markdown-logo](https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/filter/layer_vis_conv1.jpg)

### Feature maps Visualization(first and last convolutional layer)
![cmd-markdown-logo](https://github.com/WangHelin1997/https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/output/layer_vis_conv1.jpg)
![cmd-markdown-logo](https://github.com/WangHelin1997/https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/output/layer_vis_resblock4_2_0.jpg)

### The reconstructed patterns that cause high activations in feature maps of the last convolutional layer
![cmd-markdown-logo](https://github.com/WangHelin1997/https://github.com/WangHelin1997/CNN-model-and-visualization/blob/master/generated/layer_vis_resblock4_2_1.jpg)

