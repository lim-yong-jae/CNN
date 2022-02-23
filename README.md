# CNN

# Loss function of Multi class classification
1) **CrossEntropyLoss**  
 I think that it is not suitable when given true data is label data. Because it classfies specific class. For example, 100 classes classification's problem, it classify specific class that occupies most data in train dataset. Because label data's probabilty is one when it is true class, and others are 0. So if we want to use CrossEntropyLoss in multi class classification, we should use weight for consdiering dataset's class's appearance count. If given true data is probabilty, CrossEntropyLoss is useful. 

2) **BCEWithLogitsLoss**   
 It takes all classes output independently. It calculate BCE Loss at every class. So it is useful when given true data is label form.  

## Fully convolution network  
Classifier using convolution layer, at last layer it uses fully connected layer. Fully connected layer can be replaced by convolution layer.

## Kernel size (3,3) 
kernel size (7,7) can be implemented by using kernel size (3,3). Convolution operation with kernel size (3,3) being done 2 times is equal to Convolution operation with kernel size (7,7). It has many advantages.   
1) The number of Parameters that should be trained reduce. Conv with kernel size (3,3) has 9 parameters. As it is done twice, so we have 18 parameters. But conv with kernel size (7,7) has 49 parameters.    

2)  We can build more deeper deep neural network by using kernel size (3,3). Accroding to "Very Deep Convolutional Networks for Large-Scale Image Recognition" the deeper CNN, the better performance.   

# Reference  
* Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/pdf/1409.1556.pdf    
* Fully Convolutional Networks for Semantic Segmentation: https://arxiv.org/pdf/1411.4038.pdf    
