# CNN

## fully convolution network  
Classifier using convolution layer, at last layer it uses fully connected layer. Fully connected layer can be replaced by convolution layer.

## kernel size (3,3) 
kernel size (7,7) can be implemented by using kernel size (3,3). Convolution operation with kernel size (3,3) being done 2 times is equal to Convolution operation with kernel size (7,7). It has many advantages.   
1) The number of Parameters that should be trained reduce. Conv with kernel size (3,3) has 9 parameters. As it is done twice, so we have 18 parameters. But conv with kernel size (7,7) has 49 parameters.    

2)  We can build more deeper deep neural network by using kernel size (3,3). Accroding to "Very Deep Convolutional Networks for Large-Scale Image Recognition" the deeper CNN, the better performance.   

# Reference  
* Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/pdf/1409.1556.pdf  
