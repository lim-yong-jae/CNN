Using **Transfer Learning** for training. I implement FCN-8s using VGG16. 
FCN is composed of two parts. VGG16 makes heatmap, and it is not trained in my code because i think there should be criteria. FCN's upsample section is trained. 


# Architecture  
I implement FCN-8s in paper. The architecture of it is equal to the paper's. I seperate "8x stride" to "2x stride" 3 times for making CNN more deeper. There is a result that DNN has more higher performance when it is more deeper.  


# Dataset  
dataset1 is the dataset that i use, and i get it from: https://github.com/divamgupta/image-segmentation-keras. 


# Problem 
 1) VGG16's Conv layer's output 1/2 size of input size. What if input is odd? Under decimal point's value is rounded down. It makes dimension incorrect when VGG16's pool3,4,5's results are concatenated. So i should transform train image size to specific size which should be multiples of 2^5 (5 is conv layer's count) 
 
 2) My CNN classify only 2 class when i use CrossEntropy Loss function, but there are 11 class. So i should use another loss function that deal with predicted output independently. So i use BCELoss function which can be used for multiclass that is BCEWithLogitsLoss.  

# Train Result


# Reference:  
* paper: https://arxiv.org/pdf/1411.4038.pdf   
* Uing image dataset at: https://github.com/divamgupta/image-segmentation-keras  
