Using **Transfer Learning** for training. I implement FCN-8s using VGG16. 

# Architecture  
I implement FCN-8s in paper. The architecture of it is equal to the paper's. I seperate "8x stride" to "2x stride" 3 times for making CNN more deeper. There is a result that DNN has more higher performance when it is more deeper.  

# Dataset  
dataset1 is the dataset that i use, and i get it from: https://github.com/divamgupta/image-segmentation-keras. 

# Problem 
 My CNN classify only 2 class but there are 11 class. So i use custom loss function. My loss function is defined as
 

# Train Result


# Reference:  
* paper: https://arxiv.org/pdf/1411.4038.pdf   
* Uing image dataset at: https://github.com/divamgupta/image-segmentation-keras  
