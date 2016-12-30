Summary of hardware:  
  
I collect data and run the simulator on a Lenovo thinkpad laptop that has 8 gigs ram and i7 processor. I did training on a linux box (employer-supplied server) that has 64 cores and plenty of ram (so no need for a python generator).

Summary of previous attempts:  
  
I first tried collecting my own training data, but I only had a keyboard and mouse, and I did not emphasize recovery data.  
Then, I switched to using the Udacity training set, but only the center camera (full image, not preprocessing). I tried a simple model, and then scaled up to NVidia's architecture. I was still not able to get past the first right turn. I then tried to use VGG16's first two convolution blocks as feature extractors, but this method failed because it was too computationally costly (I had to scale the image up to 299 x 299).
  
Preparing Training Data:  
  
I only used the Udacity dataset, and I used all three camera images. For each image, I crop the image so that I only have rows 40-140 (got rid of the hood of the car and the irrelevant sky). Then I scale down the image by 4, so the final image is 25 rows and 80 columns. For the left/right cameras, the steering angle was corrected by .1 if the absolute value of the original steering angle was below .25, otherwise the correction was .25. This was meant so that the model learned that recovery angles are more severe in a turn.

Model Architecture:  
  
Input: 25x80x3  
Conv2d: 5x5 60 maps  
maxpool 2x2  
Conv2d: 3x3 30 maps  
maxpool 2x2    
flatten  
dropout with 20% chance of dropout  
fully connected 128 nodes  
fully connected 50 nodes  
fully connected to one output  

Relevant activations are relu.  
  
Training Details:  
  
Kept random 10% hold-out set as end of epoch validation. MSE loss and adam optimizer with default learning rate. Early stopping of training based on improvement of hold out loss.  
  
General Notes about perfomance:  
  
  
  
