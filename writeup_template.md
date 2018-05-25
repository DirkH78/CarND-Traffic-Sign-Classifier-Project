# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./newExamples/01_speed_limit_30.jpg "Speed Limit 30km/h"
[image2]: ./newExamples/12_priority.jpg "Priority Road"
[image3]: ./newExamples/13_yield.jpg "Yield"
[image4]: ./newExamples/14_stop.jpg "Stop"
[image5]: ./newExamples/17_no_entry.jpg "No Entry"
[image6]: ./newExamples/38_keep_right.jpg "Keep Right"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

A histogram of the training data was created and is displayed in the html-file. Especially the speed limits (30,50,70,80 km/h), priority road and yield occur more often than average.

### Design and Test a Model Architecture

#### 1. image conditioning

As a first step, I decided to convert the images to grayscale because color layers include redundant information that will only slow down the training process.

As a last step, I normalized the image data to create a zero mean data set with equal variance.

No additional optimizations (rotation, zoom into significant content) were implemented since the model needs to identify all possible angles when used in real world.


#### 2. Architecture

I used a standard LeNet as descriped in the course but could not archieve a satisfying result. So the following model was implemented:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x56 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x56 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x112  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x56 				|
| Convolution 4x4	    | 1x1 stride, same padding, outputs 2x2x24  	|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 896  									|
| Fully connected		| output 224  									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 43  									|
|						|												|
 


#### 3. Training

Best and quickest results were archieved by using:
Epochs: 25
Batch Size: 128
To adapt the rate and the keeping probability of the dropout to the advancing accuracy, a dynamic rate and KP was implemented. This made sure that the accuracy increased fast in the beginning of the training process and still increased at the end.

#### 4. Approach

My final model results were:
* validation set accuracy of ~97% 
* test set accuracy of ~95%

An iterative approach was chosen:
* I started by using the standard LeNet architecture provided in the course
* The test accuracy was < 90%
* The model was optimized by adding additional feature maps and two dropouts
* That resulted in a significant increase in the accuracy because the training stability was increased (dropouts) and additional layers for feature recognition were added.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![alt text][image1]
* High amount of training stes (+)
* Different shapes/colors in the background (-)

![alt text][image2]
* Very little background noise (+)
* Direct angle (+)

![alt text][image3]
* High amount of training stes (+)
* No direct angle (-)

![alt text][image4]
* Different shapes/colors in the background (-)

![alt text][image5]
* No direct angle (-)
* Different shapes/colors in the background (-)

![alt text][image6]
* Less background noise (+)
* No direct angle (-)
* Impurities (-)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30km/h				| 30km/h 										| 
| Priority Road			| Priority Road									|
| Yield					| Yield											|
| Stop					| Stop							 				|
| No Entry				| No Entry										|
| Keep Right			| Keep Right									|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

First image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| Speed limit (30km/h)							| 
| 0						| Speed limit (20km/h) 							|
| 0						| Speed limit (80km/h)							|
| 0						| Speed limit (50km/h)							|
| 0						| Speed limit (100km/h)							|


Second image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| Priority road									| 
| 0						| Roundabout mandatory 							|
| 0						| Speed limit (50km/h)							|
| 0						| End of no passing								|
| 0						| Keep right									|

Third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| Yield											| 
| 0						| Ahead only 									|
| 0						| No passing									|
| 0						| Speed limit (60km/h)							|
| 0						| No vehicles									|

Fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| Stop											| 
| 0						| Speed limit (60km/h) 							|
| 0						| No entry										|
| 0						| Speed limit (20km/h)							|
| 0						| Turn left ahead								|

Fifth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| No entry										| 
| 0						| Turn left ahead 								|
| 0						| No passing									|
| 0						| Keep right									|
| 0						| Stop											|

Sixth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1						| Keep right									| 
| 0						| General caution								|
| 0						| Turn left ahead								|
| 0						| Yield											|
| 0						| Dangerous curve right							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Couldn't get this working. any suggestions?
