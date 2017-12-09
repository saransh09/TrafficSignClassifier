# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic Summary of the dataset

I used numpy and the pandas library to assess each class and the number of images that were present in each of the image

* The size of training set is 39209
* The size of the validation set is 6274
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

For the visualization please refer to the notebook

I performaed various visualizations and plotted the frequencies of each image class 

What I inferred from this was that :
* the quality of images was really poor and a probably a higher resolution image should have been used
* the brightness and the clarity of the actual captured images also differed, which could result in bad results
* There was a huge class imbalance in the dataset, which is extremely poor for the dataset and therefore we will have to create more data for some particular classes

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried many pre processing steps, and finally decided that I stick with the conversion of image to grayscale followed by normalizing the image, 
for Normalizing the image I used the (X - mu)/sigma formula, which is considered a better way to normalize the image

For the data augmentation process, I makde some custom functions:
* Random Scaling
* Random Translation
* Random Warping
* Random Brightness

These were specifically chosen, because they boosted the performance of the dataset massively

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For this model, I have used the LeNet architecture

I have used three convolution Layers, followed by a layer, which has the concatenated results of the outputs of the previous two layers

1) Conv2D layer - with a 5x5 filter and depth 6 <br>
2) Activation layer in the form of ReLu<br>
3) Max Pool Layer with strides as 2x2<br>
4) Conv2D layer - with a 5x5 filter and a depth 16<br>
5) Activation layer in the form of ReLu<br>
6) Max Pool layer with strides 2x2<br>
7) Conv2D Layer with 5x5 filter size and 400 depth<br>
8) Activation Layer in the form of ReLu<br>
9) Layer 4 which consists of the flattened output of layer2 and layer3<br>
10) dropout layer<br>
11) final output layer or logits<br>

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I tried various hyperparameters and finally I sticked with the following <br>

learning rate = 0.0009
NUMBER_EPOCHS = 50
BATCH_SIZE = 120
Optimizer = Adam

I shuffled the data and trained it in batches

By the end of training the validation accuracy stood at around 99.4%
which is very decent

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 99.4 
* test set accuracy of 95%

I basically used the LeNet architecture for the purpose and made a few simple changes to it
Also, I sticked to a basic model only and tried to tune the hyperparameters more as it looked right and the results that were achived were quite decent, so I did not feel like playing way too much with the model architecture


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found around seven images on the internet, bt unfortunately they were of very poor quality which might be a problem, the reason being they were of varied resolutions and the resizing operation which is necessary to be performed on the images, for the purpose of prediction might really blow the qualtiy of the image

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 58%. 

Although this might look bad, but I am not really concerned the reason being there was way much noise in the image due to a couple of website logos that were floation in 3 of the images and also the resizing operation as mentioned above which would be a problem if the resolution of the image is very skewed which is actually the case in 3 of our images


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image that was turn right it was 100% certain
for the second image it 30km/hr it was 90% certain
for the third image 60km/hr it was 0% certain
for the fourth image it was 100% certain
for the fifth image it was 100% certain
for the sixth image it was 27% certain

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


