**Traffic Sign Recognition** 

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

[image1]: ./write_up_image/gray.png "gray"
[image2]: ./write_up_image/no-gray.png "no-gray"
[image3]: ./write_up_image/test_dis.png "test_dis"
[image4]: ./write_up_image/train_dis.png "train_dis"
[image5]: ./write_up_image/valid_dis.png "valid_dis"
[image6]: ./real_word_image/bumpy_road_1.png "bumpy_road"
[image7]: ./real_word_image/caution_1.png "caution"
[image8]: ./real_word_image/limit50_1.png "limit50"
[image9]: ./real_word_image/limit120.png "limit120"
[image10]: ./real_word_image/road_work1.png "road_work"
[image11]: ./real_word_image/stop1.png "stop"
[image12]: ./write_up_image/filter.png "filter"
[image13]: ./write_up_image/filter2.png "filter2"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used library to calculate summary statistics of the traffic
signs data set, such as numpy.shape():

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. In the template there is a bar chart, but in my view a pie chart is a better choice, because it can show how much percentage each class occupy more detailly. As you can see, nearly each class take up similar scale(1%-5%), there are no classes having especially more images. So I think it's not necessary to solve imbalance problem. It is similar for valid set and test set.

![alt text][image4]
![alt text][image5]
![alt text][image3]

**Design and Test a Model Architecture**

1.Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I find without grayscale the train my model have only 89% accuracy, and then I apply grayscale on data set, the accuracy improve nearly 5%, I think it is because we compress data so that our model can easier learn feature from images.

Here is an example of a traffic sign image before and after grayscaling.


![alt text][image2]
![alt text][image1]

As a last step, I normalized the image data because it seems useful.

besides, I decided to generate additional data, more detailly, with rotation I got more data but when I use them to train my model, the result achieve lower performance. As result I didn't apply this approche in my project. After submition I will try other way to generate data and train model again.

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input = 400, hidden1 = 120, hidden2 = 84, Output = 43  |
| Softmax				|         									|

 

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, where learning rate = 0.001. And batch size is 128. For dropout I choose keep_prob = 0.9. And for number of epochs I find our model need more then 10 epochs so that it can learn better, so I choose 30 epochs, this need more time to train but model can achieve higher valid accuracy.

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.936
* test set accuracy of 0.927

I choose LeNet as fundamental architecture.Because our task is similar to MNIST problem, both of them are classification on images.With LeNet I got validation accuracy 0.91, it's not bad but still needs improving. So I apply dropout on convolutional layers, got higher validation accuracy 0.93.
 

**Test a Model on New Images**

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9] 
![alt text][image10]
![alt text][image11]

The "limit 120" image might be difficult to classify because in this image there are some noise, such as lane and another 
signage, I guess they will influence the final result. For similar reason, I think "road work" is also a little hard to be recognized.

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| caution     			| caution 										|
| 50 km/h					| End of speed limit (80km/h)										|
| 120 km/h	      		| double curve					 				|
| road work			|  road work      							|
| Stop sign			| Stop sign      							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. comparing with test result, it seems still to need improving. And I don't konw which factors contribute this problem. It would be nice you can give me some advice. 

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Bumpy	road sign (probability of almost 1.0), and the image does contain a Bumpy	road sign. The top five soft max probabilities were

| Probability	|Prediction	| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00|22,Bumpy	road| 
| 1.29787023e-27|28,Children	crossing|
| 4.53688143e-33|31,Wild	animals	crossing|
| 1.96769904e-37|20,Dangerous	curve	to	the	right|
| 2.09718608e-38|25,Road	work|


For the second image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 	18,General	caution							| 
| 9.73418984e-11    				| 11,Right-of-way	at	the	next	intersection   |
| 4.14415030e-11					| 40,Roundabout	mandatory											|
| 1.49591017e-11      			| 27,Pedestrians			|
| 1.52826764e-12				    | 5,Speed	limit	(80km/h)    							|

For the thrid image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5.79330266e-01         			| 	23,Slippery	road						| 
| 4.03478414e-01    				| 11,Right-of-way	at	the	next	intersection   |
| 1.23237120e-02					| 20,Dangerous	curve	to	the	right										|
| 2.78520002e-03      			| 30,Beware	of	ice/snow			|
| 1.94263062e-03				    | 10,No	passing	for	vehicles	over	3.5	metric	tons    							|

This image was classified by mistake, as you can see in list, our model was also not very sure which class to choose.

For the fourth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 	6,End	of	speed	limit	(80km/h)						| 
| 2.46765097e-08    				| 4,Speed	limit	(70km/h)   |
| 2.07055906e-09					| 1,Speed	limit	(30km/h)										|
| 5.12363485e-10      			| 5,Speed	limit	(80km/h)			|
| 6.86260038e-15				    | 2,Speed	limit	(50km/h)   							|

This image was also classified by mistake, but unlike the thrid image, our model said it was sure this image is "End	of	speed	limit	(80km/h)", but unlucky, it are not. Another point should to be noticed, is the top five result are all about speed	limit, and this image is actul speed	limit. So it makes sense, even it mistakes.

For the fifth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.28746271e-01         			| 	25,Road	work						| 
| 1.53453305e-01    				| 21,Double	curve   |
| 1.50383070e-01					| 31,Wild	animals	crossing										|
| 4.30265777e-02      			| 29,Bicycles	crossing			|
| 1.14517119e-02				    | 24,Road	narrows	on	the	right   							|

For the sixth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 	14,Stop						| 
| 1.14233883e-11    				| 36,Go	straight	or	right   |
| 7.49400755e-21					| 34,Turn	left	ahead										|
| 4.40621793e-23      			| 13,Yield			|
| 3.18760548e-25				    | 20,Dangerous	curve	to	the	right   							|

**(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)**
1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Let us look at visualized image of first and sencond convolutional layers'  filter.

![alt text][image12]
![alt text][image13]

From first convolutional layer we can know this model can capture shape characteristics. But second convolutional layer seems dosen't make sense. How can I analyze result like this?


