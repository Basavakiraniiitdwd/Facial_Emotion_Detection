# Facial_Emotion_Detection


A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database. It is also described as a Biometric Artificial Intelligence based application that can uniquely identify a person by analyzing patterns based on the person’s facial textures and shape.
FACE EXPRESSION RECOGNITION
Facial expression recognition software is a technology which uses biometric markers to detect emotions in human faces. More precisely, this technology is a sentiment analysis tool and is able to automatically detect the six basic or universal expressions: happiness, sadness, anger, surprise, fear, and disgust.
I have created a model that will be recognising human emotion based on the dataset that is given. Here we have a list of emotions that includes “Angry” , “Disgust”, “Fear”, “Happy”, “Neutral”, “Sad” and “Surprise”.
For this , I downloaded one dataset from kaggle which we will be using in our model. You can download this dataset from kaggle.
You can create your own Dataset by taking your images or any facial images.
Image for post
We have to load MobileNet data to use it. This code is used to load MobileNet Data and remove the last layer from the model so that we can add our own. It also displays all the layers present in the model.
I have divided the dataset into training and testing and this will be helping at the time of prediction.The training data consist of 28821 images belonging to 7 classes & the testing data consists of 7066 images belonging to 7 classes.
Here the 7 classes belong to the following categories:
1) angry
2) disgust
3) fear
4) happy
5) neutral
6) sad
7) surprise

The most important part of the project is creating layers of Convolutional Neural Network. CNNs are regularized versions of multilayer percetrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. Fully connected layers are an essential component of Convolutional Neural Networks (CNNs), which have been proven very successful in recognizing and classifying images for computer vision.
Image for post
First we have created a Convolutional layer. Convolutional layers convolve the input and pass its result to the next layer. To increase the stability of a neural network, I have used batch normalization that normalizes the output of a previous activation layer.
I have used ‘relu’. ReLU stands for rectified linear unit, and is a type of activation function. Pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. Dropout is used for regularization.
I have added two fully connected layers as well. The input to the fully connected layer is the output from the final Pooling or Convolutional Layer, which is flattened and then fed into the fully connected layer.
Image for post
In the end we fit the model. We are training it for 15 complete cycle or epochs. We have set the learning rate at 0.00001 as it is a configurable hyperparameter used in training of neural network.
Image for post
The number of epochs increases the loss is reduced and the accuracy is increased.

#Conclution

Face Recognition works very well with transfer learning. Transfer learning is just using a model which is pretrained and stacking your layers on top of the the pretrained model for predictions based on your custom dataset.
Transfer learning is a method that allows us to use knowledge gained from other tasks in order tackle new but similar problems quickly and effectively. … This reduces the need for data related to the specific task we are dealing with.
You can use Transfer Learning to train your Face Recognition model by using a pre-trained model so that such big data can be trained easily taking less time and at the same time also costs less.
