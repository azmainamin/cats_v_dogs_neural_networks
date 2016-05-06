# cats_v_dogs_neural_networks
# CSCI-315: Artificial Intelligence Final Project

We used a neural network to distinguish between cats and dogs. For our dataset, we used Kaggle's** set of 25K images of cats and dogs. First, we divided the dataset into three parts: 12000 images for the training set, 8000 for testing set and 5000 for validation set. We used the file names to generate the target labels. We pickled the reshaped and grayscaled images and their labels. Our load_data() loads the pickled images and converts them into theano shared variables so that we can use our GPU to train our neural network.

We used MultiLayer Perceptron and Lenet5 network from deeplearning.com tutorials to make our neural network. We were able to successfully distinguish cats from dogs with ~70% accuracy. While we realize this is not revolutionary, we think this is a good result because the images are not straight-forward pictures of cats and dogs. There are a lot of noise in the images i.e. people holding the cat etc. But there are a lot of things to be improved. 

HOW TO USE:
1. Use prepare_data.py to get the pickled images and labels.
2. Run convolutional_mlp.py *
 

* You must have Theano installed and enable python to use your GPU. It is 4 times faster than using the CPU.
**Kaggle's dataset: https://www.kaggle.com/c/dogs-vs-cats/data


