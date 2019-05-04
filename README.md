# Simple-Java-Neural-Network
A simple framework in Java for creating a configurable neural network out of ReLUs or Sigmoids. 

To run the framework:

    javac NeuralNetwork.java 
    java NeuralNetwork example_config.txt

The framework will interpert and build a neural network that is defined in the example_config. The idea behind this is to have an easy configurable file that will allow the user to define network architecture in a simple manner.

The hw2_midterm_A_(eval/test/train) is the working dataset for the framework. It represents two midterm exams in each entry along with the binary classifaction on whether that student recieved a letter grade of A or not.

The network will run through the training set and calculate current error against the evaluation set. It will train for an infinite number of epochs as long as the cost function decreases.

Expected output with given example_config.txt: 

    WEIGHTS AFTER EPOCH:788
    Layer 0:
    Neuron 0 has the weights:1.20379,4.45346,-3.51459,end
    Neuron 1 has the weights:0.93258,3.44859,-2.72310,end
    Layer 1:
    Neuron 0 has the weights:2.11219,1.63381,-3.99435,end
    Error from eval set: 0.87178
    Accuracy of Neural Network with threshold 0.5 is: 0.96000

This project was my initial inspiration into the field of Machine Learning. This was first born out of my Introduction to Artificial Intelligence course as an undergrad. After graduating I wanted to build a framework that was easy to manipulate in order to try different architectures quickly. Upon inspection you can see how naive I was in the beginning. This is not a vector representation implmentation but an object oriented approach thus highly inefficient. It performs stochastic gradient descent, assumes dense connections between layers, and can only make single binary classsifcations. Despite the feature limitations it performs exactly as intended. I believe the object oriented style to the activation neurons and the SGD method is definitely unique. With a standard Neuron interface definied, it is open to having more activation functions implemented. 

Further development on this project has been shelved for focus on a modern approach to neural network frameworks. However this is a good display of an easily digestable approach to neural networks for beginners. 
