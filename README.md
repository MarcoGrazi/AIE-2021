# AIE-2021
Artificial Neural Network implementation inspired by the book "Artificial Intelligence Engines" by James Stone.
The code implements:
- neural networks with fully connected layers with the possibility to choose number of layers, neurons and activation function (linear, sigmoid, tanh, relu, softmax),
- backpropagation algorithm
- Gradient Descent training algorithm, with the possibility to choose batchsize, learning rate, momentum, number of epochs, loss function,
  metrics, validation data split ratio, EarlyStopping and ReduceOnPleateau criteria

I tested the network on a simple housing dataset, as a regression model, with noticeable results. The loss function does reach a mimimum through training and the metrics 
do improve.
