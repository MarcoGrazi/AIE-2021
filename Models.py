import math as M
import random as R

# TODO capire come implementare il softmax al ritorno
class Model:
    G = []   # model graph
    L = []   # layers of neurons

    # G (Graph) = [number of layers][number of neurons, activation]
    def __init__(self, G, bias):
        self.G = G
        self.L = []
        for i, layer in enumerate(G):
            l = []
            for _ in range(layer[0]):
                if i > 0:
                    n = Neuron(layer[1], G[i-1][0], bias)
                else:
                    n = Neuron(layer[1], -1, bias)
                l.append(n)
            self.L.append(l)

    def feedforward(self, x):
        for i, v in enumerate(x):
            self.L[0][i].state = v
        for i in range(1, len(self.L)):
            for n in self.L[i]:
                n.update_state(self.L[i-1])
        output = []
        for n in self.L[-1]:
            output.append(n.state)

        if self.G[-1][1] == 'softmax':
            for i, v in enumerate(output):
                output[i] = M.exp(v)
            s = sum(output)
            for i, v in enumerate(output):
                output[i] = v/s
        return output

    def backpropagate(self, y_pred, y_true, E):
        y_pred = [y_pred]
        y_true = [y_true]
        # update errors of output neurons
        for i, n in enumerate(self.L[-1]):
            # the multiplication of the delta between predicted and target values by the value of the loss function is
            # my own interpretation, to make use of the loss function other than a mere monitoring tool.
            n.error = float(n.error + (y_true[i] - y_pred[i])*E * n.Derivative(n.state))
            for j, v in enumerate(n.inputs):
                n.gradient[j] = n.gradient[j] + n.error * v
            n.gradient[-1] = n.gradient[-1] + n.error * n.bias

        # iterate through layers, except the output one, in reverse
        for i in reversed(range(1, len(self.L)-1)):
            for j, n in enumerate(self.L[i]):
                w = []
                # retrieve all weights from next layer neurons to the jth neuron of the current layer
                for nn in self.L[i+1]:
                    w.append(nn.W[j])
                n.update_error(w, self.L[i+1])
        for i in range(1, len(self.L)):
            for n in self.L[i]:
                n.state = 0

    def train(self, momentum, learning_rate):
        for layer in self.L:
            for neuron in layer:
                neuron.update_weights(momentum, learning_rate)
        for layer in self.L:
            for n in layer:
                n.error = 0
                n.state = 0


class Neuron:
    state = 0
    error = 0
    gradient = []
    pre_gradient = []
    inputs = []
    bias = 0
    W = []
    activation = 'linear'

    def __init__(self, activation, plnn, bias):  # activation function, previous layer neurons number, bias neuron value
        self.error = 0
        self.state = 0
        self.gradient = []
        self.pre_gradient = []
        self.inputs = []
        self.activation = activation
        self.W = self.Initialize(plnn+1)  # weights to previous layer neurons + bias
        self.bias = bias
        # gradient initialization
        for i in range(len(self.W)):
            self.gradient.append(0)

    def Initialize(self, n):
        w = []
        for _ in range(0, n):
            w.append(R.uniform(-1, 1))
        return w

    def Activation(self, z):
        o = 0
        if self.activation == 'linear' or self.activation == 'softmax':
            o = z
        elif self.activation == 'sigmoid':
            o = 1/(1+M.exp(-z))
        elif self.activation == 'tanh':
            o = M.tanh(z)
        elif self.activation == 'relu':
            o = max(0, z)
        return o

    def Derivative(self, z):
        e = 0
        if self.activation == 'linear' or self.activation == 'softmax':
            e = z
        elif self.activation == 'sigmoid':
            e = self.Activation(z)/(1 - self.Activation(z))
        elif self.activation == 'tanh':
            e = 1/(M.cosh(z))**2
        elif self.activation == 'relu':
            if z < 0:
                e = 0
            else:
                e = z
        return e

    def update_error(self, wf, nln):     # Weight forward, next layer neurons
        e = 0
        # update neuron error (delta)
        for i in range(len(wf)):
            e = e + nln[i].error * wf[i]
        self.error = e * self.Derivative(self.state)  # there is no bias weight in the forward direction
        # calculate gradient for each weight:
        for i, v in enumerate(self.inputs):
            self.gradient[i] = self.gradient[i] + self.error * v
        self.gradient[-1] = self.gradient[-1] + self.error * self.bias
        self.inputs.clear()

    def update_state(self, pln):  # previous layer neurons
        z = 0
        self.inputs = []
        for i, n in enumerate(pln):
            z = z + n.state * self.W[i]
            self.inputs.append(n.state)
        z = z + self.bias * self.W[-1]
        self.state = self.Activation(z)

    def update_weights(self, momentum, learning_rate):
        if len(self.pre_gradient) != 0:
            # if it is not the first weight update, we can apply the momentum parameter
            for i in range(len(self.gradient)):
                self.W[i] = self.W[i] + momentum*self.pre_gradient[i]*learning_rate + learning_rate*self.gradient[i]
                self.pre_gradient[i] = self.gradient[i]
                self.gradient[i] = 0
        else:
            for i in range(len(self.gradient)):
                self.W[i] = self.W[i] + learning_rate*self.gradient[i]
                self.pre_gradient.append(self.gradient[i])
                self.gradient[i] = 0






