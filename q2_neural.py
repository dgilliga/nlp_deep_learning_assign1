#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation

    # z1.shape = M * H
    z1 = data.dot(W1) + b1

    # h.shape = M * H
    h = sigmoid(z1)

    # z2.shape = M * Dy
    z2 = h.dot(W2) + b2

    # y_hat.shape = M * Dy
    y_hat = softmax(z2)
    ### END YOUR CODE

    # cost.shape = 1 * M
    cost = -np.sum(labels * np.log(y_hat))

    ### YOUR CODE HERE: backward propagation

    # dE_dz2.shape = M * Dy
    dE_dz2 = y_hat - labels  # Derivative of Error Function with Respect to z2

    # Relabel for convenience
    d1 = dE_dz2

    # dz2_dh.shape = H * Dy
    dz2_dh = W2

    # dE_dh.shape = M*H
    dE_dh = d1.dot(dz2_dh.T)

    # Relabel for convenience
    d2 = dE_dh

    # dh_dz1.shape = M * H
    dh_dz1 = sigmoid_grad(h)

    # dE_dz1.shape = M*H
    # Element wise multiplication:
    #
    dE_dz1 = np.multiply(d2, dh_dz1)

    # Relabel for convenience
    d3 = dE_dz1



    # dz1_dW1.shape = M * Dx
    dz1_dW1 = data

    # dz2_dW2.shape = M * H
    dz2_dW2 = h

    # dE/dW2.shape = H * M . M * Dy  = H * Dy
    # dE/dW2 = dE_dz2 * dz2_dW2

    gradW2 = np.dot(dz2_dW2.T, d1)

    # dE/dW1.shape = H * M . M * Dx = H * Dx
    # dE/dW1 = dE_dz1 *
    # gradW1 = np.dot(data.T, dE_dz1)
    gradW1 = np.dot(dz1_dW1.T, d3)
    dz1_db1 = 1

    dz2_db2 = 1

    # dE/dW1.shape = H * M . M * Dx = H * Dx
    gradb2 = np.sum(d1, axis=0)

    # dE/dW1.shape = 1*M . M * Dy
    gradb1 = np.sum(d3, axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
