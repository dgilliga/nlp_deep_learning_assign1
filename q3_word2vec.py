#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE

    # USE L2 normalization
    sqrs = np.square(x)
    sqr_sums = np.sum(sqrs, axis=1)

    l2 = np.sqrt(sqr_sums)[:, np.newaxis]
    x = x / l2
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE

    # z = u_o dot v_c
    # Dimensions: z = 1*Dw * Dw * W
    # Dw is length of single word vector
    # W is number of words
    # Dimensions of z are 1*W
    z = np.dot(predicted, outputVectors.T)

    # y_hat = 1*W
    y_hat = softmax(z)

    # Same as -Sum(y_i*log(y_hat_i)) where y is target one hot encoded vector
    # and we are given index i (target)

    cost = -np.log(y_hat[target])

    # Our gradient below is equivalent our derived gradient y-yhat
    # where y is one hot encoded vector
    # with one hot at index=target.
    # Dimensions: DJ_dy = 1*W
    DJ_dz = y_hat
    DJ_dz[target] -= 1

    # dJ_dvc = dJ_dz dot dz_dvc
    # Dimensions: dJ_dvc= 1*W * W*Dx = 1*Dx
    gradPred = np.dot(DJ_dz, outputVectors)

    # dJ_dU = dJ_dz.T dot dz_dU
    # Dimensions: dJ_dU = W*1 * 1*Dx
    grad = np.dot(DJ_dz[np.newaxis, :].T, predicted[np.newaxis, :])

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    # dimensions of a single word vector
    Dw = outputVectors.shape[1]

    ### YOUR CODE HERE
    grad = np.zeros_like(outputVectors)

    # Target + negative word vectors
    outputChosenWords = outputVectors[indices, :]

    # 1st vector is positive target vector
    # the rests are negative
    directions = np.array([-1] * len(indices))
    directions[0] = 1

    # Dimensions: K+1*Dw * Dw*1 = K+1*1

    sigmoid_vec = sigmoid(np.dot(outputChosenWords, predicted) * directions)
    cost = -np.sum(np.log(sigmoid_vec))

    # dJ_dvc = (sigma(uo*vc)-1)*uo - sum( (sigma(uk*vc)-1)*uk)
    # Dimensions: 1*K+1 dot K+1*Dw = 1 *Dw
    sigmoid_vec_minus = (sigmoid_vec - 1) * directions
    gradPred = np.dot(sigmoid_vec_minus.reshape(1, K + 1), outputChosenWords).flatten()

    # Dimensions: gradMin = K+1 * Dw
    # This is the gradient for all our target word and all our negative sampled words.
    gradMin = np.dot(sigmoid_vec_minus.reshape(K + 1, 1), predicted.reshape(1, Dw))

    # Dimensions: grad = 1*K+1
    # grad includes dJ_duo as first term and remaining terms are dJ_duk.
    # gradient for only the words that were selected is updated
    # Some words are duplicated and will have gradient updated multiple times

    for k in xrange(K + 1):
        grad[indices[k]] += gradMin[k, :]

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    for contextWord in contextWords:
        target_index = tokens[contextWord]
        vc_index = tokens[currentWord]
        cur_cost, cur_gradIn, cur_gradOut = word2vecCostAndGradient(inputVectors[vc_index], target_index, outputVectors,
                                                                   dataset)

        cost += cur_cost
        gradIn[tokens[currentWord]] += cur_gradIn
        gradOut += cur_gradOut

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #                 dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #                 dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"],
    #            dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"],
    #            dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
    #            negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
