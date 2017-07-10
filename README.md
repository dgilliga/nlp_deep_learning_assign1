# nlp_deep_learning_assign1
Part of CS224n: Natural Language Processing with Deep Learning

This assignment requires user to manually derive the back propagation gradients by hand and then codes them up.
Then implement word2vec style skip gram to do sentiment analysis.

Part 1 implements Softmax.
 
Part 2 implements feedforward neural network with sigmoid hidden layer, softmax
score function and cross entropy loss function. Also implements backpropagation algorithm.

Part 3 implements basic skip gram model as per word2vec. Softmax and cross entropy 
loss function used. Also negative sample loss function used. Stochastic gradient descent
implemented and used to train word vectors with Stanford Sentiment Treebank (SST). This took
about 50 mins and created a 2d visualization of word vectors.
![2d visualization output](/q3_word_vectors.png)

Part 4 Then uses the trained word vectors to do a sentiment analysis on sentences
 in the Stanford Sentiment Treebank. This outputs confusion matrix, error graph and a text file
 with the following: true sentiment scores; predicted sentiment scores; sentence text.

![Error on training and test with against Regulization Parameter](/q4_reg_v_acc.png)
![Confusion Matrix](/q4_dev_conf.png)

example error results from sentiment output where 0 is very negatiave and 4 is very positive
Below sentence misclassifed as negative:

4	1	at a time when half the so-called real movies are little more than live-action cartoons , it 's refreshing to see a cartoon that knows what it is , and knows the form 's history .

Link to assignment: http://web.stanford.edu/class/cs224n/assignment1/assignment1.pdf
