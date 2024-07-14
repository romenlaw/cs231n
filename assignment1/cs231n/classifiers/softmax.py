from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # DONE: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):  # loop around N
      scores = X[i].dot(W)  # score is 1-dimentional with C elements
      # shift values to have maximum value of 0 to improve stability
      scores -= np.max(scores)
      correct_class_score = scores[y[i]]
      scores_sum = np.sum(np.exp(scores))
      #margin =0.0
      Li=-correct_class_score + np.log(scores_sum)
      for j in range(num_classes):  # loop around C
        pj = np.exp(scores[j]) / scores_sum
        if j==y[i] :         
          # grad Wyi = (Pyi-1)*xi
          dW[:,y[i]] += (pj-1)*X[i]
        else :
          # grad Wj = Pj*xi
          dW[:,j] += pj*X[i]
      loss += Li
    
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # DONE: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    # scores is N x C matrix
    scores -= np.max(scores, axis=1)[:, np.newaxis]
    # scores_y and scores_sum are 1-dimentional with N elements
    scores_y = scores[np.arange(num_train),y]
    scores_exp_sum = np.sum(np.exp(scores), axis=1)

    # calculate loss
    losses = np.log(scores_exp_sum) - scores_y
    loss=np.sum(losses) / num_train
    loss += reg*np.sum(W**2)

    # P=exp(scores) / scores_exp_sum, dimention is (N,C)
    # grad Wj = Pj * xi
    # grad Wyi = (Pyi-1) * xi
    P=np.exp(scores) / scores_exp_sum[:, np.newaxis]
    P[np.arange(num_train), y] -= 1 
    dW += X.T.dot(P)
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
