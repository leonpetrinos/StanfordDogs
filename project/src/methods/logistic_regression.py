import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr=1, max_iters=500, task_kind="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.w = None  # weights obtained from gradient descent (D x C array)
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        C = get_n_classes(training_labels) # number of classes
        D = training_data.shape[1]  # number of features
        labels = label_to_onehot(training_labels)  # labels in one hot encoding of shape (N, C)
        
        # Random initialization of the weights
        self.w = np.random.normal(0, 0.1, (D, C))
        for i in range(self.max_iters):
            self.w = self.w - self.lr * self.gradient_logistic(training_data, labels, self.w)
        pred_labels = self.logistic_regression_predict(training_data, self.w)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        pred_labels = self.logistic_regression_predict(test_data, self.w)
        return pred_labels

    def f_softmax(self, data, w):
        """
        Softmax function for multi-class logistic regression.

        Arguments:
            data (array): Input data of shape (N, D)
            w (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """

        top = np.exp(data @ w)  # nominator (shape (N, C))
        bottom = np.sum(top, axis=1)  # denominator (shape (N,))
        return top / bottom[:, np.newaxis]  # now denominator is (N, 1)

    def gradient_logistic(self, data, labels, w):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        Arguments:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """

        return data.T @ (self.f_softmax(data, w) - labels)

    def logistic_regression_predict(self, data, w):
        """
        Prediction the label of data for multi-class logistic regression.

        Arguments:
            data (array): Dataset of shape (N, D).
            w (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """

        return onehot_to_label(self.f_softmax(data, w))
