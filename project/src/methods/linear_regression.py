import numpy as np


class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda=0, task_kind="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.w = None
        self.lmda = lmda
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """

        F = training_data.shape[1]
        L = np.eye(F) * self.lmda
        self.w = np.linalg.inv((training_data.T @ training_data) + L) @ (training_data.T @ training_labels)
        pred_regression_targets = training_data @ self.w
        return pred_regression_targets

    def predict(self, test_data):
        """
                Runs prediction on the test data.

                Arguments:
                    test_data (np.array): test data of shape (N,D)
                Returns:
                    test_labels (np.array): labels of shape (N,regression_target_size)
            """

        pred_regression_targets = test_data @ self.w

        return pred_regression_targets
