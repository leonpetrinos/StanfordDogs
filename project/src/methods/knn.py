import numpy as np


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.training_data = None
        self.training_labels = None
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.knn(training_data, training_data, training_labels, self.k, self.task_kind)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """

        test_labels = self.knn(test_data, self.training_data, self.training_labels, self.k, self.task_kind)
        return test_labels

    def euclidean_distance(self, training_data, example):
        """Compute the Euclidean distance between a single example
            vector and all training_examples.

            Inputs:
                example: shape (D,)
                training_examples: shape (NxD)
            Outputs:
                Euclidean distances: shape (N,)
            """
        distance = np.sqrt(np.sum((training_data - example) ** 2, axis=1))
        return distance

    def k_nearest_neighbours(self, k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
               Tip: use np.argsort()

           Inputs:
               k: integer
               distances: shape (N,)
           Outputs:
               indices of the k nearest neighbors: shape (k,)
           """
        indices = np.argsort(distances)[:k]
        return indices

    def predict_label(self, neighbor_labels):
        """Return the most frequent label in the neighbors'.

           Inputs:
               neighbor_labels: shape (N,)
           Outputs:
               most frequent label
           """
        return np.argmax(np.bincount(neighbor_labels))

    def knn_one_example(self, unlabeled_example, training_features, training_labels, k, task_kind):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,)
            training_features: shape (NxD)
            training_labels: shape (N,)
            k: integer
        Outputs:
            predicted label
        """
        # Compute distances
        distances = self.euclidean_distance(unlabeled_example, training_features)

        # Find neighbors
        nn_indices = self.k_nearest_neighbours(k, distances)

        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]

        # Pick the most common
        if task_kind == "classification":
            best_label = self.predict_label(neighbor_labels)
        else:
            best_label = np.mean(neighbor_labels, axis=0)

        return best_label

    def knn(self, unlabeled, training_features, training_labels, k, task_kind):
        """Return the labels vector for all unlabeled datapoints.

        Inputs:
            unlabeled: shape (MxD)
            training_features: shape (NxD)
            training_labels: shape (N,)
            k: integer
        Outputs:
            predicted labels: shape (M,)
        """

        return np.apply_along_axis(func1d=self.knn_one_example, axis=1, arr=unlabeled,
                                   training_features=training_features,
                                   training_labels=training_labels, k=k, task_kind=task_kind)
