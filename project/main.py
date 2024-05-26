import argparse

import numpy as np
from matplotlib import pyplot as plt
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
import time
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz', allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest = feature_data['xtrain'], feature_data['xtest'], \
            feature_data['ytrain'], feature_data['ytest'], feature_data['ctrain'], feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, 'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    mean = np.mean(xtrain, axis=0, keepdims=True)
    std = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std)

    if args.method in {"linear_regression", "logistic_regression"}:
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        validation_percentage = 0.2
        N = xtrain.shape[0]
        num_elements = int(N * validation_percentage)
        indices = np.arange(N)

        ### Shuffling the elements
        np.random.permutation(indices)

        ### validation indices & training_indices
        valid_ind = indices[:num_elements]  # 20%
        train_ind = indices[num_elements:]  # 80%

        x_train_copy = np.copy(xtrain)
        y_train_copy = np.copy(ytrain)
        c_train_copy = np.copy(ctrain)

        xtrain, xtest = xtrain[train_ind], x_train_copy[valid_ind]
        ytrain, ytest = ytrain[train_ind], y_train_copy[valid_ind]
        ctrain, ctest = ctrain[train_ind], c_train_copy[valid_ind]

    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.
    if not args.validation:
        # Use NN (FOR MS2!)
        if args.method == "nn":
            raise NotImplementedError("This will be useful for MS2.")

        # Follow the "DummyClassifier" example for your methods
        if args.method == "dummy_classifier":
            method_obj = DummyClassifier(arg1=1, arg2=2)

        elif args.method == "linear_regression":
            method_obj = LinearRegression(lmda=args.lmda)

        elif args.method == "logistic_regression":
            method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

        elif args.method == "knn":
            if args.task == "center_locating":
                method_obj = KNN(k=args.K, task_kind="regression")
            else:
                method_obj = KNN(k=args.K, task_kind="classification")

        else:
            raise NotImplementedError("no method of this name exists")

        ## 4. Train and evaluate the method
        if args.task == "center_locating":
            s1 = time.time()
            # Fit parameters on training data

            preds_train = method_obj.fit(xtrain, ctrain)

            # Perform inference for training and test data
            train_pred = method_obj.predict(xtrain)
            preds = method_obj.predict(xtest)
            s2 = time.time()

            ## Report results: performance on train and valid/test sets
            train_loss = mse_fn(train_pred, ctrain)
            loss = mse_fn(preds, ctest)

            print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")
            if args.time:
                print("This function takes", s2 - s1, "seconds")

        elif args.task == "breed_identifying":
            s1 = time.time()

            # Fit (:=train) the method on the training data for classification task
            preds_train = method_obj.fit(xtrain, ytrain)

            # Predict on unseen data
            preds = method_obj.predict(xtest)
            s2 = time.time()

            ## Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
            if args.time:
                print("This function takes", s2 - s1, "seconds")

        else:
            raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    else: ## IF WE ARE DOING OUR CROSS VALIDATION FOR A PARAM-LIST

        ########################################################################################################
        ########################################################################################################

        ## Here we calculate the best hyperparameters
        train_accuracy, test_accuracy = [], []
        train_f1, test_f1 = [], []
        train_mse_loss, test_mse_loss = [], []
        param_list = []
        if args.method == "linear_regression":
            param_list = np.linspace(start=0, stop=2, num=50)  # lambda
            for param in param_list:
                method_obj = LinearRegression(lmda=param)

                preds_train = method_obj.fit(xtrain, ctrain)

                train_pred = method_obj.predict(xtrain)
                preds = method_obj.predict(xtest)

                train_loss = mse_fn(train_pred, ctrain)
                loss = mse_fn(preds, ctest)

                train_mse_loss.append(train_loss)
                test_mse_loss.append(loss)

        elif args.method == "logistic_regression":
            param_list = np.linspace(start=10e-5, stop=0.1, num=100)  # learning rate
            for param in param_list:
                method_obj = LogisticRegression(lr=param, max_iters=500)

                preds_train = method_obj.fit(xtrain, ytrain)
                preds = method_obj.predict(xtest)

                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                train_accuracy.append(acc)
                train_f1.append(macrof1)

                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                test_accuracy.append(acc)
                test_f1.append(macrof1)

        elif args.method == "knn":
            param_list = [i for i in range(1, 100)]
            if args.task == "center_locating":
                for param in param_list:
                    method_obj = KNN(k=param, task_kind="regression")
                    preds_train = method_obj.fit(xtrain, ctrain)

                    train_pred = method_obj.predict(xtrain)
                    preds = method_obj.predict(xtest)

                    train_loss = mse_fn(train_pred, ctrain)
                    loss = mse_fn(preds, ctest)

                    train_mse_loss.append(train_loss)
                    test_mse_loss.append(loss)

            else:
                for param in param_list:
                    method_obj = KNN(k=param, task_kind="classification")

                    preds_train = method_obj.fit(xtrain, ytrain)
                    preds = method_obj.predict(xtest)

                    acc = accuracy_fn(preds_train, ytrain)
                    macrof1 = macrof1_fn(preds_train, ytrain)
                    train_accuracy.append(acc)
                    train_f1.append(macrof1)

                    acc = accuracy_fn(preds, ytest)
                    macrof1 = macrof1_fn(preds, ytest)
                    test_accuracy.append(acc)
                    test_f1.append(macrof1)

        ########################################################################################################
        ########################################################################################################

        ## Now we can plot our graph
        if args.task == "breed_identifying":
            plt.figure(figsize=(9, 6))
            plt.title(f"Performance on the validation set for different values of hyperparameter in {args.method}")
            plt.plot(param_list, test_accuracy, marker='o', linestyle='-', color='b', label='Test Accuracy')
            plt.xlabel("hyperparameter")
            plt.ylabel("Accuracy")
            plt.grid(True)  # Show grid lines

            # Set x-ticks and scale
            if args.method == "logistic_regression":
                xticks = np.arange(1e-5, 0.15, 1e-2)
                rounded_xticks = [round(x, 6) for x in xticks]
                plt.xticks(xticks, rounded_xticks)
            elif args.method == "knn":
                plt.xticks(range(1, len(param_list), 5))

            max_acc = max(test_accuracy)
            min_acc = min(test_accuracy)
            max_param = param_list[test_accuracy.index(max_acc)]
            min_param = param_list[test_accuracy.index(min_acc)]

            # Mark max and min accuracy points on the plot
            plt.scatter(max_param, max_acc, color='r', label=f'Max Accuracy: ({round(max_param, 6)}, {round(max_acc, 3)})', zorder=5)
            plt.scatter(min_param, min_acc, color='g', label=f'Min Accuracy: ({round(min_param, 6)}, {round(min_acc, 3)})', zorder=5)

            plt.legend()  # Add legend
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(9, 6))
            plt.title(
                f"Performance on the validation set for different values of hyperparameter in {args.method}")
            plt.plot(param_list, test_mse_loss, marker='o', linestyle='-', color='b', label='Test MSE Loss')
            plt.xlabel("hyperparameter")
            plt.ylabel("MSE Loss")
            plt.grid(True)  # Show grid lines

            # Set x-ticks and scale
            if args.method == "knn":
                plt.xticks(range(0, len(param_list), 5))  # Show ticks every 5 values
            elif args.method == "linear_regression":
                plt.xticks(range(0, len(param_list)))  # Show ticks every 5 values

            min_loss = min(test_mse_loss)
            max_loss = max(test_mse_loss)
            min_param = param_list[test_mse_loss.index(min_loss)]
            max_param = param_list[test_mse_loss.index(max_loss)]

            # Mark min and max loss points on the plot
            plt.scatter(min_param, min_loss, color='g', label=f'Min MSE Loss: ({round(min_param, 6)}, {round(min_loss, 6)})', zorder=5)
            plt.scatter(max_param, max_loss, color='r', label=f'Max MSE Loss: ({round(max_param, 6)}, {round(max_loss, 6)})', zorder=5)

            plt.legend()  # Add legend
            plt.tight_layout()
            plt.show()

        ########################################################################################################
        ########################################################################################################


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str,
                        help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--validation', action="store_true")
    parser.add_argument('--time', action="store_true")
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
